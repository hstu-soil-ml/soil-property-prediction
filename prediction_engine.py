# ============================================================
# prediction_engine.py
# Shared model-loading, feature-engineering, and prediction logic.
# Used by BOTH tool.py (Streamlit UI) and batch_predict.py (batch CSV).
# Deliberately has NO Streamlit dependency, so it works identically
# inside or outside a running Streamlit app -- model loading is
# cached with a plain in-memory dict instead of @st.cache_resource.
#
# CHANGES (this revision):
#   1. FIX: TARGETS previously listed "gamma_wet" and "gamma_dry", which
#      were never trained anywhere in the Step 1-3 pipeline (Step 1 drops
#      gamma_sat/gamma_unsat/void_ratio as FEATURES, and Step 2/3's real
#      TARGETS were always just phi, sp_gr, cohesion, mc). Leaving them in
#      here meant predict() would always crash trying to load a
#      "gamma_wet_best_model.pkl" that Step 3 never produces. TARGETS now
#      lists exactly the 4 real targets.
#   2. NEW: Clay(CL) has NO laboratory phi or cohesion values anywhere in
#      the training dataset, so those two predictions for Clay(CL) would
#      be pure extrapolation, not interpolation. predict() now blocks
#      exactly that combination (phi+Clay(CL), cohesion+Clay(CL)) and
#      raises UnsupportedPredictionError with a clear reason instead of
#      silently returning a number. Every OTHER (target, soil_type)
#      combination is unaffected -- sp_gr and mc still predict normally
#      for Clay(CL), and phi/cohesion still predict normally for every
#      other soil type.
#   3. FIX: build_features() had two bugs that produced silently wrong
#      predictions:
#        a) parameter was named `n60` but the body referenced an
#           undefined `Nfield` variable -> NameError on every call.
#        b) even after fixing the name, the feature dict keys
#           ("depth_m", "gwt_m", "layer_thickness_m", "depth_x_Nfield",
#           "Nfield_over_depth") did NOT match the real Step 2
#           BASE_FEATURES column names ("Depth(m)", "GWT(m)",
#           "Layer Thickness(m)", "Depth(m)_x_Nfield",
#           "Nfield_over_Depth(m)"). Because the final DataFrame is
#           built with `row.get(c, 0)` against feature_order, every
#           mismatched key silently defaulted to 0 instead of raising
#           an error -- so predictions ran without crashing, but with
#           Depth/GWT/Layer Thickness/engineered features zeroed out.
#      Both are fixed below: the parameter is renamed `nfield`
#      (matches tool.py's variable name) and every dict key now
#      matches the exact Step 2 BASE_FEATURES column names.
#   4. FIX (this patch, synced with Step 3 v2.0): predict() previously
#      called model.predict(X.values) and manually applied an external
#      scaler.transform() before predicting. Since Step 3 v2.0, "model"
#      is a full sklearn Pipeline (imputer [+ scaler] + estimator)
#      fit end-to-end on raw features -- it imputes/scales internally.
#      The manual scaler step is now REMOVED (it was redundant and, if
#      any older .pkl ever had a non-None scaler, risked silently
#      double-scaling the input). X is now passed straight into
#      model.predict(X) as a DataFrame (not X.values), so the Pipeline
#      sees the same column names it was trained with -- this also
#      removes the harmless-but-noisy "X does not have valid feature
#      names" sklearn UserWarning seen during batch_predict.py runs,
#      and removes reliance on column ORDER alone with no safety net.
# ============================================================

import os
import json
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Synced to Step 2/3's actual trained targets (safe-name order).
TARGETS = ["phi", "sp_gr", "cohesion", "mc"]

SOIL_TYPES = [
    "Sand(SM)",
    "Silty Sand(SM)",
    "Clay(CL)",
    "Silty Clay(CL)",
    "Sand-Other",
    "Clay-Other",
    "Silt-Other",
]
SOIL_DUMMY_COLS = [f"soil_{s}" for s in SOIL_TYPES]

MODEL_MAP_CANDIDATES = [
    os.path.join(MODEL_DIR, "tool_best_model_map.json"),
    os.path.join(BASE_DIR, "tool_best_model_map.json"),
    os.path.join(BASE_DIR, "Paper_Final_Slim", "tool_best_model_map.json"),
]


# ------------------------------------------------------------------
# TRAINING-DATA COVERAGE GAPS
# ------------------------------------------------------------------
class UnsupportedPredictionError(Exception):
    """Raised when a (target, soil_type) pair has no lab-tested training
    support, so a prediction would be an unreliable extrapolation rather
    than an interpolation within the training data."""
    pass


# The Dinajpur ML training dataset contained NO laboratory Angle of
# Friction (phi) or Cohesion values for Clay(CL) samples. This gap is
# limited to exactly these two (target, soil_type) pairs -- nothing else
# is restricted. Specific Gravity and Moisture Content ARE supported for
# Clay(CL), and phi/cohesion ARE supported for every other soil type.
UNSUPPORTED_TARGET_SOIL_COMBINATIONS = {
    ("phi", "Clay(CL)"): (
        "The training dataset contained no laboratory Angle of Friction (ϕ) "
        "values for Clay(CL) samples, so this model cannot reliably predict "
        "phi for Clay(CL)."
    ),
    ("cohesion", "Clay(CL)"): (
        "The training dataset contained no laboratory Cohesion values for "
        "Clay(CL) samples, so this model cannot reliably predict cohesion "
        "for Clay(CL)."
    ),
}


def is_supported(target: str, soil_type: str) -> bool:
    """True unless this exact (target, soil_type) pair is a known
    training-data coverage gap (currently: phi and cohesion on Clay(CL))."""
    return (target, soil_type) not in UNSUPPORTED_TARGET_SOIL_COMBINATIONS


# ------------------------------------------------------------------
# MODEL LOADING -- plain dict cache, works in any Python process
# ------------------------------------------------------------------
_MODEL_MAP_CACHE = None
_BUNDLE_CACHE = {}


def load_model_map():
    global _MODEL_MAP_CACHE
    if _MODEL_MAP_CACHE is not None:
        return _MODEL_MAP_CACHE

    for path in MODEL_MAP_CANDIDATES:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _MODEL_MAP_CACHE = json.load(f)
                return _MODEL_MAP_CACHE

    _MODEL_MAP_CACHE = {t: f"{t}_best_model.pkl" for t in TARGETS}
    return _MODEL_MAP_CACHE


def load_bundle(target: str):
    """Loads (and caches, once per process) the model bundle for a target."""
    if target in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[target]

    model_map = load_model_map()
    entry = model_map.get(target, f"{target}_best_model.pkl")
    filename = entry["pkl_file"] if isinstance(entry, dict) else entry
    path = os.path.join(MODEL_DIR, filename)

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    _BUNDLE_CACHE[target] = bundle
    return bundle


def preload_all_bundles():
    """Loads every target's model once. Call this before a batch loop so
    the first row of the loop isn't slower than the rest.
    """
    for t in TARGETS:
        load_bundle(t)


# ------------------------------------------------------------------
# FEATURE ENGINEERING
# Keys below MUST exactly match Step 2's BASE_FEATURES column names,
# since the final DataFrame is built with `row.get(c, 0)` against
# feature_order -- any mismatch silently zeroes that feature instead
# of raising an error.
# ------------------------------------------------------------------
def build_features(depth_m, nfield, gwt_m, layer_thickness_m, soil_type, feature_order):
    row = {
        "Nfield": nfield,
        "Depth(m)": depth_m,
        "GWT(m)": gwt_m,
        "Layer Thickness(m)": layer_thickness_m,
        "Nfield_log": np.log1p(max(nfield, 0.0)),
        "Nfield_sqrt": np.sqrt(max(nfield, 0.0)),
        "Depth(m)_x_Nfield": depth_m * nfield,
        "Nfield_over_Depth(m)": nfield / (depth_m + 0.1),
    }
    for col in SOIL_DUMMY_COLS:
        row[col] = 0
    key = f"soil_{soil_type}"
    if key in SOIL_DUMMY_COLS:
        row[key] = 1
    else:
        raise ValueError(
            f"Unknown soil_type '{soil_type}'. Must exactly match one of: {SOIL_TYPES}"
        )

    missing = [c for c in feature_order if c not in row]
    if missing:
        raise KeyError(
            f"build_features() is missing model-expected feature(s): {missing}. "
            f"This means feature_order (from the saved model bundle) no longer "
            f"matches what build_features() produces -- check for a Step 2 "
            f"feature-name change."
        )

    return pd.DataFrame([[row[c] for c in feature_order]], columns=feature_order)


# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
def predict(target, depth_m, nfield, gwt_m, layer_thickness_m, soil_type):
    # Block known training-data coverage gaps BEFORE touching any model
    # (currently: phi and cohesion on Clay(CL) only -- see
    # UNSUPPORTED_TARGET_SOIL_COMBINATIONS above).
    reason = UNSUPPORTED_TARGET_SOIL_COMBINATIONS.get((target, soil_type))
    if reason is not None:
        raise UnsupportedPredictionError(reason)

    bundle = load_bundle(target)
    model = bundle["model"]
    feature_order = bundle["features"]

    X = build_features(depth_m, nfield, gwt_m, layer_thickness_m, soil_type, feature_order)

    # NOTE (synced with Step 3 v2.0): "model" is now a full sklearn
    # Pipeline (imputer [+ scaler] + estimator) fit end-to-end on raw
    # features. It imputes/scales internally, fold-consistent with how
    # it was trained -- so we must NOT apply any external scaler here.
    # (bundle["scaler"] is intentionally None in v2.0 .pkl files; the old
    # manual scaler.transform() step has been removed to avoid
    # double-scaling if an older .pkl ever had a non-None scaler.)
    #
    # X is kept as a DataFrame (not X.values) so the Pipeline sees the
    # same column names it was fit with -- passing a bare numpy array
    # triggers sklearn's "X does not have valid feature names"
    # UserWarning and relies on column ORDER matching exactly with no
    # safety net if that order ever drifts.
    return float(model.predict(X)[0])


def predict_all_targets(depth_m, nfield, gwt_m, layer_thickness_m, soil_type):
    """Convenience wrapper -- predicts all targets for one soil layer,
    returns a dict {target: value}. Used by both the UI and the batch script.

    Targets blocked by UNSUPPORTED_TARGET_SOIL_COMBINATIONS (currently
    phi/cohesion on Clay(CL)) come back as None instead of raising, so a
    single soil layer never fails outright just because one property has
    no training support -- the other targets are still predicted normally.
    """
    out = {}
    for target in TARGETS:
        try:
            out[target] = predict(target, depth_m, nfield, gwt_m, layer_thickness_m, soil_type)
        except UnsupportedPredictionError:
            out[target] = None
    return out