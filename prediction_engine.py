# ============================================================
# prediction_engine.py
# Shared model-loading, feature-engineering, and prediction logic.
# Used by BOTH tool.py (Streamlit UI) and batch_predict.py (batch CSV).
# Deliberately has NO Streamlit dependency, so it works identically
# inside or outside a running Streamlit app -- model loading is
# cached with a plain in-memory dict instead of @st.cache_resource.
# ============================================================

import os
import json
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGETS = ["phi", "gamma_wet", "gamma_dry", "sp_gr", "cohesion", "mc"]

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
# FEATURE ENGINEERING -- identical to tool.py's build_features()
# ------------------------------------------------------------------
def build_features(depth_m, n60, gwt_m, layer_thickness_m, soil_type, feature_order):
    row = {
        "n60": n60,
        "depth_m": depth_m,
        "gwt_m": gwt_m,
        "layer_thickness_m": layer_thickness_m,
        "n60_sqrt": np.sqrt(max(n60, 0.0)),
        "n60_log": np.log1p(max(n60, 0.0)),
        "depth_x_n60": depth_m * n60,
        "n60_over_depth": n60 / (depth_m + 0.1),
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

    return pd.DataFrame([[row.get(c, 0) for c in feature_order]], columns=feature_order)


# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
def predict(target, depth_m, n60, gwt_m, layer_thickness_m, soil_type):
    bundle = load_bundle(target)
    model = bundle["model"]
    feature_order = bundle["features"]
    needs_scale = bundle.get("needs_scale", False)
    scaler = bundle.get("scaler", None)

    X = build_features(depth_m, n60, gwt_m, layer_thickness_m, soil_type, feature_order)

    if needs_scale and scaler is not None:
        num_cols = bundle.get("num_cols") or bundle.get("numerical_features") or []
        num_cols = [c for c in num_cols if c in X.columns]
        if num_cols:
            X.loc[:, num_cols] = scaler.transform(X[num_cols])

    return float(model.predict(X.values)[0])


def predict_all_targets(depth_m, n60, gwt_m, layer_thickness_m, soil_type):
    """Convenience wrapper -- predicts all 6 targets for one soil layer,
    returns a dict {target: value}. Used by both the UI and the batch script.
    """
    out = {}
    for target in TARGETS:
        out[target] = predict(target, depth_m, n60, gwt_m, layer_thickness_m, soil_type)
    return out
