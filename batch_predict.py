# ============================================================
# batch_predict.py
# Reads a CSV of soil layers (e.g. PlaxisInput_Standardized.csv),
# predicts phi, sp_gr, cohesion, mc for every row using the SAME
# models/feature-engineering as tool.py (via prediction_engine.py),
# and writes the result as a new CSV/Excel with Predicted_<target>
# columns appended.
#
# Uses the exact same load_bundle()/build_features()/predict() as
# the Streamlit app, so batch results are guaranteed identical to
# what tool.py would give for the same single-row input.
#
# CHANGES (this revision):
#   1. TARGETS now comes straight from prediction_engine.py, which is
#      synced to the 4 real trained targets (gamma_wet/gamma_dry were
#      never trained -- see prediction_engine.py).
#   2. NEW: rows with soil_type = "Clay(CL)" get Predicted_phi and
#      Predicted_cohesion left blank (the training dataset had no lab
#      values for those two on Clay(CL) -- see
#      UNSUPPORTED_TARGET_SOIL_COMBINATIONS in prediction_engine.py).
#      This is expected and reported separately in the console summary
#      below, NOT counted as a row error -- Predicted_sp_gr and
#      Predicted_mc are still filled in normally for those rows.
#   3. FIX: renamed n60 -> nfield everywhere, matching the corrected
#      prediction_engine.py (predict()/build_features() now take
#      `nfield`, the raw field SPT value, not N60). INPUT_COLUMN_MAP
#      key updated to "nfield" to match.
#   4. INPUT_PATH set to the file to predict on.
#   5. FIX (this patch): INPUT_COLUMN_MAP's right-hand values were
#      placeholders ("Depth_m", "GWT_m", "Layer_Thickness") that never
#      matched the real column names produced by Step 1 / soil_grouping.py
#      ("Depth(m)", "GWT(m)", "Layer Thickness(m)", "Soil_Type_Tool").
#      Updated to the exact column names present in
#      PlaxisInput_Standardized.csv.
# ============================================================

import os
import pandas as pd

from prediction_engine import (
    predict_all_targets,
    TARGETS,
    SOIL_TYPES,
    preload_all_bundles,
    UNSUPPORTED_TARGET_SOIL_COMBINATIONS,
)

# ------------------------------------------------------------------
# CONFIG -- edit these to match your actual file
# ------------------------------------------------------------------
INPUT_PATH = r"E:\Plaxis\ready_predictNow.csv"
OUTPUT_CSV_PATH = r"E:\Plaxis\Predicted_Output.csv"
OUTPUT_XLSX_PATH = r"E:\Plaxis\Predicted_Output.xlsx"

# ⭐ Map the engine's expected field names to YOUR CSV's actual column
# headers. Only the values (right side) need to change -- edit these
# to match PlaxisInput_Standardized.csv exactly.
#
# ⚠️ "nfield" below must point to whatever column in your CSV holds the
# raw field SPT blow count (previously called N60 in older versions of
# this script). Check your actual header name and update the value on
# the right if it isn't "Nfield".
INPUT_COLUMN_MAP = {
    "depth_m": "Depth(m)",
    "nfield": "Nfield",
    "gwt_m": "GWT(m)",
    "layer_thickness_m": "Layer Thickness(m)",
    "soil_type": "Soil_Type_Tool",
}


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    missing_cols = [c for c in INPUT_COLUMN_MAP.values() if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"These columns from INPUT_COLUMN_MAP were not found in {INPUT_PATH}: "
            f"{missing_cols}\nActual columns in the file: {list(df.columns)}\n"
            f"Edit INPUT_COLUMN_MAP at the top of this script to match your file."
        )

    print(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print("Preloading models (once)...")
    preload_all_bundles()

    predicted_cols = {t: [] for t in TARGETS}
    error_rows = []
    clay_cl_rows = 0

    for idx, row in df.iterrows():
        depth_m = float(row[INPUT_COLUMN_MAP["depth_m"]])
        nfield = float(row[INPUT_COLUMN_MAP["nfield"]])
        gwt_m = float(row[INPUT_COLUMN_MAP["gwt_m"]])
        layer_thickness_m = float(row[INPUT_COLUMN_MAP["layer_thickness_m"]])
        soil_type = str(row[INPUT_COLUMN_MAP["soil_type"]]).strip()

        if soil_type not in SOIL_TYPES:
            error_rows.append((idx, soil_type))
            for t in TARGETS:
                predicted_cols[t].append(None)
            continue

        if soil_type == "Clay(CL)":
            clay_cl_rows += 1

        try:
            # predict_all_targets() already returns None (not an error) for
            # any target blocked by UNSUPPORTED_TARGET_SOIL_COMBINATIONS,
            # e.g. phi/cohesion on Clay(CL) -- so sp_gr/mc are still filled
            # in normally for those rows.
            preds = predict_all_targets(depth_m, nfield, gwt_m, layer_thickness_m, soil_type)
            for t in TARGETS:
                predicted_cols[t].append(preds[t])
        except Exception as e:
            error_rows.append((idx, f"prediction error: {e}"))
            for t in TARGETS:
                predicted_cols[t].append(None)

        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(df)} rows done")

    for t in TARGETS:
        df[f"Predicted_{t}"] = predicted_cols[t]

    if error_rows:
        print(f"\n WARNING: {len(error_rows)} row(s) failed and were left blank:")
        for idx, reason in error_rows[:20]:
            print(f"  Row {idx}: {reason}")
        if len(error_rows) > 20:
            print(f"  ...and {len(error_rows) - 20} more.")
        print(f"  Most common cause: soil_type value doesn't exactly match one of {SOIL_TYPES}")

    if clay_cl_rows:
        blocked_targets = sorted({
            target for (target, soil) in UNSUPPORTED_TARGET_SOIL_COMBINATIONS if soil == "Clay(CL)"
        })
        blocked_cols = ", ".join(f"Predicted_{t}" for t in blocked_targets)
        print(
            f"\nℹ️ {clay_cl_rows} row(s) have soil_type = 'Clay(CL)'. For these rows, "
            f"{blocked_cols} were intentionally left blank because the training "
            f"dataset had no lab values for {', '.join(blocked_targets)} on Clay(CL). "
            f"This is not an error -- the other predicted columns for these rows are "
            f"still filled in normally."
        )

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    try:
        df.to_excel(OUTPUT_XLSX_PATH, index=False)
    except Exception as e:
        print(f"(Excel export skipped -- install openpyxl if needed: {e})")

    print(f"\nDone. {len(df) - len(error_rows)}/{len(df)} rows predicted successfully.")
    print(f"Saved -> {OUTPUT_CSV_PATH}")
    print(f"Saved -> {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()