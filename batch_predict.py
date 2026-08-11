# ============================================================
# batch_predict.py
# Reads a CSV of soil layers (e.g. PlaxisInput_Standardized.csv),
# predicts phi, gamma_wet, gamma_dry, sp_gr, cohesion, mc for every
# row using the SAME models/feature-engineering as tool.py (via
# prediction_engine.py), and writes the result as a new CSV/Excel
# with Predicted_<target> columns appended.
#
# Uses the exact same load_bundle()/build_features()/predict() as
# the Streamlit app, so batch results are guaranteed identical to
# what tool.py would give for the same single-row input.
# ============================================================

import os
import pandas as pd

from prediction_engine import predict_all_targets, TARGETS, SOIL_TYPES, preload_all_bundles

# ------------------------------------------------------------------
# CONFIG -- edit these to match your actual file
# ------------------------------------------------------------------
INPUT_PATH = r"E:\Plaxis\PlaxisInput_Final.csv"
OUTPUT_CSV_PATH = r"E:\Plaxis\Predicted_Output.csv"
OUTPUT_XLSX_PATH = r"E:\Plaxis\Predicted_Output.xlsx"

# ⭐ Map the engine's expected field names to YOUR CSV's actual column
# headers. Only the values (right side) need to change -- edit these
# to match PlaxisInput_Standardized.csv exactly.
INPUT_COLUMN_MAP = {
    "depth_m": "depth_m",
    "n60": "N60",
    "gwt_m": "GWT_m",
    "layer_thickness_m": "Thickness_m",
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

    for idx, row in df.iterrows():
        depth_m = float(row[INPUT_COLUMN_MAP["depth_m"]])
        n60 = float(row[INPUT_COLUMN_MAP["n60"]])
        gwt_m = float(row[INPUT_COLUMN_MAP["gwt_m"]])
        layer_thickness_m = float(row[INPUT_COLUMN_MAP["layer_thickness_m"]])
        soil_type = str(row[INPUT_COLUMN_MAP["soil_type"]]).strip()

        if soil_type not in SOIL_TYPES:
            error_rows.append((idx, soil_type))
            for t in TARGETS:
                predicted_cols[t].append(None)
            continue

        try:
            preds = predict_all_targets(depth_m, n60, gwt_m, layer_thickness_m, soil_type)
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
