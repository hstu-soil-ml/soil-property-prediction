# ============================================================
# Geotechnical Soil Property Prediction Tool
# Version 2.0 | August 2026
# BSc Thesis — Department of Civil Engineering, HSTU
#
# Model loading, feature engineering, and prediction now live in
# prediction_engine.py (shared with batch_predict.py) so a fix in
# one place applies to both the interactive tool and any future
# batch/automation use.
#
# CHANGE LOG (v2.0):
#   - Feature/target names aligned with Step 1 frozen dataset:
#       Nfield, Depth(m), GWT(m), Layer Thickness(m),
#       Angle of Friction "phi", Specific Gravity "sp_gr",
#       Moisture Content "mc", Cohesion "cohesion"
#   - gamma_wet, gamma_dry, void_ratio removed (dropped from Step 1)
#   - N60 input replaced with raw field SPT value, Nfield
#     (no more energy-correction note; matches Step 1/2 pipeline)
#   - No more separate model override for Cohesion — every target
#     now uses whichever model was selected as "best" in Step 4/5
#   - New limitation handling: Clay(CL) has no valid phi/cohesion
#     training samples, so the tool only predicts Specific Gravity
#     and Moisture Content for that soil type
# ============================================================

import os
import streamlit as st
from PIL import Image

from prediction_engine import (
    TARGETS,
    SOIL_TYPES,
    predict,
)

# ── Page config (সবার আগে)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "hstu_logo.png")

icon = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="Geotechnical Soil Property Prediction Tool",
    page_icon=icon,
    layout="centered",
)

# ============================================================
# CONSTANTS (UI-only -- model logic lives in prediction_engine.py)
# ============================================================
TARGET_LABELS = {
    "phi":      "Friction Angle, φ",
    "sp_gr":    "Specific Gravity, Gs",
    "mc":       "Moisture Content, w",
    "cohesion": "Cohesion, c",
}
TARGET_UNITS = {
    "phi":      "°",
    "sp_gr":    "—",
    "mc":       "%",
    "cohesion": "kPa",
}

# Observed ranges in the ML training dataset (Step 1 frozen output)
OBSERVED_TRAINING_RANGES = {
    "depth_m": (1.50, 28.96),
    "nfield":  (1.00, 100),
    "gwt_m":   (0.76, 10.67),
}

# Observed ranges of each predicted target in the ML training dataset
OUTPUT_RANGES = {
    "phi":      (11.00, 45.00),
    "sp_gr":    (2.19, 3.13),
    "mc":       (10.31, 40.75),
    "cohesion": (1.96, 49.03),
}

# ── Soil types with no valid phi/cohesion training samples.
#    For these, only Specific Gravity and Moisture Content are predicted.
LIMITED_DATA_SOIL_TYPES = {"Clay(CL)"}
LIMITED_TARGETS = {"phi", "cohesion"}

# ============================================================
# CUSTOM CSS — Clean, Professional Academic Theme
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F5F7FA; }

    /* Header card */
    .header-card {
    background: linear-gradient(135deg, #1C5D99 0%, #2E86C1 100%);
    border-radius: 14px;
    padding: 28px;
    display: flex;
    align-items: center;
    box-shadow: 0 10px 25px rgba(0,0,0,.12);
    margin-bottom: 28px;
}

.header-text {
    width: 100%;
}

.header-text h2 {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: .3px;
    margin: 0 0 8px 0;
}

.header-text p {
    color: #DCE7F2;
    font-size: 15px;
    line-height: 1.7;
    margin: 0;
}

    /* Input Card Form */
    div[data-testid="stForm"] {
        background: white;
        padding: 28px;
        border-radius: 14px;
        border: 1px solid #DCE3EA;
        box-shadow:0 8px 24px rgba(15,53,87,.12);
    }

    /* Section titles */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F3557;
        border-left: 5px solid #D4AF37;
        padding-left: 14px;
        margin: 28px 0 18px 0;
    }

    /* Input Box */
    .stNumberInput input {
        border-radius: 8px;
    }
    .stSelectbox {
        border-radius: 8px;
    }

    /* Predict button */
    div.stButton > button {
        background: #0F3557;
        color: white;
        font-size: 17px;
        font-weight: 600;
        height: 54px;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin-top: 8px;
        transition: background 0.2s;
    }
    div.stButton > button:hover {
        background: #174B77;
    }

    /* Result table */
    .result-table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,.08);
        margin-top: 8px;
    }
    .result-table th {
        background: #0F3557;
        color: white;
        padding: 14px;
        font-size: 15px;
        text-align: left;
        font-weight: 600;
    }
    .result-table td {
        padding: 13px;
        border-bottom: 1px solid #E8EEF5;
        color: #1C2833;
    }
    .result-table tr:nth-child(even) td {
        background: #F8FAFC;
    }
    .result-table tr:hover td {
        background: #EDF5FD;
    }
    .value-cell {
        font-weight: 700;
        color: #1F5F99;
        font-size: 17px;
    }

    /* Note box */
    .note-box {
        background: #EEF6FF;
        border-left: 5px solid #1F5F99;
        border-radius: 10px;
        padding: 18px;
        line-height: 1.8;
        font-size: 14px;
        color: #1C2833;
        margin: 24px 0 18px 0;
    }

    /* Warning box */
    .warn-box {
        background: #FFF8E6;
        border-left: 5px solid #D4AF37;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #7D6608;
        margin: 8px 0;
    }

    /* Limitation box */
    .limit-box {
        background: #FDEEEE;
        border-left: 5px solid #C0392B;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #7B241C;
        margin: 8px 0;
    }

    /* Footer */
    .footer-premium {
    background: linear-gradient(135deg,#F8FBFF 0%,#EEF6FF 100%);
    border: 1px solid #D9E6F2;
    border-top: 5px solid #1F5F99;
    border-radius: 14px;
    padding: 28px 32px 22px;
    margin-top: 28px;
    box-shadow: 0 8px 22px rgba(31,95,153,.10);
}

.footer-columns{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:40px;
}

.footer-left{flex:1;text-align:left;}
.footer-right{flex:1;text-align:right;}

.footer-name{line-height:1.9;color:#0F3557;}
.footer-role{font-style:italic;color:#667085;}

.footer-department{
    text-align:center;
    font-style:italic;
    color:#667085;
    margin:18px 0 8px;
}

.footer-divider{
    border:none;
    border-top:1px solid #DCE6EF;
    margin:12px 0;
}
.footer-title {
        color: #D4AF37;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .footer-version {
        text-align: center;
        color: #6B7280;
        font-size: 13px;
        line-height: 1.8;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-card">
<div class="header-text">
<h2>Geotechnical Soil Property Prediction Tool</h2>
<p>
Machine Learning-Based Prediction System<br>
Department of Civil Engineering<br>
Hajee Mohammad Danesh Science and Technology University (HSTU)
</p>
</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# VALIDATION HELPERS (UI-only -- decides what warnings to show)
# ============================================================
def check_inputs(depth_m, nfield, gwt_m, layer_thickness_m):
    warns = []
    for name, val, label in [
        ("depth_m", depth_m, "Depth"),
        ("nfield",  nfield,  "Nfield"),
        ("gwt_m",   gwt_m,   "GWT"),
    ]:
        lo, hi = OBSERVED_TRAINING_RANGES[name]
        if val < lo or val > hi:
            warns.append(
                f"⚠️ {label} = {val} is outside the observed training-data range "
                f"({lo} – {hi}). Prediction is outside the observed training-data range and should be interpreted with engineering judgment."
            )
    if depth_m < layer_thickness_m:
        warns.append(
            f"⚠️ Depth ({depth_m} m) is less than Layer Thickness "
            f"({layer_thickness_m} m). Please verify inputs."
        )
    return warns

def check_outputs(results):
    warns = []
    for key, (lo, hi) in OUTPUT_RANGES.items():
        val = results.get(key)
        if val is None:
            continue
        if val < lo:
            warns.append(
                f"⚠️ Predicted {TARGET_LABELS[key]} ({val:.2f}) "
                f"is below the expected minimum ({lo})."
            )
        if val > hi:
            warns.append(
                f"⚠️ Predicted {TARGET_LABELS[key]} ({val:.2f}) "
                f"is above the expected maximum ({hi})."
            )
    return warns

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="section-title">Input Parameters</div>',
            unsafe_allow_html=True)

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        depth_m = st.number_input(
            "Depth (m)",
            min_value=0.0, max_value=100.0,
            value=6.0, step=0.5,
            help="Depth of the soil layer from ground surface"
        )
        nfield = st.number_input(
            "Nfield - Field SPT Blow Count",
            min_value=0.1, max_value=100.0,
            value=15.0, step=0.5,
            help="Raw (uncorrected) field Standard Penetration Test blow count"
        )
        gwt_m = st.number_input(
            "Ground Water Table - GWT (m)",
            min_value=0.0, max_value=30.0,
            value=3.4, step=0.1,
            help="Depth to groundwater table from ground surface"
        )

    with col2:
        layer_thickness_m = st.selectbox(
    "Layer Thickness (m)",
    options=[0.91, 1.52],
    index=1,
    help="Thickness of the soil layer being tested"
)
        soil_type = st.selectbox(
            "Soil Type (USCS Classification)",
            options=SOIL_TYPES,
            index=0,
            help="Select soil type based on visual classification or sieve analysis"
        )

    if soil_type in LIMITED_DATA_SOIL_TYPES:
        st.markdown(
            '<div class="limit-box">⚠️ <strong>Data limitation:</strong> '
            f'the training dataset has no valid Friction Angle (φ) or Cohesion (c) '
            f'samples for <strong>{soil_type}</strong>. Only Specific Gravity and '
            f'Moisture Content will be predicted for this soil type.</div>',
            unsafe_allow_html=True
        )

    submitted = st.form_submit_button("🔍  Predict Soil Properties")

# ============================================================
# PREDICTION + OUTPUT
# ============================================================
if submitted:

    # ── Input validation
    input_warns = check_inputs(depth_m, nfield, gwt_m, layer_thickness_m)
    for w in input_warns:
        st.markdown(f'<div class="warn-box">{w}</div>',
                    unsafe_allow_html=True)

    # ── Determine which targets can actually be predicted for this soil type
    is_limited_soil = soil_type in LIMITED_DATA_SOIL_TYPES
    targets_to_predict = [
        t for t in TARGETS
        if not (is_limited_soil and t in LIMITED_TARGETS)
    ]
    skipped_targets = [t for t in TARGETS if t not in targets_to_predict]

    # ── Run predictions (via prediction_engine.predict) — one best model per target
    results = {}
    errors  = []

    with st.spinner("Generating predictions..."):
        for target in targets_to_predict:
            try:
                results[target] = predict(
                    target, depth_m, nfield, gwt_m,
                    layer_thickness_m, soil_type
                )
            except Exception as e:
                results[target] = None
                errors.append(f"{TARGET_LABELS[target]}: {e}")

    for err in errors:
        st.error(f"❌ Prediction failed — {err}")

    if skipped_targets:
        skipped_names = ", ".join(TARGET_LABELS[t] for t in skipped_targets)
        st.markdown(
            f'<div class="limit-box">ℹ️ {skipped_names} not predicted for '
            f'<strong>{soil_type}</strong> — insufficient training data for this soil type.</div>',
            unsafe_allow_html=True
        )

    # ── Results table
    st.markdown(
        '<div class="section-title">Predicted Soil Properties</div>',
        unsafe_allow_html=True
    )

    rows_html = ""
    result_rows = []
    for t in TARGETS:
        val = results.get(t)
        if val is None:
            continue
        val_r = round(val, 2)
        rows_html += (
            f"<tr>"
            f"<td>{TARGET_LABELS[t]}</td>"
            f'<td class="value-cell">{val_r}</td>'
            f"<td>{TARGET_UNITS[t]}</td>"
            f"</tr>"
        )
        result_rows.append({
            "Property": TARGET_LABELS[t],
            "Predicted Value": val_r,
            "Unit": TARGET_UNITS[t],
        })

    st.markdown(f"""
    <table class="result-table">
        <thead>
            <tr>
                <th>Property</th>
                <th>Predicted Value</th>
                <th>Unit</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # ── Output sanity warnings
    output_warns = check_outputs(results)
    for w in output_warns:
        st.markdown(f'<div class="warn-box">{w}</div>',
                    unsafe_allow_html=True)

    # ── CSV download
    if result_rows:
        import pandas as pd
        csv = pd.DataFrame(result_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Results (CSV)",
            data=csv,
            file_name="geotechnical_predictions.csv",
            mime="text/csv",
        )

# ============================================================
# NOTE BOX
# ============================================================
st.markdown("""
<div class="note-box">
    <strong>Note:</strong><br>
    Predictions are generated using machine learning models trained on
    borehole data collected from Dinajpur, Bangladesh.<br>
    For <strong>Clay(CL)</strong>, only Specific Gravity and Moisture Content
    can be predicted, since the training dataset has no valid Friction Angle
    or Cohesion samples for this soil type.<br>
    This tool is intended for <strong>academic and research purposes only</strong>.
    Predicted values should be verified with laboratory testing and
    professional engineering judgement before use in design.
</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-premium">
    <div class="footer-columns">
        <div class="footer-left">
            <div class="footer-title">Supervised By</div>
            <div class="footer-name">
                <strong>&nbsp;&nbsp;&nbsp;Nirmal Chandra Roy</strong><br>
                <span class="footer-role">&nbsp;&nbsp;&nbsp;Assistant Professor</span>
                <br>
                <strong>&nbsp;&nbsp;&nbsp;Md. Belal Hossain</strong><br>
                <span class="footer-role">&nbsp;&nbsp;&nbsp;Associate Professor</span>
            </div>
        </div>
        <div class="footer-right">
            <div class="footer-title">Developed By                       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div>
            <div class="footer-name">
                <strong>Md. Salauddin Munna     &nbsp;&nbsp;</strong><br>
                <strong>Md. Zonayid Hoshen&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</strong>
            </div>
        </div>
    </div>
    <div class="footer-department">Department of Civil Engineering, HSTU</div>
    <hr class="footer-divider">
    <div class="footer-version">
        Version 2.0 | August 2026<br>
        This tool was developed as part of a B.Sc. thesis research.
    </div>
</div>
""", unsafe_allow_html=True)