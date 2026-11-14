import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json

st.set_page_config(
    page_title="Heavy Rainfall Nowcaster & Flood Potential",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1e3d59; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #555; margin-bottom: 25px; }
    .metric-card {
        background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px;
        padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); text-align: center;
    }
    .alert-box {
        padding: 15px; border-radius: 8px; font-weight: 600; margin-top: 15px;
    }
    .alert-high { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .alert-low { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
</style>
""", unsafe_allow_html=True)

# Helper function for logit
def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

# SCS-CN Runoff formulation
def scs_runoff(rain_mm, curve_number=80.0):
    retention = 25400.0 / curve_number - 254.0
    abstraction = 0.2 * retention
    rain = np.asarray(rain_mm, dtype=float)
    return np.where(rain > abstraction, (rain - abstraction) ** 2 / (rain - abstraction + retention), 0.0)

def new_runoff(antecedent_24h, next_rain_3h, curve_number=80.0):
    return scs_runoff(antecedent_24h + next_rain_3h, curve_number) - scs_runoff(antecedent_24h, curve_number)

# Load model payload
MODEL_PATH = Path("outputs/models/heavy_rain_nowcaster.joblib")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_PATH}`! Please check artifacts.")
        return None
    return joblib.load(MODEL_PATH)

payload = load_model()

# Header
st.markdown('<div class="main-title">🌧️ Spatio-Temporal Heavy Rainfall Nowcasting & Flood Potential Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Inference Interface for Northern India (28°N–33°N, 75°E–80°E) | Stacking Ensemble Meta-Learner (LightGBM + XGBoost + CatBoost)</div>', unsafe_allow_html=True)

if payload is None:
    st.stop()

# Extract model elements
selected_model_name = payload["selected_model"]
base_models = payload["base_models"]
meta_learner = payload["meta_learner"]
alert_threshold = payload["threshold"]
event_threshold_mm = payload["event_threshold_mm"]
features = payload["features"]
amount_model = payload["amount_model"]
defaults = payload["feature_defaults"]

# Sidebar: Input Configuration
st.sidebar.header("🕹️ Weather & Terrain Inputs")
st.sidebar.markdown("Modify parameters below to run **live model inference**.")

# Presets for quick evaluation
preset = st.sidebar.selectbox(
    "Choose Atmospheric Scenario Preset:",
    ["Custom Inputs", "Extreme Monsoon Storm Event (High Risk)", "Dry Pre-Monsoon Condition (Low Risk)", "Moderate Orographic Shower"]
)

# Set values based on presets
if preset == "Extreme Monsoon Storm Event (High Risk)":
    init_cape = 2400.0
    init_rh = 94.0
    init_t2m = 28.5
    init_d2m = 27.5
    init_tcc = 0.95
    init_p_curr = 4.8
    init_p_12h = 24.5
    init_twi = 0.75
    init_elev = 950.0
    init_p_nbr5 = 8.5
elif preset == "Dry Pre-Monsoon Condition (Low Risk)":
    init_cape = 150.0
    init_rh = 42.0
    init_t2m = 38.0
    init_d2m = 18.0
    init_tcc = 0.15
    init_p_curr = 0.0
    init_p_12h = 0.0
    init_twi = 0.10
    init_elev = 260.0
    init_p_nbr5 = 0.0
elif preset == "Moderate Orographic Shower":
    init_cape = 950.0
    init_rh = 82.0
    init_t2m = 24.0
    init_d2m = 21.0
    init_tcc = 0.70
    init_p_curr = 1.2
    init_p_12h = 8.0
    init_twi = 0.45
    init_elev = 1450.0
    init_p_nbr5 = 2.8
else:
    init_cape = float(defaults.get("cape", 800.0))
    init_rh = float(defaults.get("rh", 75.0))
    init_t2m = float(defaults.get("t2m", 26.5))
    init_d2m = float(defaults.get("d2m", 23.5))
    init_tcc = float(defaults.get("tcc", 0.65))
    init_p_curr = float(defaults.get("precip_current", 0.5))
    init_p_12h = float(defaults.get("precip_sum_12h", 5.0))
    init_twi = float(defaults.get("terrain_high_twi_fraction", 0.35))
    init_elev = float(defaults.get("terrain_elev_mean", 650.0))
    init_p_nbr5 = float(defaults.get("precip3h_nbr5_max", 1.5))

with st.sidebar.expander("⚡ Atmospheric Instability & Moisture", expanded=True):
    cape = st.slider("CAPE (J/kg) - Convective Instability", 0.0, 5000.0, init_cape, 50.0)
    t2m = st.slider("2m Temperature (°C)", 15.0, 48.0, init_t2m, 0.5)
    d2m = st.slider("2m Dew Point (°C)", 10.0, 32.0, min(init_d2m, t2m), 0.5)
    rh = st.slider("Relative Humidity (%)", 20.0, 100.0, init_rh, 1.0)
    tcc = st.slider("Total Cloud Cover (0-1)", 0.0, 1.0, init_tcc, 0.05)

with st.sidebar.expander("🌧️ Antecedent Precipitation History", expanded=True):
    p_curr = st.number_input("Current 1-hour Rain (mm/h)", 0.0, 80.0, init_p_curr, 0.5)
    p_12h = st.number_input("Antecedent 12-hour Rain Accumulation (mm)", 0.0, 200.0, init_p_12h, 1.0)
    p_nbr5 = st.number_input("Max Rain in 5-grid Neighbourhood (mm)", 0.0, 100.0, init_p_nbr5, 0.5)

with st.sidebar.expander("⛰️ Topography & Geography", expanded=False):
    elev = st.slider("Elevation (m a.s.l.)", 150.0, 4000.0, init_elev, 50.0)
    twi = st.slider("High TWI (Wetness Index) Fraction", 0.0, 1.0, init_twi, 0.05)
    lat = st.number_input("Latitude (°N)", 28.0, 33.0, 30.5, 0.25)
    lon = st.number_input("Longitude (°E)", 75.0, 80.0, 77.5, 0.25)

# Construct Feature Vector dynamically (no hardcoding)
input_dict = dict(defaults)
input_dict.update({
    "cape": cape,
    "t2m": t2m,
    "d2m": d2m,
    "rh": rh,
    "tcc": tcc,
    "precip_current": p_curr,
    "precip_lag_1": p_curr * 0.8,
    "precip_lag_2": p_curr * 0.6,
    "precip_lag_3": p_curr * 0.4,
    "precip_sum_12h": p_12h,
    "precip_max_6h": max(p_curr, p_12h * 0.4),
    "precip_max_24h": p_12h * 1.5,
    "precip3h_nbr5_max": p_nbr5,
    "cape_nbr5_max": cape * 1.15,
    "terrain_elev_mean": elev,
    "terrain_high_twi_fraction": twi,
    "lat": lat,
    "lon": lon
})

input_df = pd.DataFrame([input_dict])[features]

# Live Inference execution
# 1. Base Boosters Prediction Probabilities
p_lgb = float(base_models["lightgbm"].predict_proba(input_df)[0, 1])
p_xgb = float(base_models["xgboost"].predict_proba(input_df)[0, 1])
p_cb = float(base_models["catboost"].predict_proba(input_df)[0, 1])

# 2. Meta-Learner Stacking Inference
meta_features = np.array([[logit(p_lgb), logit(p_xgb), logit(p_cb)]])
prob_heavy_rain = float(meta_learner.predict_proba(meta_features)[0, 1])

# 3. Continuous Rainfall Amount Estimation (Tweedie Regressor)
pred_rain_amount = float(np.clip(amount_model.predict(input_df)[0], 0.0, None))

# 4. SCS-CN Hydrological Runoff & Topographic Flood Potential Index (TFPI)
runoff_3h = float(new_runoff(p_12h, pred_rain_amount, curve_number=80.0))
tfpi = float(runoff_3h * twi)

# Main Dashboard Layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Heavy Rain Probability (Next 3h)", value=f"{prob_heavy_rain * 100:.1f}%")
    st.caption(f"Alert Threshold: {alert_threshold * 100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Predicted Rain Amount (Next 3h)", value=f"{pred_rain_amount:.2f} mm")
    st.caption(f"Heavy Rain Cutoff: ≥ {event_threshold_mm:.2f} mm")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="SCS-CN Surface Runoff (3h)", value=f"{runoff_3h:.2f} mm")
    st.caption("Hydrological Infiltration Excess (CN=80)")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Topographic Flood Potential (TFPI)", value=f"{tfpi:.2f}")
    st.caption("Runoff × High TWI Saturation Area")
    st.markdown('</div>', unsafe_allow_html=True)

# Alert Banner
is_alert = prob_heavy_rain >= alert_threshold
if is_alert:
    st.markdown(
        f'<div class="alert-box alert-high">⚠️ <b>HEAVY RAINFALL WARNING ISSUED</b>: The ensemble probability of '
        f'heavy precipitation (≥ {event_threshold_mm:.2f} mm) is <b>{prob_heavy_rain*100:.1f}%</b>, exceeding the '
        f'calibrated alert threshold ({alert_threshold*100:.1f}%). High surface runoff expected!</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="alert-box alert-low">✅ <b>NORMAL CONDITIONS (NO ALERT)</b>: Heavy rain probability is '
        f'<b>{prob_heavy_rain*100:.1f}%</b>, comfortably below the alert threshold ({alert_threshold*100:.1f}%).</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# Visual Diagnostic Tabs
tab1, tab2, tab3 = st.tabs(["📊 Multi-Model Ensemble Breakdown", "🗺️ Geographical Context", "📖 Feature Pipeline Details"])

with tab1:
    st.subheader("Base Model Predictions vs. Stacking Meta-Learner")
    model_df = pd.DataFrame({
        "Model": ["LightGBM (Tuned)", "XGBoost (Tuned)", "CatBoost (Tuned)", "Stacking Meta-Learner (Selected)"],
        "Heavy Rain Probability (%)": [p_lgb * 100, p_xgb * 100, p_cb * 100, prob_heavy_rain * 100],
        "Alert Triggered": [p_lgb >= 0.368, p_xgb >= 0.711, p_cb >= 0.351, is_alert]
    })
    st.dataframe(model_df.style.format({"Heavy Rain Probability (%)": "{:.2f}%"}), use_container_width=True)

    st.bar_chart(model_df.set_index("Model")["Heavy Rain Probability (%)"])

with tab2:
    st.subheader("Target Coordinate Inspection (Northern India Study Grid)")
    map_df = pd.DataFrame({
        "lat": [lat],
        "lon": [lon],
        "Intensity": [pred_rain_amount]
    })
    st.map(map_df, zoom=6)
    st.caption(f"Coordinates: {lat}°N, {lon}°E | Elevation: {elev:.0f} m a.s.l.")

with tab3:
    st.subheader("Runtime Feature Vector (37 Spatio-Temporal & Terrain Predictors)")
    st.write(input_df)

st.markdown("<br><hr><center><small>End-to-End Spatio-Temporal Nowcasting Pipeline | Real-Time Inference Interface</small></center>", unsafe_allow_html=True)
