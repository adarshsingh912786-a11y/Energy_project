# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Consumption Predictor", layout="centered")

st.markdown("""
    <style>
        body { background-color: #0f1724; color: #e6eef8; }
        .stButton>button { background-color:#0ea5a4; color: #021; }
        .stSlider .css-1aumxhk { color: #e6eef8; }
        .stDownloadButton>button { background-color:#0ea5a4; color: #021; }
        .stTextInput>div>div>input { background-color:#111827; color: #e6eef8; }
        .stSelectbox>div>div>div { background-color:#111827; color: #e6eef8; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Consumption Predictor (UI matched to your model)")
st.write("This app expects the model file `models/rf_model_with_features.pkl` to exist (save it from your training script).")

MODEL_PATH = "models/rf_model_with_features.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Run your training script and save the model with features.")
    st.stop()

# Load model + feature names
saved = joblib.load(MODEL_PATH)
model = saved["model"]
expected_features = saved["features"]  # list of column names in training order

st.sidebar.header("Input options")
# Let user choose a timestamp (we will derive hour, day_of_week, month)
use_date = st.sidebar.date_input("Choose a date (for hour/day/month)", value=pd.to_datetime("2020-01-01"))
use_hour = st.sidebar.slider("Hour of day", 0, 23, 12)
# user can choose day_of_week or let app compute
compute_dayinfo = st.sidebar.checkbox("Compute day_of_week & is_weekend from date", value=True)

# Numeric inputs: show inputs for numeric features we detected in your training set
# We'll prompt user for these plausible numeric features if they are in expected_features
defaults = {
    "Global_reactive_power": 0.1,
    "Voltage": 240.0,
    "Sub_metering_3": 5.0
}

st.header("Feature inputs")

# Prepare an input dict for all expected features (fill with default if not provided)
input_dict = {f: 0.0 for f in expected_features}

# Populate time-derived features if present
if "hour" in expected_features:
    input_dict["hour"] = use_hour
if "month" in expected_features:
    input_dict["month"] = use_date.month
if "day_of_week" in expected_features:
    if compute_dayinfo:
        input_dict["day_of_week"] = use_date.weekday()
    else:
        input_dict["day_of_week"] = st.number_input("Day of week (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=0)
if "is_weekend" in expected_features:
    if compute_dayinfo:
        input_dict["is_weekend"] = 1 if use_date.weekday() >= 5 else 0
    else:
        is_weekend_ui = st.selectbox("Is weekend?", ("No","Yes"))
        input_dict["is_weekend"] = 1 if is_weekend_ui == "Yes" else 0

# Numeric feature inputs (only for features present)
for feat in ["Global_reactive_power", "Voltage", "Sub_metering_3"]:
    if feat in expected_features:
        input_dict[feat] = st.number_input(f"{feat}", value=float(defaults.get(feat, 0.0)))

# If there are other numeric features in expected_features, expose generic inputs
extra_feats = [f for f in expected_features if f not in ["hour","month","day_of_week","is_weekend",
                                                         "Global_reactive_power","Voltage","Sub_metering_3"]]
if extra_feats:
    st.subheader("Other features (defaults shown; adjust if needed)")
    for ef in extra_feats:
        input_dict[ef] = st.number_input(ef, value=float(input_dict.get(ef, 0.0)))

st.write("---")
st.write("Features that model expects (in order):")
st.write(expected_features)

if st.button("🔮 Predict Energy Consumption"):
    # Build DataFrame with columns in the exact expected order
    X_input = pd.DataFrame([input_dict], columns=expected_features)

    # IMPORTANT: your model was trained with Global_reactive_power already log1p-transformed.
    if "Global_reactive_power" in X_input.columns:
        # avoid transforming NaN or negative inputs: force non-negative
        val = float(X_input.at[0, "Global_reactive_power"])
        if val < 0:
            st.warning("Global_reactive_power < 0; setting to 0 before transformation.")
            val = 0.0
        X_input.at[0, "Global_reactive_power"] = np.log1p(val)

    # Ensure numeric dtype
    X_input = X_input.astype(float)

    # Predict (model output is log1p(Global_active_power) because training used log1p on target).
    pred_log = model.predict(X_input)[0]
    pred_original = np.expm1(pred_log)  # invert log1p to get kW

    st.subheader(f"Predicted Energy Consumption: **{pred_original:.4f} kW**")
    st.caption("Model internally predicts in log1p space; value above is converted back to kW (expm1).")

    # Offer raw/technical numbers
    st.write("Raw model output (log1p space):", float(pred_log))

    # Optionally show a small bar with inputs
    st.write("Input features used for this prediction:")
    st.table(X_input.T.rename(columns={0:"value"}))

# Optional: show hourly pattern if data exists (your training data used hourly resampling)
if st.checkbox("Show hourly average pattern (if 'hourly_data.csv' present)"):
    if os.path.exists("hourly_data.csv"):
        df_hourly = pd.read_csv("hourly_data.csv", parse_dates=True, index_col=0)
        if "Global_active_power" in df_hourly.columns:
            hourly_avg = df_hourly.groupby(df_hourly.index.hour)["Global_active_power"].mean()
            fig, ax = plt.subplots()
            ax.plot(hourly_avg.index, hourly_avg.values, marker="o")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Avg Global Active Power (kW)")
            ax.set_title("Average consumption by hour (from hourly_data.csv)")
            st.pyplot(fig)
        else:
            st.write("hourly_data.csv loaded but no 'Global_active_power' column found.")
    else:
        st.write("No hourly_data.csv found in repo root.")

st.markdown("---")
st.caption("Note: This UI assumes you saved the model using the training snippet provided. Predictions are only as good as the trained model and the inputs provided.")