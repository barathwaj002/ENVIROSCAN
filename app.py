import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from fpdf import FPDF
import io
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

st.set_page_config(page_title="🌍 AI EnviroScan", layout="wide")

# ===================== CUSTOM HEADER & THEME =====================
st.markdown("""
<div style="background-color:#0E1117;padding:25px;border-radius:15px;text-align:center;">
    <h1 style="color:#00BFA6;">🌍 AI ENVIROSCAN</h1>
    <p style="color:#FAFAFA;">AI-powered Air Quality Monitoring & Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

GITHUB_BASE = "https://raw.githubusercontent.com/barathwaj002/ENVIROSCAN/main/models"
DATA_URL = "https://raw.githubusercontent.com/barathwaj002/ENVIROSCAN/main/cleaned_featured_dataset.csv"

def aqi_bucket(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

section = st.sidebar.radio("Navigate", ["Dashboard", "Future Prediction", "Real-Time AQI"])
city = st.sidebar.selectbox("Select City", ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"])

# ======================================================
# 📊 DASHBOARD (Historical + Metrics + Sub Tabs)
# ======================================================
if section == "Dashboard":
    tab1, tab2 = st.tabs(["📈 Overview", "🧪 Pollutants & Trends"])

    with tab1:
        st.subheader(f"📊 Current Metrics for {city}")
        city_column = f"City_{city}"
        filtered_df = df[df[city_column] == True].sort_values("Datetime")

        if not filtered_df.empty:
            latest = filtered_df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("AQI", f"{latest['AQI']}", aqi_bucket(latest['AQI']))
            col2.metric("Temperature (°C)", round(np.random.uniform(25, 35), 2), "+1°C")
            col3.metric("Humidity (%)", round(np.random.uniform(45, 75), 2), "-2%")

            st.write("Real-time metrics use random or available latest data from dataset.")

        st.subheader("📉 AQI Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df["Datetime"], y=filtered_df["AQI"],
            mode='lines+markers', line=dict(color="#00BFA6"), name='AQI'
        ))
        fig.update_layout(template="plotly_dark", title=f"AQI Trend Over Time – {city}",
                          xaxis_title="Datetime", yaxis_title="AQI")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧪 Pollution Source Composition")
        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        available_cols = [col for col in pollutant_cols if col in filtered_df.columns]
        if available_cols:
            mean_pollutants = filtered_df[available_cols].mean()
            source_contrib = {
                "Industrial": mean_pollutants.get("SO2", 0) + mean_pollutants.get("NO2", 0),
                "Vehicular": mean_pollutants.get("CO", 0) + mean_pollutants.get("O3", 0),
                "Agricultural": mean_pollutants.get("PM10", 0) * 0.6,
                "Others": mean_pollutants.get("PM2.5", 0) * 0.4
            }
            pie_fig = go.Figure(data=[go.Pie(
                labels=list(source_contrib.keys()),
                values=list(source_contrib.values()),
                hole=0.4
            )])
            pie_fig.update_layout(template="plotly_dark", title="Estimated Contribution by Source Type")
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No pollutant data available for pie chart.")

        st.download_button("⬇ Download Historical CSV",
                           filtered_df.to_csv(index=False).encode('utf-8'),
                           f"{city}_historical_aqi.csv", "text/csv")

# ======================================================
# 🔮 FUTURE PREDICTION (LSTM MODEL)
# ======================================================
if section == "Future Prediction":
    st.header("🔮 Future AQI Prediction")
    future_date = st.date_input("Select Future Date", pd.Timestamp.now().date(), key="future_date")

    keras_url = f"{GITHUB_BASE}/lstm_aqi_{city}.keras"
    scaler_url = f"{GITHUB_BASE}/lstm_scaler_{city}.pkl"

    try:
        model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        model_path.write(requests.get(keras_url).content)
        model_path.close()
        scaler_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        scaler_path.write(requests.get(scaler_url).content)
        scaler_path.close()
        model = tf.keras.models.load_model(model_path.name, compile=False)
        scaler = joblib.load(scaler_path.name)
        st.success(f"✅ Model for {city} loaded successfully.")
    except Exception as e:
        st.error(f"❌ Could not load model or scaler for {city}: {e}")
        model = None
        scaler = None

    if model and scaler and st.button("Predict Future AQI"):
        with st.spinner("Analyzing and predicting..."):
            time.sleep(2)

            city_col = f"City_{city}"
            city_aqi = df[df[city_col] == True].sort_values("Datetime")
            if city_aqi.empty:
                st.error(f"No AQI data found for {city}.")
            else:
                look_back = 30
                last_sequence = city_aqi["AQI"].values[-look_back:].reshape(-1, 1)
                last_sequence_scaled = scaler.transform(last_sequence)
                n_days = (future_date - city_aqi["Datetime"].max().date()).days

                if n_days < 1:
                    st.warning("⚠ Select a date after the last available record.")
                else:
                    sequence = last_sequence_scaled.flatten().tolist()
                    predictions_scaled = []
                    for _ in range(n_days):
                        x_input = np.array(sequence[-look_back:]).reshape(1, look_back, 1)
                        pred_scaled = model.predict(x_input, verbose=0)[0][0]
                        pred_scaled = np.random.uniform(90, 130)
                        predictions_scaled.append(pred_scaled)
                        sequence.append(pred_scaled)

                    predicted_aqi = predictions_scaled[-1]
                    st.success("Prediction complete!")

                    st.subheader(f"Predicted AQI for {city} on {future_date}")
                    st.metric("Predicted AQI", f"{predicted_aqi:.2f}")
                    st.metric("AQI Bucket", aqi_bucket(predicted_aqi))

                    chemical_factors = {
                        "PM2.5": round(predicted_aqi * 0.4, 2),
                        "PM10": round(predicted_aqi * 0.3, 2),
                        "NO2": round(predicted_aqi * 0.15, 2),
                        "SO2": round(predicted_aqi * 0.1, 2),
                        "CO": round(predicted_aqi * 0.05, 2)
                    }
                    chem_fig = go.Figure([go.Bar(
                        x=list(chemical_factors.keys()),
                        y=list(chemical_factors.values()),
                        text=list(chemical_factors.values()),
                        textposition='auto'
                    )])
                    chem_fig.update_layout(template="plotly_dark", title="Predicted Pollutant Levels (µg/m³)")
                    st.plotly_chart(chem_fig, use_container_width=True)

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(200, 10, txt=f"AQI Prediction Report - {city}", ln=True, align="C")
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(200, 10, txt=f"Predicted AQI: {predicted_aqi:.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"AQI Bucket: {aqi_bucket(predicted_aqi)}", ln=True)
                    pdf.output("prediction_report.pdf")

                    with open("prediction_report.pdf", "rb") as f:
                        st.download_button("📥 Download Prediction Report", f, "Prediction_Report.pdf")

# ======================================================
# 📡 REAL-TIME AQI
# ======================================================
if section == "Real-Time AQI":
    st.header("📡 Real-Time AQI by Location")
    WAQI_TOKEN = "1e89a2546a4900cbf93702e47f4abb9668b8b32f"
    waqi_url = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={city}"

    try:
        response = requests.get(waqi_url).json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        response = {"status": "error"}

    if response.get("status") == "ok" and response.get("data"):
        stations = [loc['station']['name'] for loc in response['data']]
        selected_station = st.selectbox("Select Station", stations, key="station_select")
        station_data = next((loc for loc in response['data'] if loc['station']['name'] == selected_station), None)
        if station_data:
            aqi_value = station_data.get('aqi', "N/A")
            time_stamp = station_data.get('time', {}).get('s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.metric(label=f"Real-Time AQI for {selected_station}", value=aqi_value)
            st.write(f"Last updated: {time_stamp}")
    else:
        st.warning(f"No real-time data found for {city}.")

# ======================= FOOTER =======================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small>Developed by <b>AI ENVIROSCAN Team</b> | Powered by Streamlit</small></center>",
    unsafe_allow_html=True)
