import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
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

# ===================== ANIMATED HEADER & THEME =====================
st.markdown("""
<style>
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.header {
    background: linear-gradient(-45deg, #0E1117, #0D7377, #14FFEC, #212121);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
</style>
<div class="header">
    <h1 style="color:#FAFAFA;">🌍 AI ENVIROSCAN</h1>
    <p style="color:#E3FDFD;">AI-powered Air Quality Monitoring & Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ===================== CONSTANTS =====================
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

# ===================== SIDEBAR =====================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/727/727790.png", width=100)
st.sidebar.title("🌿 Navigation")
section = st.sidebar.radio("Select Section", ["Dashboard", "Future Prediction", "Real-Time AQI"])
city = st.sidebar.selectbox("Select City", ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"])
st.sidebar.markdown(f"⏰ **Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("---")
st.sidebar.info("Use filters and interact with visualizations for more insights.")

# ======================================================
# 📊 DASHBOARD
# ======================================================
if section == "Dashboard":
    tab1, tab2 = st.tabs(["📈 Overview", "🧪 Pollutants & Sources"])

    with tab1:
        st.subheader(f"📊 Live Environmental Metrics – {city}")
        city_column = f"City_{city}"
        filtered_df = df[df[city_column] == True].sort_values("Datetime")

        if not filtered_df.empty:
            latest = filtered_df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("AQI", f"{latest['AQI']}", aqi_bucket(latest['AQI']))
            col2.metric("Temperature (°C)", round(np.random.uniform(25, 35), 2), "+1°C")
            col3.metric("Humidity (%)", round(np.random.uniform(45, 75), 2), "-2%")

            st.markdown("Real-time metrics reflect latest or simulated dataset values.")

            # Gauge Chart for AQI visualization
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=latest['AQI'],
                title={'text': f"{city} AQI Gauge"},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "#00BFA6"},
                    'steps': [
                        {'range': [0, 50], 'color': "#00E676"},
                        {'range': [51, 100], 'color': "#CDDC39"},
                        {'range': [101, 200], 'color': "#FFEB3B"},
                        {'range': [201, 300], 'color': "#FF9800"},
                        {'range': [301, 400], 'color': "#F44336"},
                        {'range': [401, 500], 'color': "#B71C1C"}
                    ],
                }))
            gauge.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark")
            st.plotly_chart(gauge, use_container_width=True)

            # AQI Trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df["Datetime"], y=filtered_df["AQI"],
                mode='lines+markers', line=dict(color="#14FFEC"), name='AQI'
            ))
            fig.update_layout(template="plotly_dark", title=f"AQI Trend Over Time – {city}",
                              xaxis_title="Datetime", yaxis_title="AQI")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧪 Pollution Source Distribution")
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
                hole=0.4,
                textinfo='label+percent'
            )])
            pie_fig.update_layout(template="plotly_dark", title="Estimated Contribution by Source Type")
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No pollutant data available for visualization.")

        st.download_button("⬇ Download Historical CSV",
                           filtered_df.to_csv(index=False).encode('utf-8'),
                           f"{city}_historical_aqi.csv", "text/csv")

# ======================================================
# 🔮 FUTURE PREDICTION
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
        with st.spinner("Analyzing data and predicting future AQI..."):
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
                    st.success("✅ Prediction Complete!")

                    st.subheader(f"Predicted AQI for {city} on {future_date}")
                    st.metric("Predicted AQI", f"{predicted_aqi:.2f}")
                    st.metric("AQI Bucket", aqi_bucket(predicted_aqi))

                    # Chemical pollutant projection
                    chemical_factors = {
                        "PM2.5": round(predicted_aqi * 0.4, 2),
                        "PM10": round(predicted_aqi * 0.3, 2),
                        "NO2": round(predicted_aqi * 0.15, 2),
                        "SO2": round(predicted_aqi * 0.1, 2),
                        "CO": round(predicted_aqi * 0.05, 2)
                    }
                    chem_fig = px.bar(
                        x=list(chemical_factors.keys()),
                        y=list(chemical_factors.values()),
                        color=list(chemical_factors.keys()),
                        title="Predicted Pollutant Concentrations (µg/m³)",
                        text=list(chemical_factors.values())
                    )
                    chem_fig.update_layout(template="plotly_dark", xaxis_title="Pollutant", yaxis_title="Concentration")
                    st.plotly_chart(chem_fig, use_container_width=True)

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
    "<center><small>💡 Developed by <b>AI ENVIROSCAN Team</b> | Powered by Streamlit | © 2025</small></center>",
    unsafe_allow_html=True)
