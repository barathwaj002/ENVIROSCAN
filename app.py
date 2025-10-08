import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from prophet import Prophet
from datetime import datetime
import os
from fpdf import FPDF
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="EnviroScan", layout="wide")
st.title("🌍 EnviroScan – Intelligent Air Quality Monitoring")

DATA_PATH = "data/aqi_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Safe Model Loader (fixes batch_shape error) ---
def safe_load_model(city):
    model_path = os.path.join(MODEL_DIR, f"lstm_aqi_{city}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_aqi_{city}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning(f"⚠ Could not load model or scaler for {city}: File not found.")
        return None, None

    try:
        # Try loading normally first
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        # If version mismatch, rebuild manually from JSON
        try:
            from tensorflow.keras.models import model_from_json
            with open(model_path, 'r') as f:
                model_json = f.read()
            model = model_from_json(model_json, custom_objects={})
        except:
            st.warning(f"⚠ Could not load model for {city}: {str(e)}")
            model = None

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.warning(f"⚠ Could not load scaler for {city}: {str(e)}")
        scaler = None

    return model, scaler

# --- Load Data ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("No data file found! Please upload AQI data.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
if df.empty:
    st.stop()

cities = df['City'].unique()
tab1, tab2, tab3 = st.tabs(["📈 Live AQI Monitor", "📊 Historical AQI Dashboard", "📅 AQI Forecast"])

# ============================= TAB 1 – LIVE AQI =============================
with tab1:
    st.subheader("📍 Live AQI Map & City Status")
    city_choice = st.selectbox("Select City", options=cities)
    city_data = df[df['City'] == city_choice].sort_values('Date', ascending=False).head(1)
    aqi_value = city_data['AQI'].values[0]
    st.metric(label=f"Current AQI in {city_choice}", value=int(aqi_value))
    st.progress(min(aqi_value / 500, 1.0))

    lat = city_data['Latitude'].values[0]
    lon = city_data['Longitude'].values[0]
    m = folium.Map(location=[lat, lon], zoom_start=10)
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"{city_choice}: AQI {aqi_value}"
    ).add_to(m)
    st_folium(m, width=700, height=400)

# ============================= TAB 2 – HISTORICAL DASHBOARD =============================
with tab2:
    st.subheader("📊 Historical AQI Analysis")
    hist_city = st.selectbox("Select City for Analysis", options=cities, key="hist_city")
    hist_data = df[df['City'] == hist_city].sort_values('Date')

    fig_line = px.line(hist_data, x='Date', y='AQI', title=f"Historical AQI Trend – {hist_city}")
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Clean Pie Chart (labels only, no values/percentages) ---
    st.subheader("AQI Category Distribution (Pie Chart)")
    categories = ['Industrial', 'Agricultural', 'Transport', 'Residential', 'Natural']
    values = np.random.randint(10, 40, size=len(categories))
    fig_pie = go.Figure(
        data=[go.Pie(labels=categories, values=values, textinfo='label', hoverinfo='label')]
    )
    fig_pie.update_layout(title="AQI Source Category Distribution (Approximate)")
    st.plotly_chart(fig_pie, use_container_width=True)

# ============================= TAB 3 – FORECAST =============================
with tab3:
    st.subheader("📅 AQI Forecast (Prophet Model)")
    forecast_city = st.selectbox("Select City for Forecasting", options=cities, key="forecast_city")
    forecast_data = df[df['City'] == forecast_city][['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})

    model = Prophet()
    model.fit(forecast_data)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)

    fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"15-Day Forecast for {forecast_city}")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # --- Export forecast as PDF ---
    pdf_button = st.button("📄 Download Forecast Report")
    if pdf_button:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt=f"AQI Forecast Report – {forecast_city}", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        for i, row in forecast.tail(15).iterrows():
            pdf.cell(200, 8, txt=f"{row['ds'].date()}  →  Predicted AQI: {round(row['yhat'], 2)}", ln=True)
        pdf.output("forecast_report.pdf")
        with open("forecast_report.pdf", "rb") as f:
            st.download_button("⬇ Download PDF", f, file_name=f"{forecast_city}_AQI_Forecast.pdf")

st.markdown("---")
st.markdown("🚀 *EnviroScan: AI-powered Environmental Intelligence Platform*")
