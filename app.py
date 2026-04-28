"""
Weather Classification App — Streamlit
Halaman 1: Prediksi Cuaca
Halaman 2: Insight & Visualisasi Data
Halaman 3: Evaluasi Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ─────────────────────────────────────────────
# Paths (relative to app/)
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'model')
VIZ_DIR   = os.path.join(BASE, 'visualizations')
DATA_PATH = os.path.join(BASE, 'data', 'cleaned_dataset.csv')

@st.cache_resource
def load_artifacts():
    rf         = joblib.load(f'{MODEL_DIR}/random_forest_model.pkl')
    lr         = joblib.load(f'{MODEL_DIR}/logistic_regression_model.pkl')
    scaler     = joblib.load(f'{MODEL_DIR}/scaler.pkl')
    le_target  = joblib.load(f'{MODEL_DIR}/label_encoder_target.pkl')
    le_cloud   = joblib.load(f'{MODEL_DIR}/label_encoder_cloud.pkl')
    le_season  = joblib.load(f'{MODEL_DIR}/label_encoder_season.pkl')
    le_location= joblib.load(f'{MODEL_DIR}/label_encoder_location.pkl')
    features   = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
    with open(f'{MODEL_DIR}/metrics.json') as f:
        metrics = json.load(f)
    return rf, lr, scaler, le_target, le_cloud, le_season, le_location, features, metrics

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌤️ Weather Classifier",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🌤️ Weather Classifier")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📍 Navigasi",
    ["🔮 Prediksi Cuaca", "📊 Insight Data", "📈 Evaluasi Model"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Dataset:** Weather Classification  
    **13,200 records** | 4 kelas  
    **Model:** Random Forest + Logistic Regression
    """
)

rf, lr, scaler, le_target, le_cloud, le_season, le_location, features, metrics = load_artifacts()
df = load_data()

# ═══════════════════════════════════════════════════
# PAGE 1 — PREDIKSI CUACA
# ═══════════════════════════════════════════════════
if page == "🔮 Prediksi Cuaca":
    st.title("🔮 Prediksi Tipe Cuaca")
    st.markdown("Masukkan data kondisi cuaca untuk memprediksi tipe cuaca menggunakan model Machine Learning.")
    st.markdown("---")

    emoji_map = {"Sunny": "☀️ Sunny", "Cloudy": "☁️ Cloudy", "Rainy": "🌧️ Rainy", "Snowy": "❄️ Snowy"}
    color_map = {"Sunny": "#F4A261", "Cloudy": "#457B9D", "Rainy": "#2A9D8F", "Snowy": "#E9C46A"}

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🌡️ Suhu & Kelembaban")
            temperature = st.slider("Temperature (°C)", -20, 50, 25)
            humidity    = st.slider("Humidity (%)", 20, 100, 65)
            uv_index    = st.slider("UV Index", 0, 14, 4)

        with col2:
            st.subheader("💨 Angin & Presipitasi")
            wind_speed    = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, step=0.5)
            precipitation = st.slider("Precipitation (%)", 0, 100, 50)
            visibility    = st.slider("Visibility (km)", 0.0, 20.0, 5.0, step=0.5)

        with col3:
            st.subheader("🌍 Kondisi Lainnya")
            pressure   = st.slider("Atmospheric Pressure (hPa)", 950.0, 1050.0, 1010.0, step=0.5)
            cloud_cover= st.selectbox("Cloud Cover",  sorted(le_cloud.classes_))
            season     = st.selectbox("Season",        sorted(le_season.classes_))
            location   = st.selectbox("Location",      sorted(le_location.classes_))
            model_sel  = st.radio("🤖 Model", ["Random Forest", "Logistic Regression"])

        submitted = st.form_submit_button("👉 Prediksi Cuaca", use_container_width=True)

    if submitted:
        # Build feature vector
        cloud_enc    = le_cloud.transform([cloud_cover])[0]
        season_enc   = le_season.transform([season])[0]
        location_enc = le_location.transform([location])[0]
        hum_prec     = humidity * precipitation / 100
        temp_uv      = temperature / (uv_index + 1)

        row = pd.DataFrame([[temperature, humidity, wind_speed, precipitation,
                              pressure, uv_index, visibility, cloud_enc,
                              season_enc, location_enc, hum_prec, temp_uv]],
                           columns=features)
        row_sc = pd.DataFrame(scaler.transform(row), columns=features)

        model = rf if model_sel == "Random Forest" else lr
        pred_enc  = model.predict(row_sc)[0]
        pred_prob = model.predict_proba(row_sc)[0]
        pred_label = le_target.inverse_transform([pred_enc])[0]
        confidence = pred_prob.max() * 100

        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            color = color_map[pred_label]
            st.markdown(
                f"""
                <div style='background:{color}22; border:2px solid {color};
                            border-radius:16px; padding:24px; text-align:center;'>
                    <h1 style='color:{color}; margin:0; font-size:3rem;'>
                        {emoji_map[pred_label]}
                    </h1>
                    <h2 style='color:{color}; margin:8px 0;'>{pred_label}</h2>
                    <p style='font-size:1.1rem; margin:0;'>
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </p>
                    <p style='font-size:0.85rem; color:#666;'>Model: {model_sel}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_r2:
            st.subheader("📊 Probabilitas per Kelas")
            prob_df = pd.DataFrame({
                'Weather Type': le_target.classes_,
                'Probability (%)': pred_prob * 100
            }).sort_values('Probability (%)', ascending=False)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.barh(prob_df['Weather Type'], prob_df['Probability (%)'],
                           color=[color_map[wt] for wt in prob_df['Weather Type']], alpha=0.85)
            ax.set_xlim(0, 105)
            ax.set_xlabel('Probability (%)', fontsize=11)
            ax.set_title('Confidence per Weather Type', fontsize=12, fontweight='bold')
            for bar in bars:
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.1f}%', va='center', fontsize=10)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════════
# PAGE 2 — INSIGHT DATA
# ═══════════════════════════════════════════════════
elif page == "📊 Insight Data":
    st.title("📊 Insight & Visualisasi Data")
    st.markdown("Eksplorasi distribusi, korelasi, dan pola dalam dataset cuaca.")
    st.markdown("---")

    # Dataset Overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Weather Classes", df['Weather Type'].nunique())
    col4.metric("Balance", "Perfect (25% each)")

    st.markdown("---")

    viz_imgs = {
        "01_feature_distributions.png": ("📊 Distribusi Fitur per Tipe Cuaca",
            "Setiap fitur menunjukkan distribusi berbeda antar tipe cuaca. Temperature negatif dominan pada Snowy, "
            "Precipitation tinggi pada Rainy, UV Index tinggi pada Sunny."),
        "02_correlation_heatmap.png":   ("🔥 Heatmap Korelasi Fitur",
            "Humidity berkorelasi positif dengan Precipitation (+0.55). "
            "Temperature berkorelasi positif dengan UV Index. Fitur engineered Humidity×Precip sangat korelasi dengan keduanya."),
        "03_scatter_temp_humidity.png": ("💧 Temperature vs Humidity",
            "Snowy cluster di suhu rendah. Sunny cluster di suhu tinggi + humidity sedang. "
            "Rainy dan Cloudy overlap di humidity tinggi dengan suhu bervariasi."),
        "04_precipitation_vs_weather.png": ("🌧️ Distribusi Precipitation",
            "Rainy memiliki median precipitation tertinggi (~80%). Sunny memiliki precipitation rendah. "
            "Ini mengkonfirmasi bahwa Precipitation adalah prediktor terkuat untuk kelas Rainy."),
        "08_season_weather_heatmap.png": ("🗓️ Season vs Weather Type",
            "Winter didominasi Snowy. Summer didominasi Sunny. Spring dan Autumn lebih beragam. "
            "Season adalah fitur kontekstual penting untuk prediksi.")
    }

    for fname, (title, insight) in viz_imgs.items():
        fpath = os.path.join(VIZ_DIR, fname)
        if os.path.exists(fpath):
            st.subheader(title)
            st.image(fpath, use_container_width=True)
            st.info(f"💡 **Insight:** {insight}")
            st.markdown("---")

# ═══════════════════════════════════════════════════
# PAGE 3 — EVALUASI MODEL
# ═══════════════════════════════════════════════════
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model Machine Learning")
    st.markdown("Perbandingan performa Logistic Regression vs Random Forest.")
    st.markdown("---")

    lr_m = metrics['logistic_regression']
    rf_m = metrics['random_forest']

    # Winner banner
    winner = "Random Forest" if rf_m['accuracy'] > lr_m['accuracy'] else "Logistic Regression"
    st.success(f"🏆 **Model Terbaik: {winner}** dengan Accuracy {max(rf_m['accuracy'], lr_m['accuracy']):.2%}")
    st.markdown("---")

    # Side-by-side metrics
    col1, col2 = st.columns(2)
    for col, res, badge in [(col1, lr_m, "🔵"), (col2, rf_m, "🔴")]:
        with col:
            st.subheader(f"{badge} {res['name']}")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy",  f"{res['accuracy']:.2%}")
            m2.metric("Precision", f"{res['precision']:.2%}")
            m3, m4 = st.columns(2)
            m3.metric("Recall",    f"{res['recall']:.2%}")
            m4.metric("F1-Score",  f"{res['f1']:.2%}")

    st.markdown("---")

    # Confusion Matrices
    st.subheader("📉 Confusion Matrix")
    cm_path = os.path.join(VIZ_DIR, '05_confusion_matrices.png')
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
        st.caption("Diagonal = prediksi benar. Off-diagonal = kesalahan prediksi.")

    st.markdown("---")

    # Model Comparison
    st.subheader("📊 Perbandingan Performa")
    cmp_path = os.path.join(VIZ_DIR, '07_model_comparison.png')
    if os.path.exists(cmp_path):
        st.image(cmp_path, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.subheader("🧠 Feature Importance (Random Forest)")
    fi_path = os.path.join(VIZ_DIR, '06_feature_importance.png')
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)

    fi_data = metrics['feature_importance']
    fi_df = pd.DataFrame(list(fi_data.items()), columns=['Feature','Importance'])\
              .sort_values('Importance', ascending=False)
    fi_df['Importance (%)'] = (fi_df['Importance'] * 100).round(2)
    st.dataframe(fi_df[['Feature','Importance (%)']].reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # Insight analysis
    st.subheader("🧠 Analisis & Kesimpulan")
    st.markdown("""
    **Random Forest vs Logistic Regression:**
    - **Random Forest** unggul dengan accuracy **91.5%** vs Logistic Regression **86.7%**
    - Random Forest mampu menangkap interaksi non-linear antar fitur
    - Logistic Regression masih sangat baik (+86%) dan lebih mudah diinterpretasikan

    **Fitur Paling Penting:**
    1. 🌧️ **Precipitation (%)** — Prediktor utama untuk Rainy. Curah hujan tinggi = hujan.
    2. 🌡️ **Temperature** — Kritis untuk membedakan Snowy vs Sunny.
    3. 💧 **Humidity** — Berkorelasi kuat dengan Rainy & Cloudy.
    4. 🌊 **Humidity × Precipitation** — Feature engineering yang memperkuat sinyal Rainy.
    5. ☁️ **Cloud Cover** — Discriminator kuat untuk Cloudy vs Clear.

    **Error Pattern:**
    - Cloudy dan Rainy paling sering terkonfusi (humidity dan precipitation overlap)
    - Snowy dan Sunny paling mudah diprediksi (temperature range tidak overlap)
    """)
