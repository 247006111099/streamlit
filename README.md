# 🌤️ Weather Classification — End-to-End ML Project

Proyek sains data lengkap untuk klasifikasi tipe cuaca menggunakan Machine Learning.

## 📊 Dataset
- **13,200 records** | **4 kelas** (Sunny, Cloudy, Rainy, Snowy)
- Fitur: Temperature, Humidity, Wind Speed, Precipitation, Atmospheric Pressure, UV Index, Visibility, Cloud Cover, Season, Location
- Dataset **perfectly balanced** (3,300 per kelas)

## 🏆 Hasil Model

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **91.5%** | **91.6%** | **91.5%** | **91.5%** |
| Logistic Regression | 86.7% | 86.7% | 86.7% | 86.7% |

✅ **Winner: Random Forest** dengan akurasi 91.5%

## 🗂️ Struktur Project

```
project/
├── data/
│   └── cleaned_dataset.csv          # Dataset utama
├── notebook/
│   └── modeling.ipynb               # Jupyter Notebook end-to-end
├── model/
│   ├── random_forest_model.pkl      # Model terbaik
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder_*.pkl
│   └── metrics.json
├── app/
│   └── app.py                       # Streamlit web app
├── visualizations/
│   ├── 01_feature_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_scatter_temp_humidity.png
│   ├── 04_precipitation_vs_weather.png
│   ├── 05_confusion_matrices.png
│   ├── 06_feature_importance.png
│   ├── 07_model_comparison.png
│   └── 08_season_weather_heatmap.png
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Jupyter Notebook
```bash
cd notebook
jupyter notebook modeling.ipynb
```

### 3. Jalankan Streamlit App
```bash
cd app
streamlit run app.py
```

## 🔑 Key Insights

1. **Precipitation** adalah prediktor terkuat untuk Rainy weather
2. **Temperature** sangat discriminative antara Snowy (< 0°C) dan Sunny (> 25°C)
3. **Humidity × Precipitation** (feature engineering) memperkuat prediksi wet weather
4. **Atmospheric Pressure** rendah (< 1000 hPa) mengindikasikan cuaca ekstrem
5. Dataset seimbang sempurna → tidak perlu SMOTE/resampling

## ⚙️ Pipeline

```
Raw Data → Outlier Clipping (IQR) → Label Encoding → Feature Engineering 
→ StandardScaler → Train-Test Split (80:20) → Model Training → Evaluation → Streamlit App
```

## 🛠️ Teknologi

- **Python 3.10+**
- **scikit-learn** — ML models
- **pandas / numpy** — Data processing
- **matplotlib / seaborn** — Visualization
- **Streamlit** — Web application
- **joblib** — Model serialization
