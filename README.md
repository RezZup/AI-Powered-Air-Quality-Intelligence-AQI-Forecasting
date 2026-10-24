# 🌫️ AI-Powered Air Quality Intelligence & AQI Forecasting

### AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares

**Author:** Gaurav Rajbhar
**Repository:** https://github.com/RezZup/AI-Powered-Air-Quality-Intelligence-AQI-Forecasting

---

## 📌 Overview

Air pollution is a major public health and environmental concern in Indian cities. This project builds an **end-to-end Air Quality Intelligence system** using the CPCB (Central Pollution Control Board) city-level air quality dataset, combining exploratory analytics, leakage-safe AQI forecasting, anomaly detection, and explainable AI into a single coherent pipeline.

This is not a simple "AQI Prediction Using Machine Learning" script — it is a complete analytical case study that investigates real data quality issues, documents genuine dataset artifacts, engineers time-series-safe features, compares multiple models honestly, and explains model behavior using SHAP.

![Average AQI over time](assets/images/aqi_trend_overall.png)

*Average AQI across all cities from 2015 to 2020, showing recurring seasonal variation.*

---

## 🎯 Problem Statement

Air quality in Indian cities varies significantly across time and location, driven by seasonal and regional factors. Stakeholders need the ability to **forecast near-future air quality**, **detect abnormal pollution events**, and **understand which factors drive AQI predictions**, in a transparent, interpretable way — without relying on naive random-split evaluation that would leak future information into model training.

## 🎯 Objectives

1. Perform rigorous data quality analysis and cleaning on the CPCB city-level air quality dataset.
2. Conduct exploratory data analysis to understand pollutant distributions, correlations, and temporal/seasonal patterns.
3. Engineer leakage-safe temporal features (lags, rolling statistics) for next-day AQI forecasting.
4. Build and compare three forecasting models — Linear Regression baseline, Random Forest, and XGBoost — using a **chronological** (not random) train/test split.
5. Detect anomalous pollution readings using Isolation Forest.
6. Explain the forecasting model's behavior using SHAP.
7. Summarize findings into a factual, data-driven Air Quality Intelligence report.

---

## 🗂️ Dataset

**Source:** [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) by Rohan Rao, via Kaggle — commonly sourced from CPCB monitoring stations.

**File used:** `city_day.csv`

- **29,531 rows × 16 columns**
- **26 Indian cities**
- **Period:** 2015-01-01 to 2020-07-01
- Fields: `City`, `Date`, pollutant concentrations (`PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`), `AQI`, `AQI_Bucket`.

See [`data/README.md`](data/README.md) for download instructions.

---

## 🧰 Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data handling | Pandas 2.2.3, NumPy 2.1.3 |
| Visualization | Matplotlib 3.10.0, Seaborn 0.13.2 |
| Machine learning | Scikit-learn 1.6.1, XGBoost 3.4.1 |
| Explainability | SHAP 0.52.0 |
| Dataset access | Kagglehub 1.0.2 |
| Environment | Google Colab or Jupyter Notebook |

---

## 🧪 Methodology

```
Historical Indian Air Quality (city_day.csv)
          ↓
Data Quality Analysis (missingness, duplicates, invalid values)
          ↓
Data Cleaning (per-city time-based interpolation, median fallback)
          ↓
Exploratory Data Analysis (distributions, correlations, seasonality)
          ↓
Temporal Feature Engineering (lags, rolling averages — per city, leakage-safe)
          ↓
Chronological Train/Test Split (no random shuffling)
          ↓
Model Training & Comparison (Linear Regression, Random Forest, XGBoost)
          ↓
Pollution Anomaly Detection (Isolation Forest)
          ↓
SHAP Explainability
          ↓
Error Analysis & Air Quality Intelligence Summary
```

### Key Design Decisions

- **No random shuffling for forecasting evaluation** — an 80/20 chronological split (train: 2015-01-08 to 2019-12-14, test: 2019-12-15 to 2020-06-30) was used to simulate real-world forecasting and avoid temporal leakage.
- **Lag and rolling features computed per-city** — ensuring no city's temporal sequence leaks into another's.
- **Target variable never imputed** — AQI values were left missing where genuinely missing; only predictor/feature columns were imputed.
- **`Xylene` dropped** (61.3% missing) — too sparse to be reliable.

---

## 🧹 Data Cleaning Summary

| Step | Decision | Justification |
|---|---|---|
| Date conversion | Converted `Date` to datetime | All 29,531 dates parsed successfully |
| Sorting | Sorted by `City`, then `Date` | Required for correct time-series operations |
| Drop column | Dropped `Xylene` (61.3% missing) | Too sparse to be a reliable feature |
| Imputation | Per-city time-based interpolation on pollutant features | Preserves temporal variation better than a global fill |
| Fallback | Per-city median, then global median | Handles cities with an entire pollutant series missing |
| Target handling | `AQI` and `AQI_Bucket` were never imputed | Prevents target contamination during forecasting |
| Duplicates | 0 duplicate rows found | No action needed |
| Negative values | 0 negative pollutant values found | No action needed |

**Documented artifact:** Gurugram had 939 of 1,679 PM10 readings originally
missing. After interpolation, 766 remaining gaps were filled with the
city-level median, creating a visible aggregate PM10 distribution spike. This
is an acknowledged imputation artifact and does not affect the AQI target.

## 📊 Exploratory Data Analysis Highlights

### AQI Distribution

- Mean AQI: 166.46; median: 118; standard deviation: 140.70.
- AQI ranged from 13 to 2,049 and was strongly right-skewed.
- Category counts were: Moderate 8,829; Satisfactory 8,224; Poor 2,781;
    Very Poor 2,337; Good 1,341; Severe 1,338.

### City and Seasonal Patterns

- Highest average AQI: Ahmedabad (452.12), Delhi (259.49), and Patna (240.78).
- Lowest average AQI: Aizawl (34.77), Shillong (53.80), and Coimbatore (73.02).
- Winter (220.61) and post-monsoon (215.47) were substantially worse than
    monsoon (115.56); summer averaged 147.67.
- November and January had the highest monthly averages, while July and
    August had the lowest.

### Pollutant Correlations

The strongest pollutant correlations with AQI were CO (0.640), PM2.5 (0.635),
NO2 (0.531), and PM10 (0.519). Ahmedabad's unusually high average was also
investigated: it contained 647 missing AQI days, including consecutive gaps of
340 and 232 days. The analysis preserves those gaps rather than drawing
misleading lines between distant observations.

## 🛠️ Feature Engineering

| Feature Type | Features |
|---|---|
| AQI lags | `AQI_lag_1`, `AQI_lag_3`, `AQI_lag_7` |
| Pollutant lags | `PM2.5_lag_1`, `PM10_lag_1`, `NO2_lag_1`, `CO_lag_1` |
| Rolling averages | `AQI_roll_mean_3`, `AQI_roll_mean_7`, `PM2.5_roll_mean_3` |
| Calendar features | `Month`, `DayOfWeek`, and one-hot encoded `Season` |
| Target | `AQI_target_next_day`, shifted per city |

All lag and rolling features were computed with `groupby('City')`. Rolling
windows were shifted by one day before aggregation so the current day's value
could not leak into its own prediction. The final modeling dataset contained
21,841 rows and represented all 26 cities.

## 🤖 Models & Results

Next-day AQI forecasting, evaluated on a chronological hold-out test set:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression (Baseline) | 27.24 | 50.68 | 0.671 |
| **Random Forest (Selected)** | **27.20** | **49.96** | **0.681** |
| XGBoost | 27.87 | 50.32 | 0.676 |

**Random Forest** was selected as the best-performing model based on RMSE. Notably, all three models perform comparably, indicating that engineered lag/rolling AQI features capture most of the learnable temporal signal in this dataset.

![Model comparison](assets/images/model_comparison.png)

![Actual versus predicted AQI](assets/images/actual_vs_predicted.png)

### SHAP Top Predictive Features

![SHAP feature importance](assets/images/shap_importance.png)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---:|
| 1 | `AQI_roll_mean_7` (7-day rolling AQI average) | 48.34 |
| 2 | `AQI_lag_1` | 13.20 |
| 3 | `PM2.5_lag_1` | 12.97 |

### Anomaly Detection

- **1,243 anomalies detected (5.0%)** using Isolation Forest on AQI + key pollutants.
- Average AQI of anomalies: **545.08** vs. **146.53** for normal observations — validating that flagged points correspond to genuine severe pollution episodes.
- Anomalies were concentrated in **Ahmedabad (686)**, **Delhi (233)**, and **Jorapokhar (94)** — consistent with these cities' high average AQI.

---

## 📊 Key Findings

- **Strong seasonal pattern**: Winter (avg AQI 220.61) and Post-Monsoon (215.47) are significantly worse than Monsoon (115.56).
- **Clear geographic divide**: North Indian cities (Ahmedabad 452.12, Delhi 259.49, Patna 240.78) have far higher average AQI than South/Northeastern cities (Aizawl 34.77, Shillong 53.80, Coimbatore 73.02).
- **Ahmedabad is a persistent outlier** — highest average AQI, most anomalies, and the highest model prediction error (MAE 103.37, ~2.5x the next-worst city) — a consistent, cross-validated pattern.
- **Recent AQI history dominates predictive power**; calendar/seasonal features contribute comparatively little once lag/rolling features are included.
- **Data quality artifacts were identified and documented honestly** — e.g., a median-imputation spike in Gurugram's PM10 distribution, and large real AQI data gaps in Ahmedabad (340-day and 232-day stretches).

---

## 📁 Project Structure

```
AI-Air-Quality-Intelligence/
│
├── notebooks/
│   └── Gaurav_Rajbhar_AQI_Intelligence.ipynb
│
├── data/
│   └── README.md
│
├── Graphs/
│   └── Generated analysis figures (`.png`)
│
├── assets/
│   └── images/
│       └── Selected report visualizations (`.png`)
│
├── README.md
├── PROJECT_SUBMISSION.md
├── requirements.txt
├── Gaurav_Rajbhar_AQI_ProjectReport.docx
└── .gitignore
```

---

## ⚙️ Installation & Usage

### Option 1: Google Colab (Recommended)

1. Open `notebooks/Gaurav_Rajbhar_AQI_Intelligence.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run cells sequentially — the notebook uses `kagglehub` to automatically download the dataset (no manual upload needed).
3. All models, plots, and metrics will be generated in-notebook.

### Option 2: Local Environment

Use Python 3.10 or newer and create an isolated virtual environment before
installing the pinned dependencies:

```bash
git clone https://github.com/RezZup/AI-Powered-Air-Quality-Intelligence-AQI-Forecasting.git
cd AI-Powered-Air-Quality-Intelligence-AQI-Forecasting
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/Gaurav_Rajbhar_AQI_Intelligence.ipynb
```

See [`data/README.md`](data/README.md) for dataset download instructions if running locally.

The notebook also regenerates the analysis figures under `Graphs/`, including
AQI distributions, temporal trends, model comparisons, prediction errors,
anomaly timelines, and SHAP explanations.

### Reproducibility Checklist

1. Install the pinned packages from `requirements.txt`.
2. Download `city_day.csv` using the instructions in `data/README.md`.
3. Run the notebook from the repository root so relative paths resolve.
4. Compare regenerated figures and metrics with the committed analysis outputs.

---

## ⚠️ Limitations

- Reported metrics and figures correspond to the pinned dependency versions in
    `requirements.txt`; rerunning with different versions may produce small
    numerical or rendering differences.
- Uneven city coverage (96 to 2,009 rows per city in the final modeling set).
- High missingness in some pollutants required imputation (`Xylene` dropped entirely; `PM10`, `NH3` imputed with documented artifacts, e.g., a median-fill spike in Gurugram's PM10 data).
- Some cities have zero readings for entire pollutants (e.g., Lucknow has no PM10 data) — a real CPCB monitoring station limitation.
- No weather/meteorological data included, likely contributing to unexplained forecasting variance (R² ≈ 0.68).
- Forecast horizon limited to next-day only.
- Test period includes the COVID-19 lockdown (an atypical event with no precedent in training data).
- Anomaly detection is statistical, not a verified real-world pollution event confirmation.

Full details in the notebook's Limitations section and project report.

## 📌 Conclusion

The project demonstrates a complete and technically defensible workflow from
raw data investigation through cleaning, exploratory analysis, leakage-safe
forecasting, anomaly detection, and explainable AI. The results show that
recent AQI history captures most of the predictable next-day signal, while
Ahmedabad remains a consistent outlier in both pollution level and model error.

---

## 🚀 Future Scope

- Integrate weather/meteorological data.
- City-aware or city-specific modeling to reduce error concentration (e.g., Ahmedabad).
- Longer forecasting horizons (3-day, 7-day ahead).
- Deep learning sequence models (LSTM/GRU).
- Real-time CPCB API integration.
- Interactive dashboard (Streamlit) and API deployment.

---

## 📚 References

This project was developed independently, informed by publicly available Indian AQI analytics repositories for understanding dataset structure and common methodological approaches (e.g., avoiding temporal leakage):

1. [air_cast](https://github.com/Manglam11/air_cast) — Indian AQI forecasting
2. [AQI-Prediction-India-ComparativeStudy](https://github.com/dhriti-kourla/AQI-Prediction-India-ComparativeStudy)
3. [AQI-prediction-system](https://github.com/sayitisha/AQI-prediction-system)
4. [air-aware](https://github.com/gayathri-cg/air-aware)
5. [india-cpcb-aqi](https://github.com/Vonter/india-cpcb-aqi)

**Dataset:** [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) by Rohan Rao (Kaggle).

All code, analysis, feature engineering, and conclusions in this repository are independently developed and based on outputs generated from this dataset.

---

## 📄 License

This project is submitted as part of the AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares). Dataset usage complies with its original Kaggle licensing terms.