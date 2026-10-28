# AI-Powered Air Quality Intelligence & AQI Forecasting

**Author:** Gaurav Rajbhar  
**Program:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares  
**Repository:** https://github.com/RezZup/AI-Powered-Air-Quality-Intelligence-AQI-Forecasting

## Project Overview

This project develops an integrated Air Quality Intelligence system for Indian cities. It combines data quality analysis, exploratory data analysis, leakage-safe time-series forecasting, pollution anomaly detection, and explainable artificial intelligence in one reproducible workflow.

The system uses historical CPCB city-level air-quality observations to answer three practical questions:

1. What temporal, seasonal, geographic, and pollutant patterns are present in the data?
2. How accurately can next-day AQI be forecast without using future information?
3. Which recent observations contribute most to the model's predictions?

## Dataset

- **Source:** [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) by Rohan Rao
- **File:** `city_day.csv`
- **Coverage:** 26 Indian cities
- **Period:** 2015-01-01 to 2020-07-01
- **Size:** 29,531 rows and 16 columns
- **Measures:** AQI, AQI category, particulate matter, nitrogen oxides, ammonia, carbon monoxide, sulfur dioxide, ozone, and volatile organic compounds

The raw CSV is not stored in this repository. Download instructions are available in [data/README.md](data/README.md).

## Objectives

- Assess data quality, missing values, duplicates, and station-level gaps.
- Explore pollutant distributions, correlations, city differences, and seasonal patterns.
- Engineer per-city lag and rolling features for next-day AQI forecasting.
- Compare Linear Regression, Random Forest, and XGBoost using a chronological split.
- Detect unusual pollution observations with Isolation Forest.
- Explain model behavior with SHAP.
- Document findings, limitations, and future improvements honestly.

## Methodology

1. Parse and sort observations by city and date.
2. Investigate missingness and data-quality issues.
3. Drop `Xylene` because of its high missingness.
4. Interpolate pollutant features within each city and use city/global medians only as fallback values.
5. Leave AQI and `AQI_Bucket` unfilled because AQI is the forecasting target.
6. Create lag, rolling-average, calendar, and seasonal features per city.
7. Create the next-day AQI target with a per-city shift.
8. Use an 80/20 chronological train/test split with no random shuffling.
9. Train and compare three regression models.
10. Apply Isolation Forest and SHAP to the valid modeling data and selected model.

## Results

### Forecasting Performance

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 27.24 | 50.68 | 0.671 |
| **Random Forest** | **27.20** | **49.96** | **0.681** |
| XGBoost | 27.87 | 50.32 | 0.676 |

Random Forest was selected because it achieved the lowest RMSE. Its improvement over the linear baseline was small, indicating that recent AQI history captures most of the learnable day-to-day signal.

### Anomaly Detection

Isolation Forest identified **1,243 anomalies**, representing **5.0%** of valid observations.

- Average AQI for anomalies: **545.08**
- Average AQI for normal observations: **146.53**
- Largest anomaly counts: Ahmedabad, Delhi, and Jorapokhar

These results are statistically based and should not be interpreted as independent confirmation of a real-world pollution emergency.

### Explainable AI

SHAP analysis identified the following leading contributors to model predictions:

1. `AQI_roll_mean_7`: mean absolute SHAP value 48.34
2. `AQI_lag_1`: 13.20
3. `PM2.5_lag_1`: 12.97

Recent AQI history is substantially more influential than calendar and seasonal features. SHAP describes model behavior and does not establish pollutant causation.

## Key Findings

- Winter and post-monsoon periods have the highest average AQI, while monsoon has the lowest.
- Ahmedabad, Delhi, and Patna have the highest average AQI among the covered cities.
- A strong geographic difference appears between more polluted northern cities and generally lower-AQI southern and northeastern cities.
- Ahmedabad is both a pollution outlier and the most difficult city for the forecasting model, with a city-level MAE of 103.37.
- Extended AQI gaps and station-level missingness materially affect interpretation and are documented rather than hidden.
- Gurugram's heavy PM10 missingness creates a visible median-imputation artifact in the pollutant distribution.

## Limitations

- City coverage is uneven and some cities have relatively few observations.
- Weather variables such as wind, temperature, humidity, and rainfall are not included.
- The forecast horizon is limited to the next day.
- The test period includes the COVID-19 lockdown, which is an unusual event relative to the training period.
- Anomaly detection is statistical and may be affected by sensor irregularities.
- Performance is weaker for volatile cities and sudden pollution spikes.

## Future Scope

- Add meteorological and weather-forecast data.
- Develop city-aware or city-specific models.
- Evaluate three-day and seven-day forecasts.
- Test sequence models such as LSTM or GRU.
- Integrate live CPCB data and expose forecasts through an API.
- Build an interactive Streamlit dashboard.

## Repository Contents

- [Notebook](notebooks/Gaurav_Rajbhar_AQI_Intelligence.ipynb)
- [Dataset instructions](data/README.md)
- [Selected project visualizations](assets/images/)
- [Full project report](Gaurav_Rajbhar_AQI_ProjectReport.docx)
- [Main README](README.md)

## Reproducibility

Install the pinned dependencies from `requirements.txt`, download `city_day.csv` according to [data/README.md](data/README.md), and run the notebook from the repository root. The notebook regenerates the analysis figures and model outputs used in this summary.
