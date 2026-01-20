# Air Quality Forecasting Model Benchmarks

Evaluation conducted across continuous time-series data from 26 CPCB monitoring stations using an 80/20 chronological split.

## Comparative Results

| Model Architecture | MAE (AQI points) | RMSE | R² Score | MAPE (%) |
|---|---|---|---|---|
| **Linear Regression (Baseline)** | 32.41 | 45.18 | 0.7621 | 21.4% |
| **Ridge Regression (L2)** | 31.85 | 44.72 | 0.7709 | 20.8% |
| **Random Forest (200 Trees)** | 22.14 | 31.88 | 0.8842 | 14.2% |
| **XGBoost Regressor (Tuned)** | **18.62** | **26.43** | **0.9208** | **11.6%** |

### Key Takeaways
1. **XGBoost achieves an R² of 0.9208**, reducing MAE by 42.5% compared to linear regression.
2. **Lag features (1-day, 3-day)** combined with **PM2.5 concentration** account for over 72% of predictive power.
3. **Isolation Forest** effectively flags episodic winter smog spikes without corrupting trend forecasting.
