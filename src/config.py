"""
Central configuration, CPCB AQI standards, and hyperparameter specifications.
"""

AQI_CATEGORIES = {
    (0, 50): ("Good", "#00E400"),
    (51, 100): ("Satisfactory", "#FFFF00"),
    (101, 200): ("Moderate", "#FF7E00"),
    (201, 300): ("Poor", "#FF0000"),
    (301, 400): ("Very Poor", "#8F3F97"),
    (401, 500): ("Severe", "#7E0023"),
}

CORE_POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 350,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "early_stopping_rounds": 25,
}
