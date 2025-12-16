"""Pydantic request/response schemas for the prediction API."""

from typing import Literal, Optional
from typing import Dict, Any, List

from pydantic import BaseModel


class AQIPredictionRequest(BaseModel):
    city: Literal["Islamabad", "Lahore", "Karachi", "Peshawar", "Quetta"]
    datetime: str

    # Raw pollutants
    main_aqi: float
    components_co: float
    components_no: float
    components_no2: float
    components_o3: float
    components_so2: float
    components_pm2_5: float
    components_pm10: float
    components_nh3: float

    # Weather features
    temperature_2m: float
    relative_humidity_2m: float
    dew_point_2m: float
    precipitation: float
    surface_pressure: float
    wind_speed_10m: float
    wind_direction_10m: float
    shortwave_radiation: float

    # Engineered time features
    hour: int
    dayofweek: int
    month: int
    is_weekend: int

    # Lag & rolling features
    main_aqi_lag_1: float
    main_aqi_lag_2: float
    main_aqi_lag_3: float
    components_pm2_5_lag_1: float
    components_pm2_5_lag_2: float
    components_pm2_5_lag_3: float
    components_pm10_lag_1: float
    components_pm10_lag_2: float
    components_pm10_lag_3: float
    components_pm2_5_roll3_mean: float
    components_pm10_roll3_mean: float


class AQIPredictionResponse(BaseModel):
    city: str
    horizon_hours: int = 3

    predicted_main_aqi_t_plus_h: float
    predicted_hazard_t_plus_h: int

    hazard_label: str
    hazard_probability: Optional[float] = None

    model_version: str = "0.1.0"
    regression_rmse_test: float = 0.46
    classification_accuracy_test: float = 0.90


class FeaturesPredictRequest(BaseModel):
    features: Dict[str, Any]
    input_type: str = "features_json"


class FeaturesPredictResponse(BaseModel):
    result: Dict[str, Any]
    input_type: str = "features_json"


class FeaturesBatchResponse(BaseModel):
    results: List[Dict[str, Any]]
    input_type: str = "features_csv"
