import pandas as pd
from typing import Optional, Union
import os
import numpy as np 
from cipher_edge.ml_models.price_forecasting.lightgbm_forecaster import LightGBMForecaster
from cipher_edge.app_logger import get_logger
from cipher_edge.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class LightGBMInference:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "LightGBM_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        
        model_params_config = settings.get_strategy_params(model_config_section) 
        
        _model_base_path = model_params_config.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = model_params_config.get('modelfilename', f"lgbm_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
        
        if not os.path.isabs(_model_base_path):
            self.model_load_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_load_path_dir = _model_base_path
            
        self.model_full_load_path = os.path.join(self.model_load_path_dir, _model_filename)
        self.forecaster = LightGBMForecaster(model_path=self.model_full_load_path, params=model_params_config)
        
        if self.forecaster.model is None:
            logger.warning(f"LightGBMInference: Model could not be loaded from {self.model_full_load_path}. Predictions will not be available.")

    def get_prediction(self, current_data_history: pd.DataFrame) -> Optional[float]:
        """
        Gets a single prediction based on the current data history.
        Assumes current_data_history has enough data to form features for the last point.
        """
        if self.forecaster.model is None:
            logger.warning("No model loaded, cannot get prediction.")
            return None
        if current_data_history.empty:
            logger.warning("Data history is empty, cannot get prediction.")
            return None
        
        feature_cols_to_pass = None
        feature_cols_str_from_params = self.forecaster.params.get('feature_columns')
        if isinstance(feature_cols_str_from_params, str) and feature_cols_str_from_params:
            feature_cols_to_pass = [col.strip() for col in feature_cols_str_from_params.split(',')]
        elif isinstance(feature_cols_str_from_params, list):
                feature_cols_to_pass = feature_cols_str_from_params
        
        prediction_output = self.forecaster.predict(current_data_history, feature_columns_to_use=feature_cols_to_pass)
        
        if prediction_output is None:
            return None
        
        if isinstance(prediction_output, pd.Series):
            if not prediction_output.empty:
                return prediction_output.iloc[-1] 
            else:
                logger.warning("Prediction series is empty.")
                return None
        elif isinstance(prediction_output, (float, np.float64)):
            return float(prediction_output)
        else:
            logger.warning(f"Unexpected prediction output type: {type(prediction_output)}")
            return None