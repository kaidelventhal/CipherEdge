# FILE: kamikaze_komodo/ml_models/inference_pipelines/xgboost_classifier_inference.py
import pandas as pd
from typing import Optional, Dict, Any
import os
import numpy as np
from kamikaze_komodo.ml_models.price_forecasting.xgboost_classifier_forecaster import XGBoostClassifierForecaster
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class XGBoostClassifierInference:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "XGBoost_Classifier_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        
        model_params_config = settings.get_strategy_params(model_config_section)
        
        _model_base_path = model_params_config.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = model_params_config.get('modelfilename', f"xgb_classifier_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
        
        if not os.path.isabs(_model_base_path):
            self.model_load_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_load_path_dir = _model_base_path
            
        self.model_full_load_path = os.path.join(self.model_load_path_dir, _model_filename)
        self.forecaster = XGBoostClassifierForecaster(model_path=self.model_full_load_path, params=model_params_config)
        
        if self.forecaster.model is None:
            logger.warning(f"XGBoostClassifierInference: Model could not be loaded from {self.model_full_load_path}. Predictions will not be available.")

    def get_prediction(self, current_data_history: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Gets a single classification prediction based on the current data history.
        Returns a dictionary with 'predicted_class', 'confidence', and 'probabilities'.
        """
        if self.forecaster.model is None:
            logger.warning("XGBoost model not loaded, cannot get prediction.")
            return None
        if current_data_history.empty:
            logger.warning("Data history for XGBoost prediction is empty.")
            return None
            
        prediction_output = self.forecaster.predict(current_data_history) # XGBoostClassifierForecaster.predict returns a dict
        
        if prediction_output and isinstance(prediction_output, dict):
            return prediction_output
        else:
            logger.warning(f"Unexpected prediction output type from XGBoost forecaster: {type(prediction_output)}")
            return None