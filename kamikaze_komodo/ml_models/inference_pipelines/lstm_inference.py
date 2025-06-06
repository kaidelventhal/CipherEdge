# FILE: kamikaze_komodo/ml_models/inference_pipelines/lstm_inference.py
import pandas as pd
from typing import Optional
import os
from kamikaze_komodo.ml_models.price_forecasting.lstm_model import LSTMForecaster
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class LSTMInference:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "LSTM_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        
        model_params_config = settings.get_strategy_params(model_config_section)
        
        _model_base_path = model_params_config.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = model_params_config.get('modelfilename', f"lstm_{symbol.replace('/', '_').lower()}_{timeframe}.pth")
        
        if not os.path.isabs(_model_base_path):
            self.model_load_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_load_path_dir = _model_base_path
            
        self.model_full_load_path = os.path.join(self.model_load_path_dir, _model_filename)
        self.forecaster = LSTMForecaster(model_path=self.model_full_load_path, params=model_params_config)
        
        if self.forecaster.model is None:
            logger.warning(f"LSTMInference: Model could not be loaded from {self.model_full_load_path}. Predictions will not be available.")

    def get_prediction(self, current_data_history: pd.DataFrame) -> Optional[float]:
        """
        Gets a single prediction based on the current data history.
        """
        if self.forecaster.model is None:
            logger.warning("No model loaded, cannot get prediction.")
            return None
        if current_data_history.empty:
            logger.warning("Data history is empty, cannot get prediction.")
            return None
            
        prediction_output = self.forecaster.predict(current_data_history)
        
        if prediction_output is None:
            return None
            
        return float(prediction_output)