# kamikaze_komodo/ml_models/inference_pipelines/lightgbm_inference.py
import pandas as pd
from typing import Optional, Union
import os
import numpy as np 

from kamikaze_komodo.ml_models.price_forecasting.lightgbm_forecaster import LightGBMForecaster
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings

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

        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
        
        if not os.path.isabs(_model_base_path):
            self.model_load_path_dir = os.path.join(script_dir, _model_base_path)
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

        # The LightGBMForecaster's predict method expects a DataFrame, even for a single prediction.
        # It will use its create_features method, which typically uses lags.
        # So, pass the recent history. `predict` will internally call `create_features`.
        
        # Determine feature columns to use for prediction
        feature_cols_to_pass = None
        # Check if 'feature_columns' is in the forecaster's parameters (e.g., from config)
        feature_cols_str_from_params = self.forecaster.params.get('feature_columns')
        if isinstance(feature_cols_str_from_params, str) and feature_cols_str_from_params:
            feature_cols_to_pass = [col.strip() for col in feature_cols_str_from_params.split(',')]
        elif isinstance(feature_cols_str_from_params, list):
             feature_cols_to_pass = feature_cols_str_from_params
        # If not in params, `LightGBMForecaster.predict` will use `self.trained_feature_columns_` if available.
        # Or, `feature_columns_to_use` can be None if the model should use its internal defaults/trained features.

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


async def main_inference_example():
    from kamikaze_komodo.data_handling.database_manager import DatabaseManager
    from datetime import datetime, timedelta, timezone
    
    if not settings:
        print("Settings not loaded, cannot run LightGBM inference example.")
        return

    symbol_to_predict = settings.default_symbol
    timeframe_to_predict = settings.default_timeframe

    if not settings.config.has_section("LightGBM_Forecaster"):
        logger.error("Config section [LightGBM_Forecaster] not found. Cannot run inference.")
        return

    inference_engine = LightGBMInference(symbol=symbol_to_predict, timeframe=timeframe_to_predict)
    if inference_engine.forecaster.model is None:
        logger.error("Failed to load model for inference. Exiting example.")
        return

    db_manager = DatabaseManager()
    hist_days = 30 
    start_dt = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_dt = datetime.now(timezone.utc)
    
    bars = db_manager.retrieve_bar_data(symbol_to_predict, timeframe_to_predict, start_dt, end_dt)
    db_manager.close()

    if not bars or len(bars) < 20: 
        logger.error(f"Not enough recent data ({len(bars)} bars) to make a prediction example.")
        return
        
    data_df = pd.DataFrame([b.model_dump() for b in bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)
    data_df.sort_index(inplace=True)
    
    logger.info(f"Making prediction for {symbol_to_predict} ({timeframe_to_predict}) using last {len(data_df)} bars.")
    prediction = inference_engine.get_prediction(data_df)

    if prediction is not None:
        logger.info(f"Prediction for {symbol_to_predict} ({timeframe_to_predict}): {prediction:.6f}")
    else:
        logger.warning(f"Could not get a prediction for {symbol_to_predict} ({timeframe_to_predict}).")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_inference_example())