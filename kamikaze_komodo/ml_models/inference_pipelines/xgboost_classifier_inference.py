# FILE: kamikaze_komodo/ml_models/inference_pipelines/xgboost_classifier_inference.py
import pandas as pd
from typing import Optional, Dict, Any
import os
import numpy as np
from kamikaze_komodo.ml_models.price_forecasting.xgboost_classifier_forecaster import XGBoostClassifierForecaster
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings

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
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
        
        if not os.path.isabs(_model_base_path):
            self.model_load_path_dir = os.path.join(script_dir, _model_base_path)
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

async def main_xgboost_inference_example():
    from kamikaze_komodo.data_handling.database_manager import DatabaseManager
    from datetime import datetime, timedelta, timezone
    
    if not settings:
        print("Settings not loaded, cannot run XGBoost Classifier inference example.")
        return
        
    symbol_to_predict = settings.default_symbol
    timeframe_to_predict = settings.default_timeframe
    
    if not settings.config.has_section("XGBoost_Classifier_Forecaster"):
        logger.error("Config section [XGBoost_Classifier_Forecaster] not found. Cannot run inference example.")
        return
        
    inference_engine = XGBoostClassifierInference(symbol=symbol_to_predict, timeframe=timeframe_to_predict)
    if inference_engine.forecaster.model is None:
        logger.error("Failed to load XGBoost model for inference. Exiting example.")
        return
        
    db_manager = DatabaseManager()
    hist_days = 30 
    start_dt = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_dt = datetime.now(timezone.utc)
    
    bars = db_manager.retrieve_bar_data(symbol_to_predict, timeframe_to_predict, start_dt, end_dt)
    db_manager.close()
    
    if not bars or len(bars) < 50: # Need enough for feature generation
        logger.error(f"Not enough recent data ({len(bars)} bars) to make an XGBoost prediction example.")
        return
        
    data_df = pd.DataFrame([b.model_dump() for b in bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)
    data_df.sort_index(inplace=True)
    
    logger.info(f"Making XGBoost prediction for {symbol_to_predict} ({timeframe_to_predict}) using last {len(data_df)} bars.")
    prediction_dict = inference_engine.get_prediction(data_df)
    
    if prediction_dict:
        logger.info(f"XGBoost Prediction for {symbol_to_predict} ({timeframe_to_predict}): {prediction_dict}")
    else:
        logger.warning(f"Could not get an XGBoost prediction for {symbol_to_predict} ({timeframe_to_predict}).")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_xgboost_inference_example())