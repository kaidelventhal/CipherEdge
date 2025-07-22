# FILE: kamikaze_komodo/ml_models/training_pipelines/lstm_pipeline.py
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from kamikaze_komodo.ml_models.price_forecasting.lstm_model import LSTMForecaster
from kamikaze_komodo.data_handling.data_handler import DataHandler
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class LSTMTrainingPipeline:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "LSTM_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_config_section = model_config_section
        
        self.model_params = settings.get_strategy_params(model_config_section)
        
        _model_base_path = self.model_params.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = self.model_params.get('modelfilename', f"lstm_{symbol.replace('/', '_').lower()}_{timeframe}.pth")
        
        if not os.path.isabs(_model_base_path):
            self.model_save_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_save_path_dir = _model_base_path

        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir, exist_ok=True)
            
        self.model_full_save_path = os.path.join(self.model_save_path_dir, _model_filename)
        
        self.forecaster = LSTMForecaster(params=self.model_params)
        logger.info(f"LSTM Training Pipeline initialized. Model will be saved to: {self.model_full_save_path}")

    async def fetch_training_data(self, days_history: int) -> pd.DataFrame:
        data_handler = DataHandler()
        start_date = datetime.now(timezone.utc) - timedelta(days=days_history)
        end_date = datetime.now(timezone.utc)
        data_df = await data_handler.get_prepared_data(
            self.symbol, self.timeframe, start_date, end_date,
            needs_funding_rate=True, needs_sentiment=settings.use_sentiment_in_models if settings else True
        )
        await data_handler.close()
        
        if not data_df.empty:
            logger.info(f"Fetched and prepared {len(data_df)} bars for LSTM training.")
        return data_df

    async def run_training(self, tune_hyperparameters: bool = False):
        # Note: Hyperparameter tuning for LSTMs is more complex and computationally
        # expensive than for tree-based models. This is a simplified placeholder.
        if tune_hyperparameters:
            logger.warning("Hyperparameter tuning for LSTM is not fully implemented in this phase. Running with default parameters.")

        days_history = int(self.model_params.get('trainingdayshistory', 730))
        historical_df = await self.fetch_training_data(days_history=days_history)

        if historical_df.empty:
            logger.error("Cannot run LSTM training, no historical data.")
            return

        target_col_name = self.model_params.get('targetcolumnname', 'close_change_lag_1_future')
        feature_cols_str = self.model_params.get('featurecolumns')
        feature_columns = [col.strip() for col in feature_cols_str.split(',')] if feature_cols_str else None
        
        self.forecaster.train(historical_df, target_column=target_col_name, feature_columns=feature_columns)
        
        if self.forecaster.model:
            self.forecaster.save_model(self.model_full_save_path)
        else:
            logger.error("LSTM training did not produce a model. Model not saved.")