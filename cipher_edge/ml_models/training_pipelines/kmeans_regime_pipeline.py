import pandas as pd
from datetime import datetime, timedelta, timezone
import os

from cipher_edge.ml_models.regime_detection.kmeans_regime_model import KMeansRegimeModel
from cipher_edge.data_handling.data_fetcher import DataFetcher
from cipher_edge.data_handling.database_manager import DatabaseManager
from cipher_edge.app_logger import get_logger
from cipher_edge.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class KMeansRegimeTrainingPipeline:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "KMeans_Regime_Model"):
        if not settings:
            logger.critical("Settings not loaded. KMeansRegimeTrainingPipeline cannot be initialized.")
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_config_section = model_config_section
        
        self.model_params = settings.get_strategy_params(model_config_section)
        if not self.model_params:
            logger.warning(f"No parameters found for config section [{model_config_section}]. Using defaults for KMeansRegimeModel if any.")
            self.model_params = {} # Ensure it's a dict
        
        _model_base_path = self.model_params.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = self.model_params.get('modelfilename', f"kmeans_regime_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
        
        if not os.path.isabs(_model_base_path):
            self.model_save_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_save_path_dir = _model_base_path
            
        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir, exist_ok=True)
            logger.info(f"Created directory for trained KMeans regime models: {self.model_save_path_dir}")
            
        self.model_full_save_path = os.path.join(self.model_save_path_dir, _model_filename)
        
        self.regime_model = KMeansRegimeModel(model_path=None, params=self.model_params) # Don't load, we are training
        logger.info(f"KMeansRegimeTrainingPipeline initialized for {symbol} ({timeframe}). Model will be saved to: {self.model_full_save_path}")

    async def fetch_training_data(self, days_history: int) -> pd.DataFrame:
        db_manager = DatabaseManager()
        data_fetcher = DataFetcher()
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days_history)
        end_date = datetime.now(timezone.utc)
        logger.info(f"Attempting to retrieve data from DB for KMeans training: {self.symbol} ({self.timeframe}) from {start_date} to {end_date}")
        historical_bars = db_manager.retrieve_bar_data(self.symbol, self.timeframe, start_date, end_date)
        
        min_bars_for_features = int(self.model_params.get('minbarsfortraining', 100)) 

        if not historical_bars or len(historical_bars) < min_bars_for_features:
            logger.info(f"Insufficient data in DB ({len(historical_bars)} bars found, need {min_bars_for_features}). Fetching fresh data for KMeans training...")
            historical_bars = await data_fetcher.fetch_historical_data_for_period(self.symbol, self.timeframe, start_date, end_date)
            if historical_bars:
                db_manager.store_bar_data(historical_bars)
            else:
                logger.error("Failed to fetch training data for KMeans.")
                await data_fetcher.close()
                db_manager.close()
                return pd.DataFrame()
        
        await data_fetcher.close()
        db_manager.close()

        if not historical_bars or len(historical_bars) < min_bars_for_features:
            logger.error(f"Still not enough data ({len(historical_bars)} bars) for KMeans training after fetch attempt. Need {min_bars_for_features}.")
            return pd.DataFrame()
            
        data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
        if data_df.empty:
            logger.error("DataFrame is empty after converting BarData list.")
            return pd.DataFrame()
            
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df.set_index('timestamp', inplace=True)
        data_df.sort_index(inplace=True)
        
        logger.info(f"Fetched {len(data_df)} bars for KMeans training {self.symbol} ({self.timeframe}).")
        return data_df

    async def run_training(self):
        """
        Fetches data, trains the KMeans regime model, and saves it.
        """
        days_history = int(self.model_params.get('trainingdayshistory', 1095))
        if days_history <=0:
            logger.error(f"TrainingDaysHistory ({days_history}) must be positive. Cannot run training.")
            return

        historical_df = await self.fetch_training_data(days_history=days_history)

        if historical_df.empty:
            logger.error("Cannot run KMeans training, no historical data was retrieved or processed.")
            return

        logger.info(f"Starting KMeans regime model training using {self.regime_model.__class__.__name__}...")
        
        self.regime_model.train(historical_df) 
        
        if self.regime_model.model and self.regime_model.scaler:
            self.regime_model.save_model(self.model_full_save_path)
            logger.info(f"KMeans regime model training completed and model saved to {self.model_full_save_path}.")
        else:
            logger.error("KMeans regime model training did not produce a valid model or scaler. Model not saved.")