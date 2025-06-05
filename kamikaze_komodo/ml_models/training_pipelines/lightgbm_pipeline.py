# kamikaze_komodo/ml_models/training_pipelines/lightgbm_pipeline.py
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from kamikaze_komodo.ml_models.price_forecasting.lightgbm_forecaster import LightGBMForecaster
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings
logger = get_logger(__name__)
class LightGBMTrainingPipeline:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "LightGBM_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_config_section = model_config_section
        
        # Get model-specific params from config.ini, e.g. under [LightGBM_Forecaster]
        self.model_params = settings.get_strategy_params(model_config_section) # Re-using this method, name is a bit off
        
        # Determine model save path from settings
        # Example: ModelSavePath = models/trained_models/lightgbm/
        #          ModelFileName = lgbm_btc_usd_1h.joblib
        _model_base_path = self.model_params.get('modelsavepath', 'ml_models/trained_models') # default path in project
        _model_filename = self.model_params.get('modelfilename', f"lgbm_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
        # Ensure the base path is relative to the project root if not absolute
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # kamikaze_komodo module dir
        if not os.path.isabs(_model_base_path):
            self.model_save_path_dir = os.path.join(script_dir, _model_base_path)
        else:
            self.model_save_path_dir = _model_base_path
        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir, exist_ok=True)
            logger.info(f"Created directory for trained models: {self.model_save_path_dir}")
            
        self.model_full_save_path = os.path.join(self.model_save_path_dir, _model_filename)
        
        self.forecaster = LightGBMForecaster(params=self.model_params) # Pass specific model params
        logger.info(f"LightGBM Training Pipeline initialized for {symbol} ({timeframe}). Model will be saved to: {self.model_full_save_path}")
    async def fetch_training_data(self, days_history: int = 730) -> pd.DataFrame: # Default to ~2 years
        db_manager = DatabaseManager()
        data_fetcher = DataFetcher()
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days_history)
        end_date = datetime.now(timezone.utc)
        # Attempt to retrieve from DB first
        logger.info(f"Attempting to retrieve data from DB for {self.symbol} ({self.timeframe}) from {start_date} to {end_date}")
        historical_bars = db_manager.retrieve_bar_data(self.symbol, self.timeframe, start_date, end_date)
        
        required_bars = self.model_params.get('min_bars_for_training', 200) # Get from config or default
        if not historical_bars or len(historical_bars) < required_bars:
            logger.info(f"Insufficient data in DB ({len(historical_bars)} bars found). Fetching fresh data from exchange...")
            historical_bars = await data_fetcher.fetch_historical_data_for_period(self.symbol, self.timeframe, start_date, end_date)
            if historical_bars:
                db_manager.store_bar_data(historical_bars) # Store fresh data
            else:
                logger.error("Failed to fetch training data.")
                await data_fetcher.close()
                db_manager.close()
                return pd.DataFrame()
        
        await data_fetcher.close()
        db_manager.close()
        if not historical_bars:
            logger.error("No historical bars available for training.")
            return pd.DataFrame()
        data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df.set_index('timestamp', inplace=True)
        data_df.sort_index(inplace=True)
        
        logger.info(f"Fetched {len(data_df)} bars for training {self.symbol} ({self.timeframe}).")
        return data_df
    async def run_training(self):
        days_history = int(self.model_params.get('training_days_history', 730))
        historical_df = await self.fetch_training_data(days_history=days_history)
        if historical_df.empty:
            logger.error("Cannot run training, no historical data.")
            return
        target_col_name = self.model_params.get('target_column_name', 'close_change_lag_1_future') # e.g. (close_t+1 / close_t) -1
        
        # Feature columns can be defined in config.ini [LightGBM_Forecaster] Features = col1,col2,col3
        # Or forecaster.create_features will use its defaults if None
        feature_cols_str = self.model_params.get('feature_columns')
        feature_columns = [col.strip() for col in feature_cols_str.split(',')] if feature_cols_str else None
        
        logger.info(f"Starting training with target: '{target_col_name}', features: {feature_columns if feature_columns else 'default in forecaster'}")
        self.forecaster.train(historical_df, target_column=target_col_name, feature_columns=feature_columns)
        
        if self.forecaster.model:
            self.forecaster.save_model(self.model_full_save_path)
        else:
            logger.error("Training did not produce a model. Model not saved.")
async def main_train_lightgbm():
    if not settings:
        print("Settings not loaded, cannot run LightGBM training example.")
        return
    
    # Example: Train for default symbol and timeframe from settings
    symbol_to_train = settings.default_symbol
    timeframe_to_train = settings.default_timeframe
    
    # Check if LightGBM_Forecaster section exists, otherwise training won't run properly
    if not settings.config.has_section("LightGBM_Forecaster"):
        logger.error("Config section [LightGBM_Forecaster] not found in config.ini. Cannot run training pipeline.")
        logger.error("Please add a section like:\n[LightGBM_Forecaster]\nModelSavePath = ml_models/trained_models\nModelFileName = lgbm_model.joblib\nTargetColumnName=close_change_lag_1_future\nTrainingDaysHistory=730")
        return
    logger.info(f"Starting LightGBM training pipeline for {symbol_to_train} ({timeframe_to_train})...")
    pipeline = LightGBMTrainingPipeline(symbol=symbol_to_train, timeframe=timeframe_to_train)
    await pipeline.run_training()
    logger.info(f"LightGBM training pipeline finished for {symbol_to_train} ({timeframe_to_train}).")
if __name__ == "__main__":
    import asyncio
    # This allows running the training pipeline directly
    # Ensure that if you run this, your config.ini has the [LightGBM_Forecaster] section
    # And your base project directory is structured correctly for pathing.
    # Example: python -m kamikaze_komodo.ml_models.training_pipelines.lightgbm_pipeline
    
    # Check for GOOGLE_APPLICATION_CREDENTIALS if any part of settings/data fetching indirectly uses it (though not typical for ccxt basic fetch)
    # Primarily, this is for VertexAI, not directly for this LightGBM part.
    asyncio.run(main_train_lightgbm())