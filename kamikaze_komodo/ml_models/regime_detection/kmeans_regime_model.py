# kamikaze_komodo/ml_models/regime_detection/kmeans_regime_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, List

from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from datetime import datetime, timedelta, timezone

logger = get_logger(__name__)

class KMeansRegimeModel:
    """
    Identifies market regimes using K-Means clustering on specified features.
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else {}
        self.model_path = model_path
        
        self.n_clusters = int(self.params.get('num_clusters', 3))
        # Features string from config, e.g., "volatility_20d,atr_14d_percentage"
        features_str = self.params.get('featuresforclustering', 'volatility_20d,atr_14d_percentage')
        self.features_for_clustering = [f.strip() for f in features_str.split(',')]

        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_centers_: Optional[np.ndarray] = None # To store cluster centers post-training for interpretation

        if model_path:
            self.load_model(model_path)
        logger.info(f"KMeansRegimeModel initialized. Clusters: {self.n_clusters}, Features: {self.features_for_clustering}, Model Path: {model_path}")

    def _calculate_feature_volatility_X_day(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculates X-day rolling volatility of log returns."""
        if 'close' not in data.columns or len(data) < window:
            return pd.Series(np.nan, index=data.index)
        log_returns = np.log(data['close'] / data['close'].shift(1))
        return log_returns.rolling(window=window).std() * np.sqrt(window) # Annualize for context if daily, or use raw

    def _calculate_feature_atr_X_day_percentage(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculates X-day ATR as a percentage of closing price."""
        if not all(col in data.columns for col in ['high', 'low', 'close']) or len(data) < window:
            return pd.Series(np.nan, index=data.index)
        try:
            import pandas_ta as ta
            atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=window)
            if atr is None or data['close'].rolling(window=window).min().eq(0).any(): # Avoid division by zero
                 return pd.Series(np.nan, index=data.index)
            atr_percentage = (atr / data['close']) * 100
            return atr_percentage
        except ImportError:
            logger.warning("pandas_ta not found for ATR calculation in KMeansRegimeModel.")
            return pd.Series(np.nan, index=data.index)
        except Exception as e:
            logger.error(f"Error calculating ATR%: {e}")
            return pd.Series(np.nan, index=data.index)


    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()
        df = data.copy()
        
        generated_features = pd.DataFrame(index=df.index)

        for feature_name in self.features_for_clustering:
            if feature_name.startswith('volatility_') and feature_name.endswith('d'):
                try:
                    window = int(feature_name.split('_')[1][:-1])
                    generated_features[feature_name] = self._calculate_feature_volatility_X_day(df, window)
                except ValueError:
                    logger.warning(f"Could not parse window for volatility feature: {feature_name}")
            elif feature_name.startswith('atr_') and feature_name.endswith('d_percentage'):
                try:
                    window = int(feature_name.split('_')[1][:-1]) # atr_14d -> 14
                    generated_features[feature_name] = self._calculate_feature_atr_X_day_percentage(df, window)
                except ValueError:
                     logger.warning(f"Could not parse window for ATR feature: {feature_name}")
            else:
                logger.warning(f"Unsupported feature definition for Kmeans clustering: {feature_name}")
        
        generated_features.dropna(inplace=True)
        return generated_features

    def train(self, historical_data: pd.DataFrame):
        logger.info(f"Starting KMeans Regime Model training. Data shape: {historical_data.shape}")
        feature_df = self.create_features(historical_data)

        if feature_df.empty or len(feature_df) < self.n_clusters:
            logger.error("Not enough data points after feature creation to train KMeans model.")
            return

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_df)

        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        try:
            self.model.fit(scaled_features)
            self.cluster_centers_ = self.scaler.inverse_transform(self.model.cluster_centers_) # Store unscaled centers
            logger.info(f"KMeans Regime Model training completed. Inertia: {self.model.inertia_:.2f}")
            logger.info(f"Unscaled Cluster Centers:\n{self.cluster_centers_}")
            # Interpret clusters (e.g., by examining center values for volatility, atr_percentage)
            # For example, cluster with highest volatility could be "high volatility regime"
        except Exception as e:
            logger.error(f"Error during KMeans model training: {e}", exc_info=True)
            self.model = None
            self.scaler = None

    def predict(self, new_data: pd.DataFrame) -> Optional[int]:
        if self.model is None or self.scaler is None:
            logger.error("KMeans model or scaler not trained/loaded. Cannot predict regime.")
            return None
        
        feature_df = self.create_features(new_data)
        if feature_df.empty:
            logger.warning("No features could be created from new_data for KMeans prediction.")
            return None

        # We need to predict for the last row of feature_df
        last_features = feature_df.iloc[[-1]]
        if last_features.isnull().values.any():
            logger.warning(f"Latest features for KMeans prediction contain NaNs: {last_features}. Cannot predict.")
            return None

        scaled_features = self.scaler.transform(last_features)
        try:
            regime = self.model.predict(scaled_features)[0]
            logger.debug(f"Predicted regime for latest data: {regime}")
            return int(regime)
        except Exception as e:
            logger.error(f"Error during KMeans regime prediction: {e}", exc_info=True)
            return None

    def predict_series(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Predicts the regime for an entire DataFrame of historical data.
        Returns a Series of regimes aligned with the input DataFrame's index.
        """
        if self.model is None or self.scaler is None:
            logger.error("KMeans model or scaler not trained/loaded. Cannot predict regime series.")
            return None
        
        feature_df = self.create_features(data)
        if feature_df.empty:
            logger.warning("No features could be created from data for KMeans series prediction.")
            return pd.Series(np.nan, index=data.index)

        # Align feature_df with original data index to handle NaNs from feature creation
        aligned_feature_df = feature_df.reindex(data.index)
        
        # We can only predict where we have features. Get the valid indices.
        valid_indices = aligned_feature_df.dropna().index
        if valid_indices.empty:
            logger.warning("No valid feature rows to predict on.")
            return pd.Series(np.nan, index=data.index)

        scaled_features = self.scaler.transform(aligned_feature_df.loc[valid_indices])
        
        try:
            regimes = self.model.predict(scaled_features)
            # Create a series with the predictions, indexed correctly
            regime_series = pd.Series(regimes, index=valid_indices)
            # Reindex to match the original data's index, and forward-fill missing values
            return regime_series.reindex(data.index).ffill()
        except Exception as e:
            logger.error(f"Error during KMeans regime series prediction: {e}", exc_info=True)
            return None

    def save_model(self, path: Optional[str] = None):
        _path = path or self.model_path
        if self.model is None or self.scaler is None:
            logger.error("No KMeans model or scaler to save.")
            return
        if not _path:
            logger.error("No path specified for saving KMeans model.")
            return
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features_for_clustering': self.features_for_clustering,
                'n_clusters': self.n_clusters,
                'cluster_centers_': self.cluster_centers_
            }
            joblib.dump(model_data, _path)
            logger.info(f"KMeans Regime model saved to {_path}")
        except Exception as e:
            logger.error(f"Error saving KMeans model to {_path}: {e}", exc_info=True)

    def load_model(self, path: Optional[str] = None):
        _path = path or self.model_path
        if not _path:
            logger.debug("No path specified for loading KMeans model.")
            return
        try:
            model_data = joblib.load(_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features_for_clustering = model_data.get('features_for_clustering', self.features_for_clustering)
            self.n_clusters = model_data.get('n_clusters', self.n_clusters)
            self.cluster_centers_ = model_data.get('cluster_centers_')
            self.model_path = _path
            logger.info(f"KMeans Regime model loaded from {_path}. Clusters: {self.n_clusters}, Features: {self.features_for_clustering}")
            if self.cluster_centers_ is not None:
                 logger.info(f"Loaded Unscaled Cluster Centers:\n{self.cluster_centers_}")
        except FileNotFoundError:
            logger.error(f"KMeans model file not found at {_path}.")
            self.model = None
            self.scaler = None
        except Exception as e:
            logger.error(f"Error loading KMeans model from {_path}: {e}", exc_info=True)
            self.model = None
            self.scaler = None

async def main_kmeans_regime_example():
    if not settings:
        print("Settings not loaded, cannot run KMeans Regime model example.")
        return

    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    config_section = "KMeans_Regime_Model" # Must match config.ini section name

    if not settings.config.has_section(config_section):
        logger.error(f"Config section [{config_section}] not found. Cannot run KMeans Regime example.")
        return

    model_params = settings.get_strategy_params(config_section)
    _model_base_path = model_params.get('modelsavepath', 'kamkaze_komodo/ml_models/trained_models/regime')
    _model_filename = model_params.get('modelfilename', f"kmeans_regime_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
    
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    if not os.path.isabs(_model_base_path):
        model_save_path_dir = os.path.join(script_dir, _model_base_path)
    else:
        model_save_path_dir = _model_base_path
    if not os.path.exists(model_save_path_dir):
        os.makedirs(model_save_path_dir, exist_ok=True)
    model_full_path = os.path.join(model_save_path_dir, _model_filename)

    # --- Training ---
    logger.info("--- KMeans Regime Model Training Example ---")
    regime_model_trainer = KMeansRegimeModel(params=model_params)
    
    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()
    training_days = int(model_params.get('trainingdayshistory', 1095))
    start_dt = datetime.now(timezone.utc) - timedelta(days=training_days)
    end_dt = datetime.now(timezone.utc)
    
    bars = db_manager.retrieve_bar_data(symbol, timeframe, start_dt, end_dt)
    if not bars or len(bars) < 100: # Need substantial data for regime features
        logger.info(f"Fetching fresh data for KMeans training for {symbol}...")
        bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_dt, end_dt)
        if bars: db_manager.store_bar_data(bars)
    await data_fetcher.close()
    db_manager.close()

    if not bars or len(bars) < 100:
        logger.error(f"Not enough data ({len(bars)} bars) for KMeans training.")
        return

    training_df = pd.DataFrame([b.model_dump() for b in bars])
    training_df['timestamp'] = pd.to_datetime(training_df['timestamp'])
    training_df.set_index('timestamp', inplace=True)
    training_df.sort_index(inplace=True)

    regime_model_trainer.train(training_df)
    if regime_model_trainer.model:
        regime_model_trainer.save_model(model_full_path)

    # --- Prediction ---
    logger.info("--- KMeans Regime Model Prediction Example ---")
    if not os.path.exists(model_full_path):
        logger.error("Trained KMeans model not found. Skipping prediction example.")
        return

    regime_model_predictor = KMeansRegimeModel(model_path=model_full_path, params=model_params)
    if regime_model_predictor.model is None:
        logger.error("Could not load KMeans model for prediction.")
        return

    # Use the last N bars of training_df for prediction example (or fetch fresh small segment)
    if len(training_df) > 50:
        prediction_data_segment = training_df.tail(50) # Use recent history to get features for the last point
        predicted_regime = regime_model_predictor.predict(prediction_data_segment)
        if predicted_regime is not None:
            logger.info(f"Predicted regime for {symbol} ({timeframe}) on latest data: {predicted_regime}")
        else:
            logger.warning(f"Could not get KMeans regime prediction for {symbol} ({timeframe}).")
    else:
        logger.warning("Not enough data in training_df to run prediction example.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_kmeans_regime_example())