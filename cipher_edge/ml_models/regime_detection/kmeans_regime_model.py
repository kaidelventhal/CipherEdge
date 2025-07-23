import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, List

from cipher_edge.app_logger import get_logger
from cipher_edge.config.settings import settings
from cipher_edge.data_handling.data_fetcher import DataFetcher
from cipher_edge.data_handling.database_manager import DatabaseManager
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
        features_str = self.params.get('featuresforclustering', 'volatility_20d,atr_14d_percentage')
        self.features_for_clustering = [f.strip() for f in features_str.split(',')]

        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_centers_: Optional[np.ndarray] = None 
        self.regime_labels: Optional[Dict[int, str]] = None

        if model_path:
            self.load_model(model_path)
        logger.info(f"KMeansRegimeModel initialized. Clusters: {self.n_clusters}, Features: {self.features_for_clustering}, Model Path: {model_path}")

    def _calculate_feature_volatility_X_day(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculates X-day rolling volatility of log returns."""
        if 'close' not in data.columns or len(data) < window:
            return pd.Series(np.nan, index=data.index)
        log_returns = np.log(data['close'] / data['close'].shift(1))
        return log_returns.rolling(window=window).std() * np.sqrt(window) 

    def _calculate_feature_atr_X_day_percentage(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculates X-day ATR as a percentage of closing price."""
        if not all(col in data.columns for col in ['high', 'low', 'close']) or len(data) < window:
            return pd.Series(np.nan, index=data.index)
        try:
            import pandas_ta as ta
            atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=window)
            if atr is None or data['close'].rolling(window=window).min().eq(0).any():
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
        
        return generated_features

    def train(self, historical_data: pd.DataFrame):
        logger.info(f"Starting KMeans Regime Model training. Data shape: {historical_data.shape}")
        feature_df = self.create_features(historical_data).dropna()

        if feature_df.empty or len(feature_df) < self.n_clusters:
            logger.error("Not enough data points after feature creation to train KMeans model.")
            return

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_df)

        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        try:
            self.model.fit(scaled_features)
            self.cluster_centers_ = self.scaler.inverse_transform(self.model.cluster_centers_) 
            logger.info(f"KMeans Regime Model training completed. Inertia: {self.model.inertia_:.2f}")
            self.interpret_regimes()
        except Exception as e:
            logger.error(f"Error during KMeans model training: {e}", exc_info=True)
            self.model = None
            self.scaler = None

    def interpret_regimes(self) -> Optional[Dict[int, str]]:
        """
        Interprets and labels the clusters based on their feature characteristics.
        Assumes the first feature is related to volatility for labeling.
        Returns:
            Dict[int, str]: A mapping from cluster index to regime label (e.g., {0: 'Ranging', 1: 'Trending', 2: 'High-Volatility/Choppy'}).
        """
        if self.cluster_centers_ is None:
            logger.warning("Model has not been trained or loaded. Cannot interpret regimes.")
            return None

        volatility_feature_index = 0
        centers_volatility = self.cluster_centers_[:, volatility_feature_index]
        
        sorted_indices = np.argsort(centers_volatility)
        
        if self.n_clusters == 3:
            self.regime_labels = {
                sorted_indices[0]: "Ranging",
                sorted_indices[1]: "Trending",
                sorted_indices[2]: "High-Volatility/Choppy"
            }
        else: 
            self.regime_labels = {idx: f"Regime_{i+1}" for i, idx in enumerate(sorted_indices)}
            logger.warning(f"Automatic regime interpretation is set up for 3 clusters. Found {self.n_clusters}, using generic labels.")

        logger.info(f"Interpreted Regime Labels (based on first feature '{self.features_for_clustering[0]}'): {self.regime_labels}")
        return self.regime_labels

    def predict_regimes_for_dataframe(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Predicts the regime for each row in a historical DataFrame.
        """
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not trained/loaded. Cannot predict regimes for DataFrame.")
            return None

        feature_df = self.create_features(data)
        
        regime_series = pd.Series(np.nan, index=data.index)
        
        valid_feature_df = feature_df.dropna()
        if valid_feature_df.empty:
            logger.warning("No valid features could be calculated for the provided DataFrame.")
            return regime_series

        scaled_features = self.scaler.transform(valid_feature_df)
        predictions = self.model.predict(scaled_features)
        
        regime_series.loc[valid_feature_df.index] = predictions
        
        return regime_series

    def predict(self, new_data: pd.DataFrame) -> Optional[int]:
        if self.model is None or self.scaler is None:
            logger.error("KMeans model or scaler not trained/loaded. Cannot predict regime.")
            return None
        
        feature_df = self.create_features(new_data)
        if feature_df.empty:
            logger.warning("No features could be created from new_data for KMeans prediction.")
            return None

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
                'cluster_centers_': self.cluster_centers_,
                'regime_labels': self.regime_labels
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
            self.regime_labels = model_data.get('regime_labels')
            self.model_path = _path
            logger.info(f"KMeans Regime model loaded from {_path}. Clusters: {self.n_clusters}, Features: {self.features_for_clustering}")
            if self.cluster_centers_ is not None:
                logger.info(f"Loaded Unscaled Cluster Centers:\n{self.cluster_centers_}")
            if self.regime_labels is not None:
                logger.info(f"Loaded Regime Labels: {self.regime_labels}")
        except FileNotFoundError:
            logger.error(f"KMeans model file not found at {_path}.")
            self.model = None
            self.scaler = None
        except Exception as e:
            logger.error(f"Error loading KMeans model from {_path}: {e}", exc_info=True)
            self.model = None
            self.scaler = None