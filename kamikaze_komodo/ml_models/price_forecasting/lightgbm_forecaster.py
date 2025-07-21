# FILE: kamikaze_komodo/ml_models/price_forecasting/lightgbm_forecaster.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib 
from typing import Optional, Dict, Any, List, Union
from kamikaze_komodo.ml_models.price_forecasting.base_forecaster import BasePriceForecaster
from kamikaze_komodo.ml_models.feature_engineering import (
    add_lag_features, add_rolling_window_features, add_technical_indicators,
    add_sentiment_features, add_cyclical_time_features
)
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class LightGBMForecaster(BasePriceForecaster):
    """
    LightGBM-based price forecaster.
    Predicts price movement (e.g., next bar's close relative to current).
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_path, params)
        self.default_lgbm_params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 100,
            'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
            'boosting_type': 'gbdt',
        }
        config_lgbm_params = {k.replace('lgbm_params_', ''): v for k, v in self.params.items() if k.startswith('lgbm_params_')}
        self.lgbm_params = {**self.default_lgbm_params, **config_lgbm_params}
        
        if not self.model and self.model_path:
            self.load_model(self.model_path)

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            logger.warning("Data for feature creation is empty.")
            return pd.DataFrame()
            
        df = data.copy()
        df = add_lag_features(df)
        df = add_rolling_window_features(df)
        df = add_technical_indicators(df)
        df = add_sentiment_features(df)
        df = add_cyclical_time_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def train(self, historical_data: pd.DataFrame, target_column: str = 'close_change_lag_1_future', feature_columns_to_use: Optional[List[str]] = None):
        logger.info(f"Starting LightGBM training for target '{target_column}'. Data shape: {historical_data.shape}")
        df = historical_data.copy()
        
        if target_column == 'close_change_lag_1_future':
            df['target'] = (df['close'].shift(-1) / df['close']) - 1
        else:
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in data.")
                return
            df['target'] = df[target_column]

        df_with_all_features = self.create_features(df)
        
        if feature_columns_to_use:
            actual_features_present = [col for col in feature_columns_to_use if col in df_with_all_features.columns]
        else:
            actual_features_present = [col for col in df_with_all_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'target']]

        if not actual_features_present:
            logger.error("No valid feature columns found for training. Cannot train.")
            return

        # FIX: Combine features and target into a new DF, then drop NaNs to avoid losing all data
        training_df = df_with_all_features[actual_features_present + ['target']].copy()
        training_df.dropna(inplace=True)

        if training_df.empty:
            logger.error("Feature matrix or target vector is empty after processing. Training cannot proceed.")
            return

        X_train = training_df[actual_features_present]
        y_train = training_df['target']
        
        self.model = lgb.LGBMRegressor(**self.lgbm_params)
        logger.info(f"Training LightGBM model with {len(X_train)} samples. Features: {list(X_train.columns)}")
        try:
            self.model.fit(X_train, y_train)
            logger.info("LightGBM model training completed.")
            self.trained_feature_columns_ = list(X_train.columns)
        except Exception as e:
            logger.error(f"Error during LightGBM model training: {e}", exc_info=True)
            self.model = None

    def predict(self, new_data: pd.DataFrame, feature_columns_to_use: Optional[list] = None) -> Union[pd.Series, float, None]:
        if self.model is None:
            logger.error("Model not loaded or trained. Cannot make predictions.")
            return None
        
        df_with_all_features = self.create_features(new_data)
        
        cols_for_prediction = feature_columns_to_use or getattr(self, 'trained_feature_columns_', None)
        if not cols_for_prediction:
            logger.error("No feature columns determined for prediction.")
            return None

        cols_for_prediction = [col for col in cols_for_prediction if col in df_with_all_features.columns]
        
        X_new = df_with_all_features[cols_for_prediction].copy()
        if X_new.empty:
            logger.warning("Feature matrix is empty after selection. Cannot predict.")
            return None
        
        try:
            predictions = self.model.predict(X_new)
            if len(predictions) == 0: return None
            return predictions[-1] if isinstance(predictions, np.ndarray) else predictions
        except Exception as e:
            logger.error(f"Error during LightGBM prediction: {e}", exc_info=True)
            return None

    def save_model(self, path: Optional[str] = None):
        _path = path or self.model_path
        if self.model is None or not _path:
            logger.error(f"Model not available or path not specified. Cannot save.")
            return
        try:
            model_and_features = {
                'model': self.model,
                'feature_columns': getattr(self, 'trained_feature_columns_', None)
            }
            joblib.dump(model_and_features, _path)
            logger.info(f"LightGBM model and feature columns saved to {_path}")
        except Exception as e:
            logger.error(f"Error saving LightGBM model to {_path}: {e}", exc_info=True)

    def load_model(self, path: Optional[str] = None):
        _path = path or self.model_path
        if not _path:
            logger.debug("No path specified for loading the model.")
            return
        try:
            model_and_features = joblib.load(_path)
            self.model = model_and_features['model']
            self.trained_feature_columns_ = model_and_features.get('feature_columns') 
            self.model_path = _path 
            logger.info(f"LightGBM model loaded from {_path}. Trained features: {self.trained_feature_columns_}")
        except FileNotFoundError:
            logger.error(f"LightGBM model file not found at {_path}.")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading LightGBM model from {_path}: {e}", exc_info=True)
            self.model = None