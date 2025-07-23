# FILE: cipher_edge/ml_models/price_forecasting/xgboost_classifier_forecaster.py
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder

from cipher_edge.ml_models.price_forecasting.base_forecaster import BasePriceForecaster
from cipher_edge.ml_models.feature_engineering import (
    add_lag_features, add_rolling_window_features, add_technical_indicators,
    add_sentiment_features, add_cyclical_time_features,
    add_advanced_indicators, add_market_structure_features
)
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class XGBoostClassifierForecaster(BasePriceForecaster):
    """
    XGBoost-based classifier for price movement prediction (UP, DOWN, SIDEWAYS).
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.label_encoder = LabelEncoder()
        self.trained_feature_columns_: Optional[List[str]] = None
        
        super().__init__(model_path, params)
        
        self.default_xgb_params = {
            'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'n_estimators': 100,
            'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'use_label_encoder': False, 'seed': 42,
        }
        config_xgb_params = {k.replace('xgb_params_', ''): v for k, v in self.params.items() if k.startswith('xgb_params_')}
        self.xgb_params = {**self.default_xgb_params, **config_xgb_params}
        
        self.num_class = int(self.params.get('num_classes', 3))
        self.xgb_params['num_class'] = self.num_class
        
        if not self.model and self.model_path:
            self.load_model(self.model_path)

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty: return pd.DataFrame()
        df = data.copy()
        df = add_lag_features(df)
        df = add_rolling_window_features(df)
        df = add_technical_indicators(df)
        df = add_advanced_indicators(df)
        df = add_market_structure_features(df)
        if 'sentiment_score' in df.columns:
            df = add_sentiment_features(df)
        df = add_cyclical_time_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _define_target(self, data: pd.DataFrame, thresholds: Optional[Tuple[float, float]] = (-0.001, 0.001)) -> pd.Series:
        """Defines target classes: 0 (UP), 1 (DOWN), 2 (SIDEWAYS)."""
        future_returns = data['close'].pct_change(1).shift(-1)
        if thresholds is None: thresholds = (-0.001, 0.001)
        lower_thresh, upper_thresh = thresholds

        target = pd.Series(2, index=data.index) # Default to SIDEWAYS
        target[future_returns > upper_thresh] = 0 # UP
        target[future_returns < lower_thresh] = 1 # DOWN
        return target.astype(int)

    def train(self, historical_data: pd.DataFrame, target_definition: str = 'next_bar_direction', feature_columns: Optional[list] = None):
        logger.info(f"Starting XGBoost Classifier training. Data shape: {historical_data.shape}")
        df = historical_data.copy()

        return_thresholds_str = self.params.get('returnthresholds_percent', "-0.001,0.001")
        try:
            thresholds_list = [float(x.strip()) for x in return_thresholds_str.split(',')]
            return_thresholds = tuple(thresholds_list)
        except Exception:
            return_thresholds = (-0.001, 0.001)

        if target_definition == 'next_bar_direction':
            df['target'] = self._define_target(df, return_thresholds)
        else:
            logger.error(f"Unsupported target_definition: {target_definition}")
            return

        df_with_features = self.create_features(df)

        actual_features = feature_columns or [col for col in df_with_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'target']]
        
        # FIX: Combine features and target into a new DF, then drop NaNs
        training_df = df_with_features[actual_features + ['target']].copy()
        training_df.dropna(inplace=True)

        if training_df.empty:
            logger.error("Feature matrix X or target vector y is empty after processing.")
            return

        X = training_df[actual_features]
        y = training_df['target'].astype(int)

        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)

        self.model = xgb.XGBClassifier(**self.xgb_params)
        logger.info(f"Training XGBoostClassifier with {len(X)} samples. Features: {list(X.columns)}")
        try:
            self.model.fit(X, y_encoded)
            self.trained_feature_columns_ = list(X.columns)
            logger.info("XGBoostClassifier training completed.")
        except Exception as e:
            logger.error(f"Error during XGBoostClassifier training: {e}", exc_info=True)
            self.model = None

    def predict(self, new_data: pd.DataFrame, feature_columns: Optional[list] = None) -> Optional[Dict[str, Any]]:
        if self.model is None:
            logger.error("XGBoost model not loaded/trained. Cannot predict.")
            return None

        df_with_features = self.create_features(new_data)
        cols_for_pred = feature_columns or self.trained_feature_columns_
        if not cols_for_pred:
            logger.error("No feature columns determined for XGBoost prediction.")
            return None
            
        cols_for_pred = [col for col in cols_for_pred if col in df_with_features.columns]
        X_new = df_with_features[cols_for_pred].copy()
        if X_new.empty:
            return None

        try:
            last_row_features = X_new.iloc[[-1]]
            probabilities = self.model.predict_proba(last_row_features)[0]
            predicted_class_encoded = np.argmax(probabilities)
            predicted_class_label = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            confidence = probabilities[predicted_class_encoded]
            
            return {
                "predicted_class": int(predicted_class_label),
                "confidence": float(confidence),
                "probabilities": [float(p) for p in probabilities]
            }
        except Exception as e:
            logger.error(f"Error during XGBoostClassifier prediction: {e}", exc_info=True)
            return None

    def save_model(self, path: str):
        if self.model is None: return
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_columns': self.trained_feature_columns_
            }
            joblib.dump(model_data, path)
            logger.info(f"XGBoostClassifier model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving XGBoostClassifier model to {path}: {e}", exc_info=True)

    def load_model(self, path: str):
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.trained_feature_columns_ = model_data.get('feature_columns')
            self.model_path = path
            logger.info(f"XGBoostClassifier model loaded from {path}. Trained features: {self.trained_feature_columns_}")
        except FileNotFoundError:
            logger.error(f"XGBoostClassifier model file not found at {path}.")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading XGBoostClassifier model from {path}: {e}", exc_info=True)
            self.model = None