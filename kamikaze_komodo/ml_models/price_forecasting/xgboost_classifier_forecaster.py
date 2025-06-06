# FILE: kamikaze_komodo/ml_models/price_forecasting/xgboost_classifier_forecaster.py
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder

from kamikaze_komodo.ml_models.price_forecasting.base_forecaster import BasePriceForecaster
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class XGBoostClassifierForecaster(BasePriceForecaster):
    """
    XGBoost-based classifier for price movement prediction (UP, DOWN, SIDEWAYS).
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        # FIX: Initialize attributes BEFORE calling super().__init__
        # This prevents the super constructor's call to load_model from being overwritten.
        self.label_encoder = LabelEncoder()
        self.trained_feature_columns_: Optional[List[str]] = None
        
        super().__init__(model_path, params) # Handles loading model if path provided
        
        self.default_xgb_params = {
            'objective': 'multi:softprob', # For multiclass classification, outputs probabilities
            'eval_metric': 'mlogloss', # Multiclass logloss
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False, # Deprecated, handle encoding manually if needed
            'seed': 42,
        }
        config_xgb_params = {k.replace('xgb_params_', ''): v for k, v in self.params.items() if k.startswith('xgb_params_')}
        self.xgb_params = {**self.default_xgb_params, **config_xgb_params}
        
        self.num_class = int(self.params.get('num_classes', 3)) # UP, DOWN, SIDEWAYS
        self.xgb_params['num_class'] = self.num_class
        
        if not self.model and self.model_path:
            self.load_model(self.model_path)

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty: 
            return pd.DataFrame()
        df = data.copy()
        if 'close' not in df.columns:
            logger.error("'close' column missing for feature creation.")
            return df

        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
            df[f'close_change_lag_{lag}'] = df['close'].pct_change(lag)

        if 'log_return_lag_1' in df.columns:
            df['volatility_5'] = df['log_return_lag_1'].rolling(window=5).std()
            df['volatility_10'] = df['log_return_lag_1'].rolling(window=10).std()
            df['volatility_20'] = df['log_return_lag_1'].rolling(window=20).std()

        if all(col in df.columns for col in ['high', 'low', 'close']):
            try:
                import pandas_ta as ta
                df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
                df.ta.macd(append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
                df.ta.atr(length=14, append=True, col_names=('ATR_14',))
            except ImportError: 
                logger.warning("pandas_ta not installed. Skipping TA features for XGBoost.")
            except Exception as e: 
                logger.warning(f"Error creating TA features: {e}")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _define_target(self, data: pd.DataFrame, thresholds: Optional[Tuple[float, float]] = (-0.001, 0.001)) -> pd.Series:
        """Defines target classes: 0 (UP), 1 (DOWN), 2 (SIDEWAYS)."""
        future_returns = data['close'].pct_change(1).shift(-1) # Next bar's return
        if thresholds is None: 
            thresholds = (-0.001, 0.001) # Default if not provided
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
            if len(thresholds_list) != 2: 
                raise ValueError("ReturnThresholds_Percent must be two comma-separated floats.")
            return_thresholds = tuple(thresholds_list)
        except Exception as e:
            logger.warning(f"Invalid ReturnThresholds_Percent '{return_thresholds_str}', using defaults (-0.001, 0.001). Error: {e}")
            return_thresholds = (-0.001, 0.001)

        if target_definition == 'next_bar_direction':
            df['target'] = self._define_target(df, return_thresholds)
        else:
            logger.error(f"Unsupported target_definition: {target_definition}")
            return

        df.dropna(subset=['target'], inplace=True)
        df_with_features = self.create_features(df)

        if feature_columns:
            actual_features = [col for col in feature_columns if col in df_with_features.columns]
        else:
            default_feature_set = [
                col for col in df_with_features.columns if 
                col.startswith('log_return_lag_') or
                col.startswith('close_change_lag_') or
                col.startswith('volatility_') or
                col in ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATR_14']
            ]
            actual_features = [col for col in default_feature_set if col in df_with_features.columns]


        if not actual_features:
            logger.error("No features selected for training XGBoost.")
            return

        X = df_with_features[actual_features].copy()
        y = df.loc[X.index, 'target'].astype(int)

        logger.debug(f"XGBoost - Shape of df_with_features: {df_with_features.shape}")
        logger.debug(f"XGBoost - Features selected for X: {actual_features}")
        logger.debug(f"XGBoost - Sample of X before dropna (head):\n{X.head()}")
        logger.debug(f"XGBoost - NaN counts in X before dropna:\n{X.isnull().sum().sort_values(ascending=False)}")
        logger.debug(f"XGBoost - Shape of X before dropna: {X.shape}")

        X.dropna(inplace=True) 

        logger.debug(f"XGBoost - Shape of X after dropna: {X.shape}")
        logger.debug(f"XGBoost - Shape of y before aligning with X: {y.shape}")

        y = y.loc[X.index] 

        logger.debug(f"XGBoost - Shape of y after aligning with X: {y.shape}")
        if not X.empty:
            logger.debug(f"XGBoost - Sample of X after dropna & alignment (head):\n{X.head()}")
        if not y.empty:
            logger.debug(f"XGBoost - Sample of y after dropna & alignment (head):\n{y.head()}")
            logger.debug(f"XGBoost - Value counts of y: \n{y.value_counts(dropna=False)}")


        if X.empty or y.empty:
            logger.error("Feature matrix X or target vector y is empty after processing. Training cannot proceed.")
            if X.empty:
                logger.error(f"XGBoost - X is empty. Columns previously in X (before dropna): {str(actual_features)}")
                logger.error(f"XGBoost - df_with_features had columns: {list(df_with_features.columns)}")
            if y.empty:
                logger.error("XGBoost - y is empty.")
            return

        self.label_encoder.fit(y) # Fit encoder on the integer labels (0, 1, 2)
        y_encoded = self.label_encoder.transform(y)

        self.model = xgb.XGBClassifier(**self.xgb_params)
        logger.info(f"Training XGBoostClassifier with {len(X)} samples. Features: {actual_features}")
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
        cols_for_pred = feature_columns if feature_columns else self.trained_feature_columns_
        if not cols_for_pred:
            logger.error("No feature columns determined for XGBoost prediction.")
            return None

        # Ensure all required columns are present, even if with NaNs, before selection
        missing_cols = set(cols_for_pred) - set(df_with_features.columns)
        if missing_cols:
            logger.warning(f"Columns required for prediction are missing from generated features: {missing_cols}")
            return None

        X_new = df_with_features[cols_for_pred].copy()
        if X_new.empty:
            logger.warning("Feature matrix for prediction is empty.")
            return None

        if X_new.iloc[-1].isnull().any():
            logger.warning(f"Last row for XGBoost prediction contains NaNs in features: {X_new.columns[X_new.iloc[-1].isnull()].tolist()}. Prediction might be unreliable.")

        try:
            # Predict probabilities for the last row
            last_row_features = X_new.iloc[[-1]] # Keep as DataFrame
            probabilities = self.model.predict_proba(last_row_features)[0]
            predicted_class_encoded = np.argmax(probabilities)
            predicted_class_label = self.label_encoder.inverse_transform([predicted_class_encoded])[0] # Original label (0, 1, 2)
            confidence = probabilities[predicted_class_encoded]
        
            return {
                "predicted_class": int(predicted_class_label), # 0:UP, 1:DOWN, 2:SIDEWAYS
                "confidence": float(confidence),
                "probabilities": [float(p) for p in probabilities] # Prob for each class
            }
        except Exception as e:
            logger.error(f"Error during XGBoostClassifier prediction: {e}", exc_info=True)
            return None

    def save_model(self, path: str):
        if self.model is None:
            logger.error("No XGBoost model to save.")
            return
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