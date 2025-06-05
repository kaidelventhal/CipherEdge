# kamikaze_komodo/ml_models/price_forecasting/lightgbm_forecaster.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib # For saving/loading model
from typing import Optional, Dict, Any, List, Union
from kamikaze_komodo.ml_models.price_forecasting.base_forecaster import BasePriceForecaster
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings as app_settings # Use app_settings to avoid conflict
logger = get_logger(__name__)
class LightGBMForecaster(BasePriceForecaster):
    """
    LightGBM-based price forecaster.
    Predicts price movement (e.g., next bar's close relative to current).
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_path, params) # This calls load_model if model_path is provided
        self.default_lgbm_params = {
            'objective': 'regression_l1', # Or 'regression_l2'
            'metric': 'rmse', # Root Mean Squared Error
            'n_estimators': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }
        # Update with any params passed from config for 'lgbm_params' specifically
        config_lgbm_params = {}
        if self.params: # self.params comes from config section like [LightGBM_Forecaster]
            for key, value in self.params.items():
                if key.startswith('lgbm_params_'): # e.g. lgbm_params_n_estimators
                    param_name = key.replace('lgbm_params_', '').lower()
                    config_lgbm_params[param_name] = value
        
        self.lgbm_params = {**self.default_lgbm_params, **config_lgbm_params}
        # If model_path was passed to super() and model loaded, self.model is set.
        # If model_path is in params (e.g. from config) but not passed directly to init, load it.
        if not self.model and self.model_path: # self.model_path is set by super if path given
             self.load_model(self.model_path)
        elif not self.model and self.params.get('modelfilename'): # Check if model path is in params from config
            # Construct full path if model_path is not set by direct argument to __init__
            # This logic is typically handled by the training/inference pipeline that instantiates this.
            # For direct use, ensure model_path is passed or params correctly configure it.
            pass
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a standard set of known features from the input data.
        This method generates a superset of features; selection for training/prediction happens elsewhere.
        """
        if data.empty:
            logger.warning("Data for feature creation is empty.")
            return pd.DataFrame()
        df = data.copy() 
        if 'close' not in df.columns:
            logger.error("'close' column not found in data for feature creation.")
            return df 
        for lag in [1, 3, 5, 10]:
            df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
            df[f'close_change_lag_{lag}'] = df['close'].pct_change(lag)
        
        if 'log_return_lag_1' in df.columns: # Check if base lag feature was created
            df['volatility_5'] = df['log_return_lag_1'].rolling(window=5).std()
            df['volatility_10'] = df['log_return_lag_1'].rolling(window=10).std()
        else:
            df['volatility_5'] = np.nan
            df['volatility_10'] = np.nan
        if all(col in df.columns for col in ['high', 'low', 'close']):
            try:
                import pandas_ta as ta
                df.ta.rsi(close=df['close'], length=14, append=True, col_names=('RSI_14',))
                df.ta.macd(close=df['close'], append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
            except ImportError:
                logger.warning("pandas_ta not installed. Skipping TA features for LightGBM.")
            except Exception as e_ta: 
                logger.warning(f"Error during pandas_ta feature creation: {e_ta}. TA features might be missing or incomplete.")
        else:
            logger.warning("Missing 'high', 'low', or 'close' columns. Skipping TA features.")
        df = df.replace([np.inf, -np.inf], np.nan)
        return df # Return DataFrame with ALL generated features, NaNs from shifts/calcs are expected here
    def train(self, historical_data: pd.DataFrame, target_column: str = 'close_change_lag_1_future', feature_columns_to_use: Optional[List[str]] = None):
        logger.info(f"Starting LightGBM training for target '{target_column}'. Data shape: {historical_data.shape}")
        df = historical_data.copy()
        if target_column == 'close_change_lag_1_future':
            df['target'] = (df['close'].shift(-1) / df['close']) - 1
        elif target_column.startswith('log_return_lag_') and target_column.endswith('_future'):
            try:
                shift_val = int(target_column.split('_')[3])
                df['target'] = np.log(df['close'].shift(-shift_val) / df['close'])
            except Exception:
                logger.error(f"Could not parse shift value from target_column: {target_column}. Using default log return shift -1.")
                df['target'] = np.log(df['close'].shift(-1) / df['close']) # Default to next bar log return
        else:
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not in data or not a recognized dynamic target format.")
                return
            df['target'] = df[target_column]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['target'], inplace=True) 
        df_with_all_features = self.create_features(df) 
        
        X_final_features = pd.DataFrame()
        if feature_columns_to_use:
            actual_features_present = [col for col in feature_columns_to_use if col in df_with_all_features.columns]
            missing = set(feature_columns_to_use) - set(actual_features_present)
            if missing:
                logger.warning(f"During training, specified feature_columns not all found/generated: {missing}. Using available: {actual_features_present}")
            if not actual_features_present:
                 logger.error("None of the specified_feature_columns_to_use are present after generation. Cannot train.")
                 return
            X_final_features = df_with_all_features[actual_features_present].copy()
        else:
            # Default: use a predefined list of potentially generated features
            default_feature_set = [
                col for col in df_with_all_features.columns if col.startswith('log_return_lag_') or \
                col.startswith('close_change_lag_') or \
                col.startswith('volatility_') or \
                col in ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
            ]
            # Filter this default set by what's actually in df_with_all_features
            default_features_present = [col for col in default_feature_set if col in df_with_all_features.columns]
            if not default_features_present:
                logger.error("No default features could be found/generated. Cannot train.")
                return
            X_final_features = df_with_all_features[default_features_present].copy()
        
        X_final_features.dropna(inplace=True) 
        y_train = df.loc[X_final_features.index, 'target'] 
        X_train = X_final_features
        if X_train.empty or y_train.empty:
            logger.error("Feature matrix X_train or target vector y_train is empty after processing. Training cannot proceed.")
            return
        self.model = lgb.LGBMRegressor(**self.lgbm_params)
        logger.info(f"Training LightGBM model with {len(X_train)} samples. Features: {list(X_train.columns)}")
        try:
            self.model.fit(X_train, y_train)
            logger.info("LightGBM model training completed.")
            self.trained_feature_columns_ = list(X_train.columns) # Store actual columns used
        except Exception as e:
            logger.error(f"Error during LightGBM model training: {e}", exc_info=True)
            self.model = None
    def predict(self, new_data: pd.DataFrame, feature_columns_to_use: Optional[list] = None) -> Union[pd.Series, float, None]:
        if self.model is None:
            logger.error("Model not loaded or trained. Cannot make predictions.")
            return None
        
        df_with_all_features = self.create_features(new_data)
        cols_for_prediction = None
        # Priority: 1. Explicitly passed `feature_columns_to_use`, 2. `self.trained_feature_columns_`
        if feature_columns_to_use:
            cols_for_prediction = [col for col in feature_columns_to_use if col in df_with_all_features.columns]
            missing_explicit = set(feature_columns_to_use) - set(cols_for_prediction)
            if missing_explicit: logger.warning(f"Explicitly requested prediction features not found: {missing_explicit}")
        elif hasattr(self, 'trained_feature_columns_') and self.trained_feature_columns_:
            cols_for_prediction = [col for col in self.trained_feature_columns_ if col in df_with_all_features.columns]
            missing_trained = set(self.trained_feature_columns_) - set(cols_for_prediction)
            if missing_trained: logger.warning(f"Features model was trained on are not all available for prediction: {missing_trained}")
        
        if not cols_for_prediction:
            logger.error("No feature columns determined for prediction. Make sure model is trained or features are specified and generatable.")
            return None
        
        X_new = df_with_all_features[cols_for_prediction].copy()
        if X_new.empty:
            logger.warning("Feature matrix X_new is empty after selection. Cannot predict.")
            return None
        
        # Handle potential NaNs in the very last row for prediction
        # LightGBM can handle NaNs internally if not too many, but for the last row it's critical.
        # If X_new is just one row (latest data point), ensure it's complete or handle.
        if len(X_new) == 1 and X_new.isnull().values.any():
            logger.warning(f"Latest data row for prediction contains NaNs in selected features: {X_new[X_new.isnull().any(axis=1)].columns[X_new.isnull().any(axis=1)[0]]}. Prediction may fail or be zero.")
            # Depending on LightGBM's setup and data, it might predict 0 or error.
            # Option: return None or fill specific ways if this is an issue.
            # For now, let LightGBM try.
        try:
            predictions = self.model.predict(X_new)
            logger.debug(f"Made {len(predictions)} predictions. Latest prediction input shape: {X_new.shape}. Prediction output: {predictions[-1] if len(predictions)>0 else 'N/A'}")
            if len(predictions) == 0: return None
            # We usually want the prediction for the last row of input `new_data`
            return predictions[-1] if isinstance(predictions, np.ndarray) else predictions # if it was a single value already
        except Exception as e:
            logger.error(f"Error during LightGBM prediction: {e}", exc_info=True)
            return None
    def save_model(self, path: Optional[str] = None):
        _path = path or self.model_path
        if self.model is None:
            logger.error("No model to save.")
            return
        if not _path:
            logger.error("No path specified for saving the model.")
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
            # This case might happen if model_path is None and params don't specify it.
            # Initialization of LightGBMForecaster should handle this (e.g. by not setting self.model)
            logger.debug("No path specified for loading the model during load_model call.")
            return
        try:
            model_and_features = joblib.load(_path)
            self.model = model_and_features['model']
            self.trained_feature_columns_ = model_and_features.get('feature_columns') 
            self.model_path = _path 
            logger.info(f"LightGBM model and feature columns loaded from {_path}. Trained features: {self.trained_feature_columns_}")
        except FileNotFoundError:
            logger.error(f"LightGBM model file not found at {_path}.")
            self.model = None
            self.trained_feature_columns_ = None
        except Exception as e:
            logger.error(f"Error loading LightGBM model from {_path}: {e}", exc_info=True)
            self.model = None
            self.trained_feature_columns_ = None