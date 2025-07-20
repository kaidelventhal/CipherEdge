# FILE: kamikaze_komodo/ml_models/training_pipelines/xgboost_classifier_pipeline.py
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from typing import Optional, List
from kamikaze_komodo.ml_models.price_forecasting.xgboost_classifier_forecaster import XGBoostClassifierForecaster
from kamikaze_komodo.data_handling.data_handler import DataHandler
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class XGBoostClassifierTrainingPipeline:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "XGBoost_Classifier_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_config_section = model_config_section
        
        self.model_params = settings.get_strategy_params(model_config_section)
        
        _model_base_path = self.model_params.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = self.model_params.get('modelfilename', f"xgb_classifier_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
        
        if not os.path.isabs(_model_base_path):
            self.model_save_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_save_path_dir = _model_base_path
            
        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir, exist_ok=True)
            
        self.model_full_save_path = os.path.join(self.model_save_path_dir, _model_filename)
        
        self.forecaster = XGBoostClassifierForecaster(params=self.model_params)
        logger.info(f"XGBoost Training Pipeline initialized. Model will be saved to: {self.model_full_save_path}")

    async def fetch_training_data(self, days_history: int = 730) -> pd.DataFrame:
        data_handler = DataHandler()
        start_date = datetime.now(timezone.utc) - timedelta(days=days_history)
        end_date = datetime.now(timezone.utc)
        data_df = await data_handler.get_prepared_data(
            self.symbol, self.timeframe, start_date, end_date,
            needs_funding_rate=True, needs_sentiment=True
        )
        await data_handler.close()
        if not data_df.empty:
            logger.info(f"Fetched and prepared {len(data_df)} bars for XGBoost training.")
        return data_df

    async def run_training(self, tune_hyperparameters: bool = False):
        days_history = int(self.model_params.get('trainingdayshistory', 730))
        historical_df = await self.fetch_training_data(days_history=days_history)
        if historical_df.empty:
            logger.error("Cannot run XGBoost training, no historical data.")
            return

        target_def = self.model_params.get('targetdefinition', 'next_bar_direction')
        feature_cols_str = self.model_params.get('feature_columns')
        feature_columns = [col.strip() for col in feature_cols_str.split(',')] if feature_cols_str else None
        
        if tune_hyperparameters:
            logger.info("Starting hyperparameter tuning for XGBoost...")
            self._tune_and_train(historical_df, target_def, feature_columns)
        else:
            logger.info(f"Starting XGBoost training with target: '{target_def}', features: {feature_columns or 'default'}")
            self.forecaster.train(historical_df, target_definition=target_def, feature_columns=feature_columns)
        
        if self.forecaster.model:
            self.forecaster.save_model(self.model_full_save_path)
        else:
            logger.error("XGBoost training did not produce a model. Model not saved.")

    def _tune_and_train(self, data: pd.DataFrame, target_definition: str, feature_columns: Optional[List[str]]):
        # Prepare data
        df = data.copy()
        df['target'] = self.forecaster._define_target(df) # Use internal method to create target
        df.dropna(subset=['target'], inplace=True)
        df_with_features = self.forecaster.create_features(df)
        
        features = feature_columns or [col for col in df_with_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'target']]
        
        X = df_with_features[features].copy()
        X.dropna(inplace=True)
        y = df_with_features.loc[X.index, 'target'].astype(int)

        if X.empty:
            logger.error("No data left for hyperparameter tuning after processing.")
            return
            
        y_encoded = self.forecaster.label_encoder.fit_transform(y)

        # Hyperparameter grid for RandomizedSearch
        param_dist = {
            'n_estimators': sp_randint(100, 1000),
            'learning_rate': sp_uniform(0.01, 0.2),
            'max_depth': sp_randint(3, 10),
            'subsample': sp_uniform(0.7, 0.3),
            'colsample_bytree': sp_uniform(0.7, 0.3),
            'gamma': sp_uniform(0, 0.5)
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_clf = xgb.XGBClassifier(**self.forecaster.xgb_params)
        
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_dist,
            n_iter=25,
            cv=tscv,
            scoring='accuracy', # Or 'f1_weighted'
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X, y_encoded)
        
        logger.info(f"Best parameters found: {random_search.best_params_}")
        
        # Update forecaster with best model and retrain on full data
        self.forecaster.model = random_search.best_estimator_
        self.forecaster.xgb_params = random_search.best_params_
        self.forecaster.trained_feature_columns_ = list(X.columns)
        logger.info("Retraining final model on all available data with best parameters...")
        self.forecaster.model.fit(X, y_encoded)