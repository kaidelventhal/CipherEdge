import pandas as pd
import numpy as np
import os
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from collections import Counter

from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger
from cipher_edge.ml_models.inference_pipelines.lightgbm_inference import LightGBMInference
from cipher_edge.ml_models.inference_pipelines.xgboost_classifier_inference import XGBoostClassifierInference
from cipher_edge.ml_models.inference_pipelines.lstm_inference import LSTMInference
from cipher_edge.config.settings import settings

logger = get_logger(__name__)

class EnsembleMLStrategy(BaseStrategy):
    """
    A stateful ensemble strategy that combines signals from multiple ML models.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.ensemble_method = self.params.get('ensemble_method', 'majority_vote').lower()
        
        lgbm_section = self.params.get('lgbm_config_section', 'LightGBM_Forecaster')
        xgb_section = self.params.get('xgb_config_section', 'XGBoost_Classifier_Forecaster')
        lstm_section = self.params.get('lstm_config_section', 'LSTM_Forecaster')

        self.models = {}
        try:
            lgbm_engine = LightGBMInference(symbol, timeframe, lgbm_section)
            if lgbm_engine.forecaster.model is not None:
                self.models["LGBM"] = lgbm_engine
        except Exception as e:
            logger.error(f"Failed to load LGBM model for ensemble: {e}")
        try:
            xgb_engine = XGBoostClassifierInference(symbol, timeframe, xgb_section)
            if xgb_engine.forecaster.model is not None:
                self.models["XGB"] = xgb_engine
        except Exception as e:
            logger.error(f"Failed to load XGB model for ensemble: {e}")
        try:
            lstm_engine = LSTMInference(symbol, timeframe, lstm_section)
            if lstm_engine.forecaster.model is not None:
                self.models["LSTM"] = lstm_engine
        except Exception as e:
            logger.error(f"Failed to load LSTM model for ensemble: {e}")

        self.regressor_thresholds = settings.get_strategy_params('MLForecaster_Strategy') if settings else {}
        
        self.model_weights = {
            "LGBM": float(self.params.get('model_weights_lgbm', 0.4)),
            "XGB": float(self.params.get('model_weights_xgb', 0.4)),
            "LSTM": float(self.params.get('model_weights_lstm', 0.2))
        }

        logger.info(f"Initialized EnsembleMLStrategy with method: {self.ensemble_method}. Models loaded: {list(self.models.keys())}")

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ATR for compatibility with risk management modules.
        Other features are created on-the-fly by the inference models.
        """
        df = data.copy()
        # FIX: Add ATR calculation to ensure compatibility with ATR-based risk modules
        atr_period = int(self.params.get('atr_period', 14))
        if 'high' in df.columns and len(df) >= atr_period:
            df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=atr_period)
        
        logger.info(f"Ensemble strategy '{self.name}' prepared base data (ATR).")
        return df

    def _get_model_predictions(self) -> Dict[str, Any]:
        predictions = {}
        for name, model_engine in self.models.items():
            predictions[name] = model_engine.get_prediction(self.data_history)
        return predictions

    def _get_ensemble_signal(self, model_predictions: Dict[str, Any]) -> SignalType:
        if self.ensemble_method == 'majority_vote':
            return self._get_majority_vote_signal(model_predictions)
        elif self.ensemble_method == 'weighted_average':
            return self._get_weighted_average_signal(model_predictions)
        return SignalType.HOLD

    def _convert_prediction_to_vote(self, model_name: str, prediction: Any) -> int:
        if prediction is None: return 0
        if model_name == "XGB":
            pred_class = prediction.get('predicted_class')
            if pred_class == 0: return 1
            if pred_class == 1: return -1
            return 0
        else: # Regressors
            long_thresh = self.regressor_thresholds.get('longthreshold', 0.0005)
            short_thresh = self.regressor_thresholds.get('shortthreshold', -0.0005)
            if prediction > long_thresh: return 1
            if prediction < short_thresh: return -1
            return 0

    def _get_majority_vote_signal(self, model_predictions: Dict[str, Any]) -> SignalType:
        votes = [self._convert_prediction_to_vote(name, pred) for name, pred in model_predictions.items() if pred is not None]
        if not votes: return SignalType.HOLD
        
        vote_counts = Counter(votes)
        if vote_counts.get(1, 0) > vote_counts.get(-1, 0): return SignalType.LONG
        if vote_counts.get(-1, 0) > vote_counts.get(1, 0): return SignalType.SHORT
        return SignalType.HOLD

    def _get_weighted_average_signal(self, model_predictions: Dict[str, Any]) -> SignalType:
        weighted_sum, total_weight = 0.0, 0.0
        for name, pred in model_predictions.items():
            if pred is None or name not in self.model_weights: continue
            
            numeric_pred = 0.0
            if name == "XGB":
                probs = pred.get('probabilities')
                if probs and len(probs) == 3: numeric_pred = probs[0] * 1 + probs[1] * -1
            else:
                numeric_pred = float(pred)

            weighted_sum += self.model_weights[name] * numeric_pred
            total_weight += self.model_weights[name]
        
        if total_weight == 0: return SignalType.HOLD
            
        final_score = weighted_sum / total_weight
        
        long_thresh = self.regressor_thresholds.get('longthreshold', 0.0005)
        short_thresh = self.regressor_thresholds.get('shortthreshold', -0.0005)

        if final_score > long_thresh: return SignalType.LONG
        if final_score < short_thresh: return SignalType.SHORT
        return SignalType.HOLD

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(current_bar)
        
        if len(self.data_history) < 65: return SignalType.HOLD

        model_predictions = self._get_model_predictions()
        if not model_predictions: return SignalType.HOLD

        logger.debug(f"{current_bar.timestamp} - Raw Model Predictions: {model_predictions}")
        
        entry_signal = self._get_ensemble_signal(model_predictions)
        
        if self.current_position_status is None:
            if entry_signal == SignalType.LONG:
                self.current_position_status = SignalType.LONG
                return SignalType.LONG
            elif entry_signal == SignalType.SHORT and self.enable_shorting:
                self.current_position_status = SignalType.SHORT
                return SignalType.SHORT
        elif self.current_position_status == SignalType.LONG and entry_signal == SignalType.SHORT:
            self.current_position_status = None
            return SignalType.CLOSE_LONG
        elif self.current_position_status == SignalType.SHORT and entry_signal == SignalType.LONG:
            self.current_position_status = None
            return SignalType.CLOSE_SHORT
                    
        return SignalType.HOLD