# FILE: kamikaze_komodo/strategy_framework/strategies/ml_forecaster_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
import os
import numpy as np
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings as app_settings
from kamikaze_komodo.ml_models.inference_pipelines.lightgbm_inference import LightGBMInference
from kamikaze_komodo.ml_models.inference_pipelines.xgboost_classifier_inference import XGBoostClassifierInference
from kamikaze_komodo.ml_models.inference_pipelines.lstm_inference import LSTMInference

logger = get_logger(__name__)

class MLForecasterStrategy(BaseStrategy):
    """
    A stateful strategy that uses an ML price forecaster to generate trading signals.
    **IMPROVEMENT**: Can now accept a pre-initialized inference_engine to leverage caching.
    """
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        params: Optional[Dict[str, Any]] = None,
        inference_engine: Optional[Any] = None  # Accept a pre-loaded engine
    ):
        super().__init__(symbol, timeframe, params)
        
        self.model_config_section = self.params.get('modelconfigsection', 'LightGBM_Forecaster')
        self.forecaster_type = self.params.get('forecastertype', 'lightgbm')
        
        self.long_threshold = float(self.params.get('longthreshold', 0.0005))
        self.short_threshold = float(self.params.get('shortthreshold', -0.0005))
        self.exit_long_threshold = float(self.params.get('exitlongthreshold', 0.0))
        self.exit_short_threshold = float(self.params.get('exitshortthreshold', 0.0))
        self.min_prediction_confidence = float(self.params.get('minpredictionconfidence', 0.0))
        
        # Use cached engine if provided, otherwise initialize a new one
        if inference_engine:
            self.inference_engine = inference_engine
        else:
            self.inference_engine = self._initialize_inference_engine()
        
        logger.info(
            f"Initialized MLForecasterStrategy for {symbol} ({timeframe}) "
            f"using {self.forecaster_type}. Long Thresh: {self.long_threshold}, Short Thresh: {self.short_threshold}."
        )

    def _initialize_inference_engine(self):
        """Initializes the correct inference engine based on config."""
        try:
            if self.forecaster_type.lower() == 'lightgbm':
                return LightGBMInference(self.symbol, self.timeframe, model_config_section=self.model_config_section)
            elif self.forecaster_type.lower() == 'xgboost_classifier':
                return XGBoostClassifierInference(self.symbol, self.timeframe, model_config_section=self.model_config_section)
            elif self.forecaster_type.lower() == 'lstm':
                return LSTMInference(self.symbol, self.timeframe, model_config_section=self.model_config_section)
            else:
                logger.error(f"Unsupported forecaster_type: {self.forecaster_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize Inference Engine for {self.forecaster_type}: {e}", exc_info=True)
            return None

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ATR for compatibility with risk management modules.
        """
        df = data.copy()
        atr_period = int(self.params.get('atr_period', 14))
        if 'high' in df.columns and len(df) >= atr_period:
            df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=atr_period)
        return df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(current_bar)
        
        if self.inference_engine is None or self.inference_engine.forecaster.model is None:
            return SignalType.HOLD

        min_history_len = int(self.params.get('min_bars_for_prediction', 60))
        if len(self.data_history) < min_history_len:
            return SignalType.HOLD

        prediction_output = self.inference_engine.get_prediction(self.data_history)
        
        if prediction_output is None:
            return SignalType.HOLD
            
        long_signal, short_signal, close_long, close_short = False, False, False, False

        if self.forecaster_type.lower() in ['lightgbm', 'lstm']:
            pred_val = float(prediction_output)
            if pred_val > self.long_threshold: long_signal = True
            if pred_val < self.short_threshold: short_signal = True
            if pred_val < self.exit_long_threshold: close_long = True
            if pred_val > self.exit_short_threshold: close_short = True
        elif self.forecaster_type.lower() == 'xgboost_classifier':
            pred_class = prediction_output.get('predicted_class')
            confidence = prediction_output.get('confidence', 1.0)
            if confidence < self.min_prediction_confidence:
                return SignalType.HOLD
            if pred_class == 0: long_signal = True
            if pred_class == 1: short_signal = True
            if self.current_position_status == SignalType.LONG and pred_class in [1, 2]: close_long = True
            if self.current_position_status == SignalType.SHORT and pred_class in [0, 2]: close_short = True
        
        # State machine for signals
        if self.current_position_status == SignalType.LONG and close_long:
            self.current_position_status = None
            return SignalType.CLOSE_LONG
        if self.current_position_status == SignalType.SHORT and close_short:
            self.current_position_status = None
            return SignalType.CLOSE_SHORT
        if self.current_position_status is None:
            if long_signal:
                self.current_position_status = SignalType.LONG
                return SignalType.LONG
            if short_signal and self.enable_shorting:
                self.current_position_status = SignalType.SHORT
                return SignalType.SHORT

        return SignalType.HOLD