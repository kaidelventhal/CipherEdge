# kamikaze_komodo/strategy_framework/strategies/ml_forecaster_strategy.py
import pandas as pd
import pandas_ta as ta # For ATR calculation
from typing import Dict, Any, Optional
import os

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings as app_settings # Use app_settings
from kamikaze_komodo.ml_models.inference_pipelines.lightgbm_inference import LightGBMInference # Example

logger = get_logger(__name__)

class MLForecasterStrategy(BaseStrategy):
    """
    A strategy that uses an ML price forecaster to generate trading signals.
    This example uses the LightGBMInference pipeline.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        # Get model configuration (e.g., path, type) from strategy params or global settings
        self.model_config_section = self.params.get('model_config_section', 'LightGBM_Forecaster') # Config section for model
        self.forecaster_type = self.params.get('forecaster_type', 'lightgbm') # e.g., lightgbm, lstm
        
        # Thresholds for converting prediction to signal
        self.long_threshold = float(self.params.get('long_threshold', 0.0005)) # e.g., predicted_return > 0.05%
        self.short_threshold = float(self.params.get('short_threshold', -0.0005)) # e.g., predicted_return < -0.05%
        self.exit_long_threshold = float(self.params.get('exit_long_threshold', 0.0)) # Prediction < this to close long
        self.min_prediction_confidence = float(self.params.get('min_prediction_confidence', 0.0)) # Optional

        self.inference_engine = None
        if self.forecaster_type.lower() == 'lightgbm':
            try:
                self.inference_engine = LightGBMInference(symbol, timeframe, model_config_section=self.model_config_section)
                if self.inference_engine.forecaster.model is None:
                    logger.error(f"MLForecasterStrategy: LightGBM model for {symbol} ({timeframe}) could not be loaded. Strategy will not generate signals.")
                    self.inference_engine = None # Disable if model not loaded
            except Exception as e:
                logger.error(f"Failed to initialize LightGBMInference for {symbol} ({timeframe}): {e}", exc_info=True)
                self.inference_engine = None
        else:
            logger.error(f"Unsupported forecaster_type: {self.forecaster_type} for MLForecasterStrategy.")

        logger.info(
            f"Initialized MLForecasterStrategy for {symbol} ({timeframe}) "
            f"using {self.forecaster_type}. Long Threshold: {self.long_threshold}, Short Threshold: {self.short_threshold}"
        )
        self.current_position_status: Optional[SignalType] = None


    def _get_prediction(self, current_history_df: pd.DataFrame) -> Optional[float]:
        """Internal method to get prediction from the inference engine."""
        if self.inference_engine:
            # The inference engine's get_prediction expects a DataFrame of historical data
            # from which features for the *latest point* will be derived.
            return self.inference_engine.get_prediction(current_history_df)
        return None

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        For MLForecasterStrategy, this might involve adding predictions as a column if doing batch processing.
        Or simply calculating ATR if needed.
        """
        df = data_df.copy()
        
        # ATR for BarData object (optional, but good practice)
        if all(col in df.columns for col in ['high', 'low', 'close']):
             atr_period = int(self.params.get('atr_period', 14)) # Default ATR period
             if len(df) >= atr_period:
                 # Ensure correct inputs for pandas_ta.atr
                 df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=atr_period)
             else:
                 df['atr'] = pd.NA 
        else:
            df['atr'] = pd.NA

        # For batch signal generation, we could try to get predictions for all rows.
        # This is more complex as `_get_prediction` typically works on the latest data.
        # For simplicity in on_bar_data, prediction is done bar-by-bar.
        # For generate_signals (vectorized), this would need careful implementation.
        # Here, we'll just calculate ATR and let on_bar_data handle predictions.
            
        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        """
        Generates signals in batch. For ML, this would involve generating predictions for the whole series.
        This is a simplified version for now; robust batch prediction needs careful handling of feature generation.
        """
        if data.empty or self.inference_engine is None:
            logger.warning("Data is empty or no inference engine for MLForecasterStrategy. No signals generated.")
            return pd.Series(index=data.index, dtype='object')

        df_with_indicators = self._calculate_indicators(data) # Adds ATR
        
        predictions = []
        min_hist_for_pred = int(self.params.get('min_bars_for_prediction', 50)) # Min bars for features
        
        for i in range(len(df_with_indicators)):
            if i < min_hist_for_pred -1 :
                predictions.append(None) # Not enough history
                continue
            
            current_slice = df_with_indicators.iloc[:i+1] # History up to current bar
            pred = self._get_prediction(current_slice)
            predictions.append(pred)
            
        df_with_indicators['prediction'] = predictions
        df_with_indicators['prediction'].fillna(0.0, inplace=True) # Default no prediction to 0

        signals = pd.Series(index=df_with_indicators.index, dtype='object').fillna(SignalType.HOLD)
        current_pos_state = None # Simulate state for batch processing

        for i in range(len(df_with_indicators)):
            prediction = df_with_indicators['prediction'].iloc[i]
            current_sentiment = sentiment_series.iloc[i] if sentiment_series is not None and i < len(sentiment_series) else 0.0

            if current_pos_state != SignalType.LONG:
                if prediction > self.long_threshold:
                    sentiment_ok = True
                    if sentiment_series is not None and hasattr(app_settings, 'sentiment_filter_threshold_long'):
                         s_thresh_long = app_settings.sentiment_filter_threshold_long
                         if current_sentiment < s_thresh_long:
                             sentiment_ok = False
                    if sentiment_ok:
                        signals.iloc[i] = SignalType.LONG
                        current_pos_state = SignalType.LONG
            
            elif current_pos_state == SignalType.LONG:
                if prediction < self.exit_long_threshold: 
                    signals.iloc[i] = SignalType.CLOSE_LONG
                    current_pos_state = None
        
        logger.info(f"Generated MLForecaster signals (vectorized, simplified). Longs: {signals.eq(SignalType.LONG).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}")
        return signals


    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None) -> Optional[SignalType]:
        self.update_data_history(bar_data) # Appends current bar_data to self.data_history

        # Calculate basic indicators like ATR and update bar_data with it
        # This ensures bar_data.atr is available for position sizers/stop managers if this strategy calculates it.
        if len(self.data_history) > 0:
            # Ensure enough data for ATR calculation based on its period
            atr_period = int(self.params.get('atr_period', 14))
            required_bars_for_atr = max(20, atr_period) # pandas_ta might need more than just atr_period bars

            if len(self.data_history) >= required_bars_for_atr:
                # Calculate ATR on a sufficient slice of recent history
                history_slice_for_atr = self.data_history.tail(required_bars_for_atr)
                temp_df_for_atr = self._calculate_indicators(history_slice_for_atr) 
                if not temp_df_for_atr.empty and 'atr' in temp_df_for_atr.columns and pd.notna(temp_df_for_atr['atr'].iloc[-1]):
                    bar_data.atr = temp_df_for_atr['atr'].iloc[-1]
            elif bar_data.atr is None : # Ensure ATR is None if not calculable
                 bar_data.atr = None


        if self.inference_engine is None:
            return SignalType.HOLD # No model loaded

        # Ensure enough history for feature creation by the forecaster
        min_history_len = int(self.params.get('min_bars_for_prediction', 50)) # Min bars needed for reliable features
        if len(self.data_history) < min_history_len:
            logger.debug(f"Not enough history ({len(self.data_history)}/{min_history_len}) for ML prediction on {self.symbol}.")
            return SignalType.HOLD

        # Get prediction using the historical data up to and including the current bar
        prediction = self._get_prediction(self.data_history) # Pass the updated history

        if prediction is None:
            logger.debug(f"No prediction received for {self.symbol} at {bar_data.timestamp}.")
            bar_data.prediction_value = None
            bar_data.prediction_confidence = None 
            return SignalType.HOLD
        
        bar_data.prediction_value = prediction 
        # bar_data.prediction_confidence = ... # If model provides confidence

        signal_to_return = SignalType.HOLD
        effective_sentiment = sentiment_score if sentiment_score is not None else (bar_data.sentiment_score if bar_data.sentiment_score is not None else 0.0)

        logger.debug(f"{bar_data.timestamp} - {self.symbol}: ML Prediction = {prediction:.6f}, Sentiment = {effective_sentiment:.2f}")

        if self.current_position_status != SignalType.LONG:
            if prediction > self.long_threshold:
                sentiment_ok_for_long = True
                if sentiment_score is not None and hasattr(app_settings, 'sentiment_filter_threshold_long') and app_settings.sentiment_filter_threshold_long is not None:
                    s_thresh_long = app_settings.sentiment_filter_threshold_long
                    if effective_sentiment < s_thresh_long:
                        logger.info(f"{bar_data.timestamp} - MLForecaster LONG for {self.symbol} SUPPRESSED by sentiment ({effective_sentiment:.2f} < {s_thresh_long}). Prediction was {prediction:.4f}.")
                        sentiment_ok_for_long = False
                
                if sentiment_ok_for_long:
                    signal_to_return = SignalType.LONG
                    self.current_position_status = SignalType.LONG
                    logger.info(f"{bar_data.timestamp} - MLForecaster LONG for {self.symbol}. Prediction: {prediction:.4f}. ATR: {bar_data.atr if bar_data.atr is not None else 'N/A'}")
        
        elif self.current_position_status == SignalType.LONG:
            if prediction < self.exit_long_threshold: 
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - MLForecaster CLOSE_LONG for {self.symbol}. Prediction: {prediction:.4f}")
        
        return signal_to_return