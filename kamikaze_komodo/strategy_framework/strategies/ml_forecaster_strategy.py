# kamikaze_komodo/strategy_framework/strategies/ml_forecaster_strategy.py
import pandas as pd
import pandas_ta as ta # For ATR calculation
from typing import Dict, Any, Optional, Union, List
import os
import numpy as np
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings as app_settings # Use app_settings
from kamikaze_komodo.ml_models.inference_pipelines.lightgbm_inference import LightGBMInference # Example
# from kamikaze_komodo.ml_models.inference_pipelines.xgboost_classifier_inference import XGBoostClassifierInference # Phase 6

logger = get_logger(__name__)

class MLForecasterStrategy(BaseStrategy):
    """
    A strategy that uses an ML price forecaster to generate trading signals.
    This example uses the LightGBMInference pipeline.
    Phase 6: Added shorting capability and option for XGBoostClassifier.
    Phase 6: Consumes market_regime from BarData.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        # Get model configuration (e.g., path, type) from strategy params or global settings
        self.model_config_section = self.params.get('modelconfigsection', 'LightGBM_Forecaster') # Config section for model
        self.forecaster_type = self.params.get('forecastertype', 'lightgbm') # e.g., lightgbm, lstm, xgboost_classifier
        
        # Thresholds for converting prediction to signal
        self.long_threshold = float(self.params.get('longthreshold', 0.0005)) # e.g., predicted_return > 0.05%
        self.short_threshold = float(self.params.get('shortthreshold', -0.0005)) # e.g., predicted_return < -0.05%
        self.exit_long_threshold = float(self.params.get('exitlongthreshold', 0.0)) # Prediction < this to close long
        self.exit_short_threshold = float(self.params.get('exitshortthreshold', 0.0)) # Prediction > this to close short

        self.min_prediction_confidence = float(self.params.get('minpredictionconfidence', 0.0)) # Optional
        self.inference_engine = None

        # Initialize inference engine based on forecaster_type
        if self.forecaster_type.lower() == 'lightgbm':
            try:
                self.inference_engine = LightGBMInference(symbol, timeframe, model_config_section=self.model_config_section)
                if self.inference_engine.forecaster.model is None:
                    logger.error(f"MLForecasterStrategy: LightGBM model for {symbol} ({timeframe}) could not be loaded. Strategy will not generate signals.")
                    self.inference_engine = None # Disable if model not loaded
            except Exception as e:
                logger.error(f"Failed to initialize LightGBMInference for {symbol} ({timeframe}): {e}", exc_info=True)
                self.inference_engine = None
        elif self.forecaster_type.lower() == 'xgboost_classifier': # Phase 6 Example
            try:
                from kamikaze_komodo.ml_models.inference_pipelines.xgboost_classifier_inference import XGBoostClassifierInference # Attempt import
                self.inference_engine = XGBoostClassifierInference(symbol, timeframe, model_config_section=self.model_config_section)
                if self.inference_engine.forecaster.model is None:
                    logger.error(f"MLForecasterStrategy: XGBoost model for {symbol} ({timeframe}) could not be loaded.")
                    self.inference_engine = None
            except ImportError:
                logger.error(f"XGBoostClassifierInference not found. Please ensure it's implemented for forecaster_type 'xgboost_classifier'.")
                self.inference_engine = None
            except Exception as e:
                logger.error(f"Failed to initialize XGBoostClassifierInference for {symbol} ({timeframe}): {e}", exc_info=True)
                self.inference_engine = None
        else:
            logger.error(f"Unsupported forecaster_type: {self.forecaster_type} for MLForecasterStrategy.")

        logger.info(
            f"Initialized MLForecasterStrategy for {symbol} ({timeframe}) "
            f"using {self.forecaster_type}. Long Threshold: {self.long_threshold}, Short Threshold: {self.short_threshold}. "
            f"Shorting Enabled: {self.enable_shorting}"
        )
        # self.current_position_status is inherited

    def _get_prediction(self, current_history_df: pd.DataFrame) -> Optional[Any]: # Can return float or dict for classification
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
        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        """
        Generates signals in batch. For ML, this would involve generating predictions for the whole series.
        """
        if data.empty or self.inference_engine is None:
            logger.warning("Data is empty or no inference engine for MLForecasterStrategy. No signals generated.")
            return pd.Series(index=data.index, dtype='object')

        df_with_indicators = self._calculate_indicators(data) # Adds ATR
        
        predictions_list = []
        prediction_confidences = [] # For classifiers that provide confidence
        min_hist_for_pred = int(self.params.get('min_bars_for_prediction', 50)) # Min bars for features
        
        for i in range(len(df_with_indicators)):
            if i < min_hist_for_pred -1 :
                predictions_list.append(None) 
                prediction_confidences.append(None)
                continue
            
            current_slice = df_with_indicators.iloc[:i+1] # History up to current bar
            pred_output = self._get_prediction(current_slice)

            if isinstance(pred_output, dict): # Classifier output
                predictions_list.append(pred_output.get('predicted_class')) 
                prediction_confidences.append(pred_output.get('confidence'))
            elif isinstance(pred_output, (float, np.floating, int, np.integer)): # Regressor or simple class output
                predictions_list.append(pred_output)
                prediction_confidences.append(None) # Confidence might not be available
            else:
                predictions_list.append(None) 
                prediction_confidences.append(None)
            
        df_with_indicators['prediction'] = predictions_list
        df_with_indicators['prediction_confidence'] = prediction_confidences
        
        if self.forecaster_type.lower() == 'lightgbm': # Assuming regression
            df_with_indicators['prediction'].fillna(0.0, inplace=True) 
        # For classification, None might mean 'hold' or 'no signal'

        signals = pd.Series(index=df_with_indicators.index, dtype='object').fillna(SignalType.HOLD)
        current_pos_state: Optional[SignalType] = None # Simulate state for batch processing

        for i in range(len(df_with_indicators)):
            prediction_value = df_with_indicators['prediction'].iloc[i]
            # prediction_confidence = df_with_indicators['prediction_confidence'].iloc[i] # Not used in this simplified batch logic
            current_sentiment = sentiment_series.iloc[i] if sentiment_series is not None and i < len(sentiment_series) else 0.0
            market_regime = df_with_indicators['market_regime'].iloc[i] if 'market_regime' in df_with_indicators.columns and pd.notna(df_with_indicators['market_regime'].iloc[i]) else None

            long_signal = False
            short_signal = False
            close_long_signal = False
            close_short_signal = False

            if self.forecaster_type.lower() == 'lightgbm': # Regression example
                if prediction_value is not None:
                    if prediction_value > self.long_threshold: long_signal = True
                    if prediction_value < self.short_threshold: short_signal = True
                    if prediction_value < self.exit_long_threshold: close_long_signal = True
                    if prediction_value > self.exit_short_threshold: close_short_signal = True
            elif self.forecaster_type.lower() == 'xgboost_classifier': # Classification example
                # Assuming prediction_value holds the class: 0=UP, 1=DOWN, 2=SIDEWAYS (example)
                if prediction_value == 0: long_signal = True
                if prediction_value == 1: short_signal = True
                if current_pos_state == SignalType.LONG and (prediction_value == 1 or prediction_value == 2) : close_long_signal = True
                if current_pos_state == SignalType.SHORT and (prediction_value == 0 or prediction_value == 2) : close_short_signal = True
            
            # Regime adaptation placeholder
            if market_regime is not None:
                 # Example: if regime is strongly bearish (e.g., regime == 2), suppress longs
                 # if market_regime == 2 and long_signal: long_signal = False 
                 pass


            # Apply signals based on state
            if current_pos_state == SignalType.LONG:
                if close_long_signal:
                    signals.iloc[i] = SignalType.CLOSE_LONG
                    current_pos_state = None
            elif current_pos_state == SignalType.SHORT:
                if close_short_signal:
                    signals.iloc[i] = SignalType.CLOSE_SHORT
                    current_pos_state = None
            else: # No position
                sentiment_ok_for_long = True
                if app_settings.enable_sentiment_analysis and self.params.get('sentimentfilter_long_threshold') is not None: 
                    if current_sentiment < self.params.get('sentimentfilter_long_threshold', -999.0): # Use strategy specific if available
                        sentiment_ok_for_long = False
                
                sentiment_ok_for_short = True
                if app_settings.enable_sentiment_analysis and self.params.get('sentimentfilter_short_threshold') is not None:
                    if current_sentiment > self.params.get('sentimentfilter_short_threshold', 999.0): # Use strategy specific if available
                        sentiment_ok_for_short = False

                if long_signal and sentiment_ok_for_long:
                    signals.iloc[i] = SignalType.LONG
                    current_pos_state = SignalType.LONG
                elif short_signal and self.enable_shorting and sentiment_ok_for_short:
                    signals.iloc[i] = SignalType.SHORT
                    current_pos_state = SignalType.SHORT
        
        logger.info(f"Generated MLForecaster signals (vectorized). Longs: {signals.eq(SignalType.LONG).sum()}, Shorts: {signals.eq(SignalType.SHORT).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}, CloseShorts: {signals.eq(SignalType.CLOSE_SHORT).sum()}")
        return signals

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data) # Appends current bar_data to self.data_history

        atr_period = int(self.params.get('atr_period', 14))
        required_bars_for_atr = max(20, atr_period) 
        if len(self.data_history) >= required_bars_for_atr:
            history_slice_for_atr = self.data_history.tail(required_bars_for_atr * 2) # More data for stable ATR
            temp_df_for_atr = self._calculate_indicators(history_slice_for_atr) 
            if not temp_df_for_atr.empty and 'atr' in temp_df_for_atr.columns and pd.notna(temp_df_for_atr['atr'].iloc[-1]):
                bar_data.atr = temp_df_for_atr['atr'].iloc[-1]
        elif bar_data.atr is None : 
            bar_data.atr = None
            
        if self.inference_engine is None:
            return SignalType.HOLD 

        min_history_len = int(self.params.get('min_bars_for_prediction', 50)) 
        if len(self.data_history) < min_history_len:
            logger.debug(f"Not enough history ({len(self.data_history)}/{min_history_len}) for ML prediction on {self.symbol}.")
            return SignalType.HOLD

        prediction_output = self._get_prediction(self.data_history) 

        if prediction_output is None:
            logger.debug(f"No prediction received for {self.symbol} at {bar_data.timestamp}.")
            bar_data.prediction_value = None
            bar_data.prediction_confidence = None 
            return SignalType.HOLD
        
        prediction_numeric_value = 0.0 
        predicted_class_label = None # For classifier

        if isinstance(prediction_output, dict): # Classifier output
            # Example: {'predicted_class': 0, 'confidence': 0.85, 'probabilities': [0.85, 0.10, 0.05]}
            # Where 0=UP, 1=DOWN, 2=SIDEWAYS
            predicted_class_label = prediction_output.get('predicted_class')
            bar_data.prediction_confidence = prediction_output.get('confidence')
            # For thresholding, we might use confidence or a primary probability if available
            # If 'predicted_class' is what we act on, store it in prediction_value for consistency in BarData
            bar_data.prediction_value = float(predicted_class_label) if predicted_class_label is not None else None
        elif isinstance(prediction_output, (float, np.floating, int, np.integer)): # Regressor output
            prediction_numeric_value = float(prediction_output)
            bar_data.prediction_value = prediction_numeric_value
            # Confidence might not be directly available for simple regression
            bar_data.prediction_confidence = None # Or set based on other metrics if model provides
        else: 
            logger.warning(f"Unknown prediction output type: {type(prediction_output)}. Cannot process signal.")
            return SignalType.HOLD
        
        # Use market_regime if available on bar_data (populated by engine or other means)
        current_market_regime = bar_data.market_regime

        signal_to_return = SignalType.HOLD
        effective_sentiment = sentiment_score if sentiment_score is not None else (bar_data.sentiment_score if bar_data.sentiment_score is not None else 0.0)
        
        logger.debug(f"{bar_data.timestamp} - {self.symbol}: ML Pred Output={prediction_output}, Sent={effective_sentiment:.2f}, Regime={current_market_regime}")

        long_signal_triggered = False
        short_signal_triggered = False
        close_long_signal_triggered = False
        close_short_signal_triggered = False

        # --- Signal Logic based on forecaster type ---
        if self.forecaster_type.lower() == 'lightgbm': # Regression
            if prediction_numeric_value > self.long_threshold: long_signal_triggered = True
            if prediction_numeric_value < self.short_threshold: short_signal_triggered = True
            if prediction_numeric_value < self.exit_long_threshold: close_long_signal_triggered = True
            if prediction_numeric_value > self.exit_short_threshold: close_short_signal_triggered = True
        elif self.forecaster_type.lower() == 'xgboost_classifier': # Classification
            # Example: 0=UP, 1=DOWN, 2=SIDEWAYS (as defined in XGBoostClassifierForecaster)
            if predicted_class_label == 0: long_signal_triggered = True
            if predicted_class_label == 1: short_signal_triggered = True
            if self.current_position_status == SignalType.LONG and (predicted_class_label == 1 or predicted_class_label == 2):
                close_long_signal_triggered = True
            if self.current_position_status == SignalType.SHORT and (predicted_class_label == 0 or predicted_class_label == 2):
                close_short_signal_triggered = True
        
        # Regime-based adaptation (example)
        if current_market_regime is not None:
            # Example: if current_market_regime indicates very high volatility (e.g., regime 2), reduce exposure or avoid trades
            # if current_market_regime == 2: # Assuming 2 is 'high volatility / avoid'
            # long_signal_triggered = False
            # short_signal_triggered = False
            # logger.info(f"ML signals suppressed due to market regime {current_market_regime}")
            pass # Implement specific regime logic here

        # Check min prediction confidence if available
        if bar_data.prediction_confidence is not None and bar_data.prediction_confidence < self.min_prediction_confidence:
            logger.info(f"{bar_data.timestamp} - ML signal for {self.symbol} ({'LONG' if long_signal_triggered else 'SHORT' if short_signal_triggered else 'N/A'}) suppressed due to low confidence ({bar_data.prediction_confidence:.2f} < {self.min_prediction_confidence}).")
            long_signal_triggered = False
            short_signal_triggered = False


        # --- Position Management ---
        if self.current_position_status == SignalType.LONG:
            if close_long_signal_triggered:
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - MLForecaster CLOSE_LONG for {self.symbol}. Prediction: {prediction_output}")
        elif self.current_position_status == SignalType.SHORT:
            if close_short_signal_triggered:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - MLForecaster CLOSE_SHORT for {self.symbol}. Prediction: {prediction_output}")
        else: # No current position
            if long_signal_triggered:
                sentiment_ok_for_long = True
                # Use strategy-specific sentiment thresholds if they exist in self.params, otherwise global from app_settings
                strat_sentiment_long_thresh = self.params.get('sentimentfilter_long_threshold', app_settings.sentiment_filter_threshold_long if app_settings else None)
                if app_settings and app_settings.enable_sentiment_analysis and strat_sentiment_long_thresh is not None:
                    if effective_sentiment < strat_sentiment_long_thresh:
                        logger.info(f"{bar_data.timestamp} - MLForecaster LONG for {self.symbol} SUPPRESSED by sentiment ({effective_sentiment:.2f} < {strat_sentiment_long_thresh}). Prediction: {prediction_output}")
                        sentiment_ok_for_long = False
                
                if sentiment_ok_for_long:
                    signal_to_return = SignalType.LONG
                    self.current_position_status = SignalType.LONG
                    logger.info(f"{bar_data.timestamp} - MLForecaster LONG for {self.symbol}. Prediction: {prediction_output}. ATR: {bar_data.atr if bar_data.atr is not None else 'N/A'}")

            elif short_signal_triggered and self.enable_shorting:
                sentiment_ok_for_short = True
                strat_sentiment_short_thresh = self.params.get('sentimentfilter_short_threshold', app_settings.sentiment_filter_threshold_short if app_settings else None)
                if app_settings and app_settings.enable_sentiment_analysis and strat_sentiment_short_thresh is not None:
                    if effective_sentiment > strat_sentiment_short_thresh:
                        logger.info(f"{bar_data.timestamp} - MLForecaster SHORT for {self.symbol} SUPPRESSED by sentiment ({effective_sentiment:.2f} > {strat_sentiment_short_thresh}). Prediction: {prediction_output}")
                        sentiment_ok_for_short = False

                if sentiment_ok_for_short:
                    signal_to_return = SignalType.SHORT
                    self.current_position_status = SignalType.SHORT
                    logger.info(f"{bar_data.timestamp} - MLForecaster SHORT for {self.symbol}. Prediction: {prediction_output}. ATR: {bar_data.atr if bar_data.atr is not None else 'N/A'}")
        
        return signal_to_return