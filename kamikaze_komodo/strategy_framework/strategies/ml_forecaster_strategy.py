# FILE: kamikaze_komodo/strategy_framework/strategies/ml_forecaster_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
import os
import joblib
import numpy as np
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class MLForecasterStrategy(BaseStrategy):
    """
    A strategy that uses a primary model to predict side and a secondary
    "meta-model" to predict the probability of success (confidence).
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.primary_model = None
        self.meta_model = None
        
        # Load the pre-trained primary and meta models
        base_path = os.path.join(PROJECT_ROOT, "ml_models/trained_models")
        primary_model_path = os.path.join(base_path, f"primary_{symbol.replace('/', '_')}.joblib")
        meta_model_path = os.path.join(base_path, f"meta_{symbol.replace('/', '_')}.joblib")

        try:
            if os.path.exists(primary_model_path):
                self.primary_model = joblib.load(primary_model_path)
                logger.info(f"Primary model loaded for {symbol} from {primary_model_path}")
            else:
                logger.error(f"Primary model not found at {primary_model_path}")

            if os.path.exists(meta_model_path):
                self.meta_model = joblib.load(meta_model_path)
                logger.info(f"Meta model loaded for {symbol} from {meta_model_path}")
            else:
                logger.error(f"Meta model not found at {meta_model_path}")
        except Exception as e:
            logger.error(f"Error loading ML models for {symbol}: {e}", exc_info=True)
            self.primary_model = self.meta_model = None
            
        self.long_confidence_threshold = float(self.params.get('longconfidencethreshold', 0.55))
        self.short_confidence_threshold = float(self.params.get('shortconfidencethreshold', 0.55))


    def _create_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        df = data_df.copy()
        for lag in [1, 3, 5, 10]:
            df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        df['volatility_20'] = df['log_return_lag_1'].rolling(window=20).std()
        
        # Add ATR for risk management modules
        if all(col in df.columns for col in ['high', 'low', 'close']):
            atr_period = int(self.params.get('atr_period', 14))
            if len(df) >= atr_period:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        
        return df.dropna()

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        logger.warning("generate_signals is not implemented for this real-time focused ML strategy.")
        return pd.Series(index=data.index, dtype='object').fillna(SignalType.HOLD)

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        
        if self.primary_model is None or self.meta_model is None or len(self.data_history) < 50:
            return SignalType.HOLD

        features_df = self._create_features(self.data_history)
        if features_df.empty: return SignalType.HOLD
            
        latest_features = features_df.iloc[[-1]] # Keep as DataFrame

        # 1. Get side prediction from primary model
        side_prediction = self.primary_model.predict(latest_features)[0]
        primary_pred_proba = self.primary_model.predict_proba(latest_features)[0, 1]

        # 2. Get confidence from meta model
        meta_features = pd.DataFrame({'primary_pred_proba': [primary_pred_proba]})
        confidence = self.meta_model.predict_proba(meta_features)[0, 1] # Probability of class 1 (win)
        
        bar_data.prediction_confidence = confidence

        # 3. Generate Signal based on side and confidence
        signal_to_return = SignalType.HOLD
        
        if side_prediction == 1 and confidence > self.long_confidence_threshold:
            if self.current_position_status != SignalType.LONG:
                signal_to_return = SignalType.LONG
                self.current_position_status = SignalType.LONG
        elif side_prediction == -1 and confidence > self.short_confidence_threshold:
            if self.current_position_status != SignalType.SHORT and self.enable_shorting:
                signal_to_return = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
        else: # Close position if confidence drops or side flips
            if self.current_position_status == SignalType.LONG:
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
            elif self.current_position_status == SignalType.SHORT:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None

        logger.info(f"{bar_data.timestamp} - MLMeta Signal: {signal_to_return.value}, Side Pred: {side_prediction}, Confidence: {confidence:.2f}")

        return signal_to_return