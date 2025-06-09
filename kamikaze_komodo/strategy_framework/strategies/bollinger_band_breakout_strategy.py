# kamikaze_komodo/strategy_framework/strategies/bollinger_band_breakout_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class BollingerBandBreakoutStrategy(BaseStrategy):
    """
    Implements a stateless Bollinger Band Breakout strategy.
    Generates entry signals only. Exits are handled by the StopManager.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.bb_period = int(self.params.get('bb_period', 20))
        self.bb_std_dev = float(self.params.get('bb_std_dev', 2.0))
        self.atr_period = int(self.params.get('atr_period', 14))

        logger.info(
            f"Initialized BollingerBandBreakoutStrategy for {symbol} ({timeframe}) "
            f"with BB Period: {self.bb_period}, StdDev: {self.bb_std_dev}. "
            f"Shorting Enabled: {self.enable_shorting}"
        )

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty or len(data_df) < self.bb_period:
            return pd.DataFrame()

        df = data_df.copy()
        try:
            bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
            if bbands is not None and not bbands.empty:
                df['bb_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std_dev:.1f}']
                df['bb_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std_dev:.1f}']

            if all(col in df.columns for col in ['high', 'low', 'close']):
                if len(df) >= self.atr_period:
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
            return df
        except Exception as e:
            logger.error(f"Error calculating BBands indicators: {e}")
            return pd.DataFrame()

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        logger.warning("generate_signals is not the primary method for this strategy; logic is in on_bar_data.")
        return pd.Series(index=data.index, dtype='object').fillna(SignalType.HOLD)

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < self.bb_period + 1:
            return SignalType.HOLD

        df = self._calculate_indicators(self.data_history)
        if df.empty or len(df) < 2 or 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            return SignalType.HOLD

        if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]):
            bar_data.atr = df['atr'].iloc[-1]

        latest_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        latest_bb_upper = df['bb_upper'].iloc[-1]
        prev_bb_upper = df['bb_upper'].iloc[-2]
        latest_bb_lower = df['bb_lower'].iloc[-1]
        prev_bb_lower = df['bb_lower'].iloc[-2]

        if any(pd.isna(v) for v in [latest_close, prev_close, latest_bb_upper, prev_bb_upper, latest_bb_lower, prev_bb_lower]):
            return SignalType.HOLD

        long_breakout = latest_close > latest_bb_upper and prev_close <= prev_bb_upper
        short_breakout = latest_close < latest_bb_lower and prev_close >= prev_bb_lower

        if long_breakout:
            return SignalType.LONG
        elif short_breakout and self.enable_shorting:
            return SignalType.SHORT

        return SignalType.HOLD