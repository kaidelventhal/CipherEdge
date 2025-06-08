# FILE: kamikaze_komodo/strategy_framework/strategies/funding_rate_arb_strategy.py
import pandas as pd
from typing import Dict, Any, Optional, Union, List

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class FundingRateArbStrategy(BaseStrategy):
    """
    A strategy that aims to collect funding payments from perpetual futures.
    This is a simplified version that only shorts the perpetual future when the
    funding rate is positive and high, implying that longs are paying shorts.
    It does not execute the spot leg, as the backtester currently does not
    support multi-exchange or multi-asset-type (spot vs. future) positions
    within a single strategy. The PnL will come from funding payments (simulated
    in the backtester) and the price movement of the future itself.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.entry_threshold = float(self.params.get('entry_threshold', 0.0001))  # e.g., enter short if funding is > 0.01%
        self.exit_threshold = float(self.params.get('exit_threshold', 0.00005)) # e.g., exit if funding drops below 0.005%

        logger.info(
            f"Initialized FundingRateArbStrategy for {symbol} ({timeframe}) "
            f"with Entry Threshold: {self.entry_threshold:.5f} and Exit Threshold: {self.exit_threshold:.5f}. "
            f"This strategy requires shorting to be enabled."
        )
        if not self.enable_shorting:
            logger.error(f"FundingRateArbStrategy requires 'enableshorting = True' but it is False.")

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        # This strategy's logic is primarily bar-by-bar and depends on the funding_rate field.
        logger.warning("generate_signals is not the primary method for FundingRateArbStrategy; logic is in on_bar_data.")
        return pd.Series(index=data.index, dtype='object').fillna(SignalType.HOLD)

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)

        funding_rate = bar_data.funding_rate
        if funding_rate is None:
            # If funding rate is not available on this bar, we cannot make a decision.
            # In a real system, you might hold or use the last known rate. Here we'll hold.
            return SignalType.HOLD

        signal_to_return = SignalType.HOLD

        # Entry logic: If not in a position and funding is high, enter a short to collect payments.
        if self.current_position_status is None:
            if funding_rate > self.entry_threshold and self.enable_shorting:
                signal_to_return = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
                logger.info(f"{bar_data.timestamp} - FundingRateArb SHORT for {self.symbol}. Rate: {funding_rate:.5f} > Entry: {self.entry_threshold:.5f}")

        # Exit logic: If in a short position and funding rate drops, close the position.
        elif self.current_position_status == SignalType.SHORT:
            if funding_rate < self.exit_threshold:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - FundingRateArb CLOSE_SHORT for {self.symbol}. Rate: {funding_rate:.5f} < Exit: {self.exit_threshold:.5f}")
        
        return signal_to_return