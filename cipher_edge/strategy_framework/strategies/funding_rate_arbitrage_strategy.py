import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class FundingRateArbitrageStrategy(BaseStrategy):
    """
    Implements a market-neutral funding rate arbitrage strategy.
    It simultaneously buys a spot asset and sells a futures contract (or vice-versa)
    to collect funding payments while aiming for market neutrality.

    NOTE: This is a specialized strategy. The backtesting engine requires modification
    to handle two simultaneous data feeds (spot and futures) for this strategy to work.
    """
    def __init__(self, symbol_spot: str, symbol_futures: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol=symbol_futures, timeframe=timeframe, params=params)
        
        self.symbol_spot = symbol_spot
        self.symbol_futures = symbol_futures
        
        # Strategy parameters
        self.entry_funding_rate_threshold = float(self.params.get('entry_funding_rate_threshold', 0.0002))
        self.exit_funding_rate_threshold = float(self.params.get('exit_funding_rate_threshold', 0.00005))
        self.max_basis_pct_threshold = float(self.params.get('max_basis_pct_threshold', 1.0)) 
        self.in_position = False
        self.position_type = None 
        logger.info(
            f"Initialized FundingRateArbitrageStrategy for Spot:{self.symbol_spot}/Futures:{self.symbol_futures}. "
            f"Entry Threshold: {self.entry_funding_rate_threshold}, Exit Threshold: {self.exit_funding_rate_threshold}"
        )

    def prepare_data(self, data_spot_df: pd.DataFrame, data_futures_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares and aligns data for both spot and futures assets.
        This method is expected to be called by a specialized backtesting setup.
        """

        
        merged_df = pd.merge(
            data_spot_df[['close']].rename(columns={'close': 'spot_close'}),
            data_futures_df[['close', 'funding_rate']].rename(columns={'close': 'futures_close'}),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        merged_df['basis'] = merged_df['futures_close'] - merged_df['spot_close']
        merged_df['basis_pct'] = (merged_df['basis'] / merged_df['spot_close']) * 100
        
        data_futures_df['basis_pct'] = merged_df['basis_pct']
        
        logger.info("Arbitrage strategy data prepared and basis calculated.")
        return data_spot_df, data_futures_df

    def _calculate_arbitrage_opportunity(self, bar_data_spot: BarData, bar_data_futures: BarData) -> Optional[str]:
        """
        Helper to identify when to enter or exit based on funding rate and basis.
        Returns the type of position to take or None.
        """
        funding_rate = bar_data_futures.funding_rate
        basis_pct = bar_data_futures.basis_pct

        if funding_rate is None or basis_pct is None:
            return None
        
        if abs(basis_pct) > self.max_basis_pct_threshold:
            return "close" 

        if funding_rate > self.entry_funding_rate_threshold:
            return "positive_carry"

        if funding_rate < -self.entry_funding_rate_threshold:
            return "negative_carry"
            
        if self.position_type == "positive_carry" and funding_rate < self.exit_funding_rate_threshold:
            return "close"

        if self.position_type == "negative_carry" and funding_rate > -self.exit_funding_rate_threshold:
            return "close"
            
        return None

    def _calculate_hedge_ratio(self) -> float:
        """
        Helper to determine the ratio of futures to spot.
        For simple dollar neutrality, it's 1.0. More complex strategies might use beta.
        """
        return 1.0

    def on_bar_data(self, bar_data_spot: BarData, bar_data_futures: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        """
        Processes new bars for both spot and futures to decide on trading actions.
        This is a specialized signature handled by the backtesting engine.
        """
        opportunity = self._calculate_arbitrage_opportunity(bar_data_spot, bar_data_futures)
        commands: List[SignalCommand] = []

        if opportunity == "close" and self.in_position:
            logger.info(f"Closing arbitrage position at {bar_data_spot.timestamp}.")
            if self.position_type == "positive_carry":
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_SHORT, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_LONG, symbol=self.symbol_spot))
            elif self.position_type == "negative_carry":
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_LONG, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_SHORT, symbol=self.symbol_spot))
            self.in_position = False
            self.position_type = None
            return commands

        if opportunity and not self.in_position:
            if opportunity == "positive_carry":
                logger.info(f"Entering POSITIVE CARRY arbitrage at {bar_data_spot.timestamp}. Funding Rate: {bar_data_futures.funding_rate}")
                commands.append(SignalCommand(signal_type=SignalType.SHORT, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.LONG, symbol=self.symbol_spot))
                self.in_position = True
                self.position_type = "positive_carry"
                return commands

            if opportunity == "negative_carry":
                logger.info(f"Entering NEGATIVE CARRY arbitrage at {bar_data_spot.timestamp}. Funding Rate: {bar_data_futures.funding_rate}")
                commands.append(SignalCommand(signal_type=SignalType.LONG, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.SHORT, symbol=self.symbol_spot))
                self.in_position = True
                self.position_type = "negative_carry"
                return commands

        return SignalType.HOLD