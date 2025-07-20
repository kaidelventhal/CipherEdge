# FILE: kamikaze_komodo/strategy_framework/strategies/funding_rate_arbitrage_strategy.py
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

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
        # The 'symbol' for the base class is the primary one, e.g., the futures contract.
        super().__init__(symbol=symbol_futures, timeframe=timeframe, params=params)
        
        self.symbol_spot = symbol_spot
        self.symbol_futures = symbol_futures
        
        # Strategy parameters
        self.entry_funding_rate_threshold = float(self.params.get('entry_funding_rate_threshold', 0.0002)) # e.g., 0.02%
        self.exit_funding_rate_threshold = float(self.params.get('exit_funding_rate_threshold', 0.00005)) # e.g., 0.005%
        self.max_basis_pct_threshold = float(self.params.get('max_basis_pct_threshold', 1.0)) # e.g., max 1% deviation between spot and futures

        # State management
        self.in_position = False
        self.position_type = None # "positive_carry" (short futures) or "negative_carry" (long futures)
        
        logger.info(
            f"Initialized FundingRateArbitrageStrategy for Spot:{self.symbol_spot}/Futures:{self.symbol_futures}. "
            f"Entry Threshold: {self.entry_funding_rate_threshold}, Exit Threshold: {self.exit_funding_rate_threshold}"
        )

    def prepare_data(self, data_spot_df: pd.DataFrame, data_futures_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares and aligns data for both spot and futures assets.
        This method is expected to be called by a specialized backtesting setup.
        """
        # For this strategy, data preparation is primarily about ensuring alignment.
        # The backtesting engine is expected to provide aligned data.
        # We can add features like the basis here.
        
        merged_df = pd.merge(
            data_spot_df[['close']].rename(columns={'close': 'spot_close'}),
            data_futures_df[['close', 'funding_rate']].rename(columns={'close': 'futures_close'}),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Calculate basis and basis percentage
        merged_df['basis'] = merged_df['futures_close'] - merged_df['spot_close']
        merged_df['basis_pct'] = (merged_df['basis'] / merged_df['spot_close']) * 100
        
        # Add these calculated features back to the original dataframes
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
        
        # Check if basis is within acceptable limits to avoid large price divergences
        if abs(basis_pct) > self.max_basis_pct_threshold:
            return "close" # Signal to close any position due to high divergence

        # Entry condition for positive carry (funding is positive, short futures)
        if funding_rate > self.entry_funding_rate_threshold:
            return "positive_carry"

        # Entry condition for negative carry (funding is negative, long futures)
        if funding_rate < -self.entry_funding_rate_threshold:
            return "negative_carry"
            
        # Exit condition for positive carry
        if self.position_type == "positive_carry" and funding_rate < self.exit_funding_rate_threshold:
            return "close"

        # Exit condition for negative carry
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
                # Close Short Futures, Close Long Spot
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_SHORT, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_LONG, symbol=self.symbol_spot))
            elif self.position_type == "negative_carry":
                # Close Long Futures, Close Short Spot
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_LONG, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.CLOSE_SHORT, symbol=self.symbol_spot))
            self.in_position = False
            self.position_type = None
            return commands

        if opportunity and not self.in_position:
            if opportunity == "positive_carry":
                logger.info(f"Entering POSITIVE CARRY arbitrage at {bar_data_spot.timestamp}. Funding Rate: {bar_data_futures.funding_rate}")
                # Short Futures, Long Spot
                commands.append(SignalCommand(signal_type=SignalType.SHORT, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.LONG, symbol=self.symbol_spot))
                self.in_position = True
                self.position_type = "positive_carry"
                return commands

            if opportunity == "negative_carry":
                logger.info(f"Entering NEGATIVE CARRY arbitrage at {bar_data_spot.timestamp}. Funding Rate: {bar_data_futures.funding_rate}")
                # Long Futures, Short Spot
                commands.append(SignalCommand(signal_type=SignalType.LONG, symbol=self.symbol_futures))
                commands.append(SignalCommand(signal_type=SignalType.SHORT, symbol=self.symbol_spot))
                self.in_position = True
                self.position_type = "negative_carry"
                return commands

        return SignalType.HOLD