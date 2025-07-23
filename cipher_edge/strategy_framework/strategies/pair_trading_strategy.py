import pandas as pd
import pandas_ta as ta
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller 
from typing import Dict, Any, Optional, Union, List
from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.core.enums import SignalType, OrderSide
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger
from cipher_edge.data_handling.data_fetcher import DataFetcher 
from cipher_edge.config.settings import settings as app_settings
from datetime import datetime, timedelta, timezone

logger = get_logger(__name__)

class PairTradingStrategy(BaseStrategy):
    """
    Implements a Pair Trading strategy based on cointegration.
    The 'symbol' parameter in __init__ will be considered asset1.
    Asset2 symbol must be provided in params.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params) 
        
        self.asset1_symbol = symbol
        self.asset2_symbol = self.params.get('asset2_symbol')
        if not self.asset2_symbol:
            raise ValueError("PairTradingStrategy requires 'asset2_symbol' in params.")

        self.cointegration_lookback_days = int(self.params.get('cointegration_lookback_days', 90))
        self.cointegration_test_pvalue_threshold = float(self.params.get('cointegration_test_pvalue_threshold', 0.05))
        self.spread_zscore_entry_threshold = float(self.params.get('spread_zscore_entry_threshold', 2.0))
        self.spread_zscore_exit_threshold = float(self.params.get('spread_zscore_exit_threshold', 0.5))
        self.spread_calculation_window = int(self.params.get('spread_calculation_window', 20)) 
        self.data_history_asset2 = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'atr'])
        self.is_cointegrated = False
        self.hedge_ratio: Optional[float] = None # From cointegration regression

        self.active_pair_trade_leg1_symbol: Optional[str] = None
        self.active_pair_trade_leg2_symbol: Optional[str] = None
        self.active_pair_trade_direction: Optional[str] = None # "long_spread" or "short_spread"

        logger.info(
            f"Initialized PairTradingStrategy for {self.asset1_symbol} / {self.asset2_symbol} ({timeframe}) "
            f"Cointegration Lookback: {self.cointegration_lookback_days} days, p-value: {self.cointegration_test_pvalue_threshold}. "
            f"Z-Score Entry: {self.spread_zscore_entry_threshold}, Exit: {self.spread_zscore_exit_threshold}. "
            f"Spread Window: {self.spread_calculation_window}. Shorting Enabled: {self.enable_shorting}"
        )
        if not self.enable_shorting:
            logger.warning(f"PairTradingStrategy for {self.asset1_symbol}/{self.asset2_symbol} has enable_shorting=False. This strategy requires shorting for one leg.")


    async def initialize_strategy_data(self, historical_data_asset1: pd.DataFrame, historical_data_asset2: pd.DataFrame):
        """
        Checks for cointegration using pre-fetched historical data.
        This method is called once at the start by the runner.
        """
        if historical_data_asset1.empty or historical_data_asset2.empty:
            logger.error("PairTradingStrategy received empty historical data for one or both assets.")
            self.is_cointegrated = False
            return

        self.data_history = historical_data_asset1.copy()
        self.data_history_asset2 = historical_data_asset2.copy()
        
        merged_df = pd.merge(
            self.data_history[['close']], 
            self.data_history_asset2[['close']], 
            left_index=True, 
            right_index=True, 
            how='inner', 
            suffixes=('_asset1', '_asset2')
        )
        merged_df.dropna(inplace=True)

        if len(merged_df) < self.spread_calculation_window * 2: # Need enough data
            logger.warning(f"Not enough synchronized historical data for {self.asset1_symbol} and {self.asset2_symbol} for cointegration analysis (found {len(merged_df)} bars).")
            self.is_cointegrated = False
            return
        
        close_asset1 = merged_df['close_asset1']
        close_asset2 = merged_df['close_asset2']

        model = sm.OLS(close_asset1, sm.add_constant(close_asset2, prepend=True))
        results = model.fit()
        self.hedge_ratio = results.params.iloc[1]

        spread = close_asset1 - self.hedge_ratio * close_asset2
        
        adf_result = adfuller(spread.dropna())
        p_value = adf_result[1]

        if p_value < self.cointegration_test_pvalue_threshold:
            self.is_cointegrated = True
            logger.info(f"Pair {self.asset1_symbol}/{self.asset2_symbol} IS cointegrated. ADF p-value: {p_value:.4f}, Hedge Ratio: {self.hedge_ratio:.4f}")
        else:
            self.is_cointegrated = False
            self.hedge_ratio = None 
            logger.info(f"Pair {self.asset1_symbol}/{self.asset2_symbol} is NOT cointegrated. ADF p-value: {p_value:.4f}")
        
        return


    def _calculate_spread_zscore(self) -> Optional[float]:
        """Calculates the Z-score of the current spread."""
        if self.hedge_ratio is None or len(self.data_history) < self.spread_calculation_window or len(self.data_history_asset2) < self.spread_calculation_window:
            return None
            
        last_ts_asset1 = self.data_history.index[-1]
        if last_ts_asset1 not in self.data_history_asset2.index:
            logger.debug(f"Latest timestamp {last_ts_asset1} for {self.asset1_symbol} not in {self.asset2_symbol} history. Cannot calculate current spread.")
            return None
            
        close1 = self.data_history['close'].loc[last_ts_asset1]
        close2 = self.data_history_asset2['close'].loc[last_ts_asset1]

        if pd.isna(close1) or pd.isna(close2): return None

        hist_close1 = self.data_history['close']
        hist_close2 = self.data_history_asset2['close']
        
        merged_closes = pd.merge(hist_close1.rename('c1'), hist_close2.rename('c2'), left_index=True, right_index=True, how='inner')
        if len(merged_closes) < self.spread_calculation_window: return None

        historical_spread = merged_closes['c1'] - self.hedge_ratio * merged_closes['c2']
        
        if len(historical_spread) < self.spread_calculation_window:
            return None
            
        spread_mean = historical_spread.rolling(window=self.spread_calculation_window).mean().iloc[-1]
        spread_std = historical_spread.rolling(window=self.spread_calculation_window).std().iloc[-1]

        if pd.isna(spread_mean) or pd.isna(spread_std) or spread_std == 0:
            return None
            
        current_spread = close1 - self.hedge_ratio * close2
        z_score = (current_spread - spread_mean) / spread_std
        logger.debug(f"Current Spread for {self.asset1_symbol}/{self.asset2_symbol}: {current_spread:.4f}, Mean: {spread_mean:.4f}, Std: {spread_std:.4f}, Z-Score: {z_score:.2f}")
        return z_score

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        logger.warning("generate_signals is not the primary method for PairTradingStrategy; logic is in on_bar_data.")
        return pd.Series(index=data.index, dtype='object').fillna(SignalType.HOLD)


    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        if bar_data.symbol == self.asset1_symbol:
            self.update_data_history(bar_data)
        else:
            return SignalType.HOLD

        if not self.is_cointegrated or self.hedge_ratio is None:
            return SignalType.HOLD

        current_z_score = self._calculate_spread_zscore()
        if current_z_score is None:
            return SignalType.HOLD

        signals_to_execute: List[SignalCommand] = []

        if self.current_position_status == SignalType.LONG: 
            if current_z_score >= self.spread_zscore_exit_threshold:
                logger.info(f"Exiting LONG SPREAD trade for {self.asset1_symbol}/{self.asset2_symbol}. Z-Score: {current_z_score:.2f}")
                signals_to_execute.append(SignalCommand(signal_type=SignalType.CLOSE_LONG, symbol=self.asset1_symbol))
                signals_to_execute.append(SignalCommand(signal_type=SignalType.CLOSE_SHORT, symbol=self.asset2_symbol))
                self.current_position_status = None
                return signals_to_execute

        elif self.current_position_status == SignalType.SHORT: 
            if current_z_score <= -self.spread_zscore_exit_threshold:
                logger.info(f"Exiting SHORT SPREAD trade for {self.asset1_symbol}/{self.asset2_symbol}. Z-Score: {current_z_score:.2f}")
                signals_to_execute.append(SignalCommand(signal_type=SignalType.CLOSE_SHORT, symbol=self.asset1_symbol))
                signals_to_execute.append(SignalCommand(signal_type=SignalType.CLOSE_LONG, symbol=self.asset2_symbol))
                self.current_position_status = None
                return signals_to_execute
        
        if self.current_position_status is None:
            if current_z_score < -self.spread_zscore_entry_threshold:
                logger.info(f"Entering LONG SPREAD trade for {self.asset1_symbol}/{self.asset2_symbol}. Z-Score: {current_z_score:.2f}")
                signals_to_execute.append(SignalCommand(signal_type=SignalType.LONG, symbol=self.asset1_symbol))
                signals_to_execute.append(SignalCommand(signal_type=SignalType.SHORT, symbol=self.asset2_symbol))
                self.current_position_status = SignalType.LONG # Representing "long the spread"
                return signals_to_execute

            elif current_z_score > self.spread_zscore_entry_threshold:
                logger.info(f"Entering SHORT SPREAD trade for {self.asset1_symbol}/{self.asset2_symbol}. Z-Score: {current_z_score:.2f}")
                signals_to_execute.append(SignalCommand(signal_type=SignalType.SHORT, symbol=self.asset1_symbol))
                signals_to_execute.append(SignalCommand(signal_type=SignalType.LONG, symbol=self.asset2_symbol))
                self.current_position_status = SignalType.SHORT # Representing "short the spread"
                return signals_to_execute

        return SignalType.HOLD