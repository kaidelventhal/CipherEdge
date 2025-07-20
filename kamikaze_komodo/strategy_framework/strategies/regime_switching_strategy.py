# FILE: kamikaze_komodo/strategy_framework/strategies/regime_switching_strategy.py
from typing import Dict, Any, Optional, Union, List, Type
import pandas as pd
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class RegimeSwitchingStrategy(BaseStrategy):
    """
    A meta-strategy that switches between different trading strategies based on the market regime.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None,
                 strategy_mapping: Dict[int, BaseStrategy] = None,
                 regime_labels: Optional[Dict[int, str]] = None):
        super().__init__(symbol, timeframe, params)
        
        if strategy_mapping is None:
            raise ValueError("RegimeSwitchingStrategy requires a 'strategy_mapping' dictionary.")
            
        self.strategy_mapping = strategy_mapping
        self.regime_labels = regime_labels if regime_labels is not None else {}
        self.active_regime: Optional[int] = None
        
        self.regime_confirmation_period = int(self.params.get('regime_confirmation_period', 3))
        # IMPROVEMENT: Add a cooldown period to prevent whipsawing
        self.regime_cooldown_period = int(self.params.get('regime_cooldown_period', 5))
        self.last_regime_switch_bar: int = -self.regime_cooldown_period
        self.previous_regime: Optional[int] = None

        self.pending_regime: Optional[int] = None
        self.consecutive_regime_count: int = 0
        self.current_bar_index: int = 0

        logger.info(f"Initialized RegimeSwitchingStrategy for {symbol} ({timeframe}).")
        logger.info(f"Regime confirmation: {self.regime_confirmation_period} bars, Cooldown: {self.regime_cooldown_period} bars.")
        for regime, strategy in self.strategy_mapping.items():
            regime_name = self.regime_labels.get(regime, f"Regime {regime}")
            logger.info(f" - Mapping {regime_name} to {strategy.name}")
            
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preparing data for all sub-strategies within '{self.name}'...")
        prepared_df = data.copy()
        for regime, strategy in self.strategy_mapping.items():
            prepared_df = strategy.prepare_data(prepared_df)
        return prepared_df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.current_bar_index += 1
        current_regime_from_data = int(current_bar.market_regime) if hasattr(current_bar, 'market_regime') and pd.notna(current_bar.market_regime) else None

        if current_regime_from_data is None:
            return SignalType.HOLD

        if self.pending_regime is None:
            self.pending_regime = current_regime_from_data
            self.consecutive_regime_count = 1
        elif current_regime_from_data == self.pending_regime:
            self.consecutive_regime_count += 1
        else:
            self.pending_regime = current_regime_from_data
            self.consecutive_regime_count = 1

        regime_confirmed = self.consecutive_regime_count >= self.regime_confirmation_period
        
        # IMPROVEMENT: Cooldown logic to prevent whipsawing
        is_in_cooldown = (self.current_bar_index - self.last_regime_switch_bar) < self.regime_cooldown_period
        is_flipping_back = self.pending_regime == self.previous_regime

        if regime_confirmed and self.pending_regime != self.active_regime:
            if is_in_cooldown and is_flipping_back:
                logger.debug(f"Regime change from {self.active_regime} to {self.pending_regime} IGNORED due to cooldown.")
            else:
                old_regime_label = self.regime_labels.get(self.active_regime, self.active_regime)
                new_regime_label = self.regime_labels.get(self.pending_regime, self.pending_regime)
                logger.info(f"Regime change CONFIRMED from '{old_regime_label}' to '{new_regime_label}'.")
                
                self.previous_regime = self.active_regime
                self.active_regime = self.pending_regime
                self.last_regime_switch_bar = self.current_bar_index
                
                if self.current_position_status is not None:
                    close_signal = SignalType.CLOSE_LONG if self.current_position_status == SignalType.LONG else SignalType.CLOSE_SHORT
                    self.current_position_status = None
                    for sub_strategy in self.strategy_mapping.values():
                        sub_strategy.current_position_status = None
                    return close_signal
        
        active_strategy = self.strategy_mapping.get(self.active_regime)

        if active_strategy:
            active_strategy.current_position_status = self.current_position_status
            signal = active_strategy.on_bar_data(current_bar)
            
            if isinstance(signal, SignalType):
                if signal in [SignalType.LONG, SignalType.SHORT]:
                    self.current_position_status = signal
                elif signal in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                    self.current_position_status = None
            
            return signal
        else:
            if self.current_position_status is not None:
                close_signal = SignalType.CLOSE_LONG if self.current_position_status == SignalType.LONG else SignalType.CLOSE_SHORT
                self.current_position_status = None
                return close_signal
            
            return SignalType.HOLD