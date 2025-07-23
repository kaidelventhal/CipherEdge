from typing import Dict, Any, Optional, Union, List
from collections import Counter
import pandas as pd
from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class CompositeStrategy(BaseStrategy):
    """
    A meta-strategy that combines signals from multiple component strategies.
    """
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        params: Optional[Dict[str, Any]] = None,
        components: List[BaseStrategy] = None,
        method: str = 'weighted_vote',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initializes the CompositeStrategy.

        Args:
            symbol (str): The trading symbol.
            timeframe (str): The trading timeframe.
            params (Optional[Dict[str, Any]]): General parameters for the composite strategy.
            components (List[BaseStrategy]): A list of instantiated strategy objects to combine.
            method (str): The method for combining signals ('weighted_vote' or 'hierarchical').
            weights (Optional[Dict[str, float]]): A dictionary mapping strategy names to their weights for 'weighted_vote'.
        """
        super().__init__(symbol, timeframe, params)
        
        if not components:
            raise ValueError("CompositeStrategy requires a list of component strategies.")
            
        self.components = components
        self.method = method.lower()
        self.weights = weights if weights else {comp.name: 1.0 for comp in self.components}

        self.agreement_threshold = int(self.params.get('agreement_threshold', 2))

        logger.info(f"Initialized CompositeStrategy '{self.name}' with method '{self.method}'.")
        for comp in self.components:
            logger.info(f" - Component: {comp.name}, Weight: {self.weights.get(comp.name, 1.0)}")

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data by running the prepare_data method for each component strategy.
        """
        prepared_df = data.copy()
        for component in self.components:
            prepared_df = component.prepare_data(prepared_df)
        return prepared_df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        """
        Generates a signal by aggregating signals from all component strategies.
        """
        signals: List[SignalType] = []
        for component in self.components:
            signal = component.on_bar_data(current_bar)
            if isinstance(signal, list) and signal:
                signals.append(signal[0].signal_type)
            elif isinstance(signal, SignalType):
                signals.append(signal)

        if not signals:
            return SignalType.HOLD

        if self.method == 'weighted_vote':
            return self._get_weighted_vote_signal(signals)
        elif self.method == 'hierarchical':
            return self._get_hierarchical_signal(signals)
        else:
            logger.warning(f"Unknown composite method: {self.method}. Defaulting to HOLD.")
            return SignalType.HOLD

    def _get_weighted_vote_signal(self, signals: List[SignalType]) -> SignalType:
        """
        Calculates a final signal based on weighted votes.
        LONG = +1, SHORT = -1, others = 0
        """
        score = 0.0
        total_weight = 0.0
        
        for i, signal in enumerate(signals):
            comp_name = self.components[i].name
            weight = self.weights.get(comp_name, 1.0)
            
            if signal == SignalType.LONG:
                score += weight
            elif signal == SignalType.SHORT and self.enable_shorting:
                score -= weight
            
            total_weight += weight
        
        if total_weight == 0:
            return SignalType.HOLD

        normalized_score = score / total_weight
        
        vote_threshold = float(self.params.get('vote_threshold', 0.5))

        if self.current_position_status is None:
            if normalized_score >= vote_threshold:
                self.current_position_status = SignalType.LONG
                return SignalType.LONG
            elif normalized_score <= -vote_threshold:
                self.current_position_status = SignalType.SHORT
                return SignalType.SHORT
        elif self.current_position_status == SignalType.LONG and normalized_score < 0:
            self.current_position_status = None
            return SignalType.CLOSE_LONG
        elif self.current_position_status == SignalType.SHORT and normalized_score > 0:
            self.current_position_status = None
            return SignalType.CLOSE_SHORT
            
        return SignalType.HOLD

    def _get_hierarchical_signal(self, signals: List[SignalType]) -> SignalType:
        """
        Requires N out of M strategies to agree on a signal.
        """
        long_votes = signals.count(SignalType.LONG)
        short_votes = signals.count(SignalType.SHORT)

        if self.current_position_status is None:
            if long_votes >= self.agreement_threshold:
                self.current_position_status = SignalType.LONG
                return SignalType.LONG
            elif short_votes >= self.agreement_threshold and self.enable_shorting:
                self.current_position_status = SignalType.SHORT
                return SignalType.SHORT
        elif self.current_position_status == SignalType.LONG and short_votes > 0:
             self.current_position_status = None
             return SignalType.CLOSE_LONG
        elif self.current_position_status == SignalType.SHORT and long_votes > 0:
             self.current_position_status = None
             return SignalType.CLOSE_SHORT

        return SignalType.HOLD