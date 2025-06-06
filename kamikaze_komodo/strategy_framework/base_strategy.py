# FILE: kamikaze_komodo/strategy_framework/base_strategy.py
# Updated to include optional sentiment_score in on_bar_data
# Updated update_data_history for new BarData fields
# Phase 6: Added market_regime to BarData and data_history.
# Phase 6: Added enable_shorting parameter.
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union # Added List, Union
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
import pandas as pd
from pydantic import BaseModel # Ensure pydantic.BaseModel is imported



logger = get_logger(__name__)

class SignalCommand(BaseModel):
    signal_type: SignalType
    symbol: str
    price: Optional[float] = None
    # Add amount if strategy determines it, otherwise position sizer will.
    # amount: Optional[float] = None 
    related_bar_data: Optional[BarData] = None
    # For pair trades, might include specific instructions for each leg
    custom_params: Optional[Dict[str, Any]] = None

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        self.symbol = symbol # Primary symbol for single-asset strategies or one leg of a pair
        self.timeframe = timeframe
        self.params = params if params is not None else {}
        self.current_position_status: Optional[SignalType] = None # Tracks if currently LONG, SHORT or None (no position)
    
        # Phase 6: Enable shorting based on strategy parameters
        self.enable_shorting: bool = self.params.get('enableshorting', False) # Default to False if not specified
        if isinstance(self.enable_shorting, str): # Handle string 'True'/'False' from config
            self.enable_shorting = self.enable_shorting.lower() == 'true'

        # Initialize data_history with potential columns including new ones from BarData
        self.data_history = pd.DataFrame(columns=[
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'sentiment_score',
            'prediction_value', 'prediction_confidence', # New Phase 5 fields
            'market_regime' # New Phase 6 field
        ])
        logger.info(f"Initialized BaseStrategy '{self.name}' for {symbol} ({timeframe}) with params: {self.params}. Shorting enabled: {self.enable_shorting}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        """
        Generates trading signals based on the provided historical data.
        This method is typically called once during backtesting setup or for historical analysis.
        Args:
            data (pd.DataFrame): DataFrame with historical OHLCV data, indexed by timestamp.
                                Expected columns: 'open', 'high', 'low', 'close', 'volume'.
                                May also contain 'atr', 'sentiment_score', 'prediction_value',
                                'prediction_confidence', 'market_regime'.
            sentiment_series (Optional[pd.Series]): Series with historical sentiment scores, indexed by timestamp.
        Returns:
            pd.Series: A Pandas Series indexed by timestamp, containing SignalType values.
        """
        pass

    @abstractmethod
    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        """
        Processes a new bar of data and decides on a trading action.
        This method is typically called for each new data point in a live or simulated environment.
        Can return a single SignalType or a list of SignalCommands for multi-leg strategies.
        Args:
            bar_data (BarData): The new BarData object, potentially with .atr or .sentiment_score,
                                .prediction_value, .prediction_confidence, .market_regime populated.
            sentiment_score (Optional[float]): External sentiment score for the current bar.
                                            (Note: bar_data.sentiment_score might also be used if populated by engine)
            market_regime_data (Optional[Any]): External market regime data for current bar.
                                                (Note: bar_data.market_regime might also be used if populated by engine)
        Returns:
            Union[Optional[SignalType], List[SignalCommand]]:
                - A single signal (LONG, SHORT, HOLD, CLOSE_LONG, CLOSE_SHORT) or None if no action.
                - A list of SignalCommand objects for multi-leg trades (e.g., pair trading).
        """
        pass
    
    def update_data_history(self, new_bar_data: BarData):
        """Appends new bar data to the internal history including ATR, sentiment, prediction, and regime fields if available."""
        new_timestamp = new_bar_data.timestamp
        new_row_data = {
            'open': new_bar_data.open, 'high': new_bar_data.high,
            'low': new_bar_data.low, 'close': new_bar_data.close,
            'volume': new_bar_data.volume,
            'atr': new_bar_data.atr,
            'sentiment_score': new_bar_data.sentiment_score,
            'prediction_value': new_bar_data.prediction_value,
            'prediction_confidence': new_bar_data.prediction_confidence,
            'market_regime': new_bar_data.market_regime,
        }
        # Use .loc to append the new row, which handles dtypes better and avoids FutureWarnings
        for col, value in new_row_data.items():
            if col in self.data_history.columns:
                self.data_history.loc[new_timestamp, col] = value


    def get_parameters(self) -> Dict[str, Any]:
        return self.params

    def set_parameters(self, params: Dict[str, Any]):
        self.params.update(params)
        # Update shorting capability if specified in new params
        if 'enableshorting' in self.params:
            self.enable_shorting = str(self.params['enableshorting']).lower() == 'true'
        logger.info(f"Strategy {self.__class__.__name__} parameters updated: {self.params}. Shorting enabled: {self.enable_shorting}")

    @property
    def name(self) -> str:
        return self.__class__.__name__

# Add Pydantic BaseModel for SignalCommand if not already defined elsewhere (e.g., in core.models if broadly used)
# For now, defining it here for clarity in BaseStrategy context.