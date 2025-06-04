# kamikaze_komodo/strategy_framework/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params if params is not None else {}
        self.current_position: Optional[SignalType] = None # None, LONG, SHORT
        self.data_history = pd.DataFrame() # To store historical data for calculations
        logger.info(f"Initialized BaseStrategy for {symbol} ({timeframe}) with params: {self.params}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on the provided historical data.
        This method should be implemented by concrete strategy classes.
        It is typically called once during backtesting setup or for historical analysis.

        Args:
            data (pd.DataFrame): DataFrame with historical OHLCV data, indexed by timestamp.
                                 Expected columns: 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            pd.Series: A Pandas Series indexed by timestamp, containing SignalType values.
        """
        pass

    @abstractmethod
    def on_bar_data(self, bar_data: BarData) -> Optional[SignalType]:
        """
        Processes a new bar of data and decides on a trading action.
        This method is typically called for each new data point in a live or simulated environment.

        Args:
            bar_data (BarData): The new BarData object.

        Returns:
            Optional[SignalType]: A signal (LONG, SHORT, HOLD, CLOSE_LONG, CLOSE_SHORT) or None if no action.
        """
        pass
        
    def update_data_history(self, new_bar_data: BarData):
        """
        Appends new bar data to the internal history.
        This should be called before on_bar_data if the strategy relies on an updating DataFrame.
        """
        new_row = pd.DataFrame([{
            'open': new_bar_data.open,
            'high': new_bar_data.high,
            'low': new_bar_data.low,
            'close': new_bar_data.close,
            'volume': new_bar_data.volume
        }], index=[new_bar_data.timestamp])
        
        self.data_history = pd.concat([self.data_history, new_row])
        # Optional: Keep only a certain number of recent rows to manage memory
        # max_history_length = self.params.get('max_history_length', 1000)
        # if len(self.data_history) > max_history_length:
        #     self.data_history = self.data_history.iloc[-max_history_length:]

    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameters of the strategy."""
        return self.params

    def set_parameters(self, params: Dict[str, Any]):
        """Updates the parameters of the strategy."""
        self.params.update(params)
        logger.info(f"Strategy {self.__class__.__name__} parameters updated: {self.params}")

    @property
    def name(self) -> str:
        return self.__class__.__name__