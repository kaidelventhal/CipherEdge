# kamikaze_komodo/strategy_framework/base_strategy.py
# Updated to include optional sentiment_score in on_bar_data

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
        self.current_position_status: Optional[SignalType] = None # Tracks if currently LONG, SHORT or None (no position)
        self.data_history = pd.DataFrame()
        logger.info(f"Initialized BaseStrategy '{self.name}' for {symbol} ({timeframe}) with params: {self.params}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        """
        Generates trading signals based on the provided historical data.
        This method is typically called once during backtesting setup or for historical analysis.

        Args:
            data (pd.DataFrame): DataFrame with historical OHLCV data, indexed by timestamp.
                                 Expected columns: 'open', 'high', 'low', 'close', 'volume'.
                                 May also contain 'atr', 'sentiment_score'.
            sentiment_series (Optional[pd.Series]): Series with historical sentiment scores, indexed by timestamp.

        Returns:
            pd.Series: A Pandas Series indexed by timestamp, containing SignalType values.
        """
        pass

    @abstractmethod
    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None) -> Optional[SignalType]:
        """
        Processes a new bar of data and decides on a trading action.
        This method is typically called for each new data point in a live or simulated environment.

        Args:
            bar_data (BarData): The new BarData object, potentially with .atr or .sentiment_score.
            sentiment_score (Optional[float]): External sentiment score for the current bar.
                                               (Note: bar_data.sentiment_score might also be used if populated by engine)

        Returns:
            Optional[SignalType]: A signal (LONG, SHORT, HOLD, CLOSE_LONG, CLOSE_SHORT) or None if no action.
        """
        pass
        
    def update_data_history(self, new_bar_data: BarData):
        """Appends new bar data to the internal history including ATR and sentiment if available."""
        new_row_data = {
            'open': new_bar_data.open, 'high': new_bar_data.high,
            'low': new_bar_data.low, 'close': new_bar_data.close,
            'volume': new_bar_data.volume
        }
        # Add optional fields if they exist on BarData model
        if hasattr(new_bar_data, 'atr') and new_bar_data.atr is not None:
            new_row_data['atr'] = new_bar_data.atr
        if hasattr(new_bar_data, 'sentiment_score') and new_bar_data.sentiment_score is not None:
            new_row_data['sentiment_score'] = new_bar_data.sentiment_score
            
        new_row = pd.DataFrame([new_row_data], index=[new_bar_data.timestamp])
        
        # Ensure columns match, adding missing ones with NaN if this is the first row with new columns
        if not self.data_history.empty:
            for col in new_row.columns:
                if col not in self.data_history.columns:
                    self.data_history[col] = pd.NA # Or np.nan
            for col in self.data_history.columns:
                 if col not in new_row.columns:
                    new_row[col] = pd.NA


        self.data_history = pd.concat([self.data_history, new_row])
        
        # Optional: Keep only a certain number of recent rows to manage memory
        # max_history_length = self.params.get('max_history_length', 1000)
        # if len(self.data_history) > max_history_length:
        #     self.data_history = self.data_history.iloc[-max_history_length:]

    def get_parameters(self) -> Dict[str, Any]:
        return self.params

    def set_parameters(self, params: Dict[str, Any]):
        self.params.update(params)
        logger.info(f"Strategy {self.__class__.__name__} parameters updated: {self.params}")

    @property
    def name(self) -> str:
        return self.__class__.__name__