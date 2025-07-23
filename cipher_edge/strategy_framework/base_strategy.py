from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger
import pandas as pd
from pydantic import BaseModel

logger = get_logger(__name__)

class SignalCommand(BaseModel):
    """Represents a command to be executed by the trading engine."""
    signal_type: SignalType
    symbol: str 
    price: Optional[float] = None 
    
class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        self._name = self.__class__.__name__
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params if params is not None else {}
        self.current_position_status: Optional[SignalType] = None
        
        enable_shorting_val = self.params.get('enableshorting', self.params.get('enable_shorting', False))
        if isinstance(enable_shorting_val, str):
            self.enable_shorting = enable_shorting_val.lower() == 'true'
        else:
            self.enable_shorting = bool(enable_shorting_val)
            
        self.data_history = pd.DataFrame()
        
        logger.info(f"Initialized BaseStrategy '{self.__class__.__name__}' for {symbol} ({timeframe}). Shorting enabled: {self.enable_shorting}")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def update_data_history(self, current_bar: BarData):
        """Helper method for stateful strategies to append the latest bar data."""
        new_row_dict = current_bar.model_dump()
        new_row_df = pd.DataFrame([new_row_dict], index=[current_bar.timestamp])
        
        if not isinstance(new_row_df.index, pd.DatetimeIndex):
            new_row_df.index = pd.to_datetime(new_row_df.index, utc=True)

        if self.data_history.empty:
            self.data_history = new_row_df
        else:
            self.data_history = pd.concat([self.data_history, new_row_df])
            self.data_history = self.data_history[~self.data_history.index.duplicated(keep='last')]


    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all necessary indicators and signal conditions in a vectorized manner.
        """
        logger.info(f"'{self.name}' uses default prepare_data. No indicators pre-calculated.")
        return data

    @abstractmethod
    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        """
        Processes a new bar of data and decides on a trading action.
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return self.params

    def set_parameters(self, params: Dict[str, Any]):
        self.params.update(params)
        enable_shorting_val = self.params.get('enableshorting', self.params.get('enable_shorting', False))
        if isinstance(enable_shorting_val, str):
            self.enable_shorting = enable_shorting_val.lower() == 'true'
        else:
            self.enable_shorting = bool(enable_shorting_val)
        logger.info(f"Strategy {self.name} parameters updated: {self.params}. Shorting enabled: {self.enable_shorting}")