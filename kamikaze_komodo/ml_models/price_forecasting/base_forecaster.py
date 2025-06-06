# FILE: kamikaze_komodo/ml_models/price_forecasting/base_forecaster.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any, Union
from kamikaze_komodo.app_logger import get_logger
logger = get_logger(__name__)
class BasePriceForecaster(ABC):
    """
    Abstract base class for price forecasting models.
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.params = params if params is not None else {}
        self.model: Any = None
        # FIX: The call to load_model is removed from the base class.
        # Subclasses are now responsible for calling it at the appropriate time
        # (i.e., after the model architecture has been defined).
        logger.info(f"{self.__class__.__name__} initialized with model_path: {model_path}, params: {self.params}")
    @abstractmethod
    def train(self, historical_data: pd.DataFrame, target_column: str = 'close', feature_columns: Optional[list] = None):
        """
        Trains the forecasting model.
        Args:
            historical_data (pd.DataFrame): DataFrame with historical OHLCV and feature data.
            target_column (str): The name of the column to predict.
            feature_columns (Optional[list]): List of column names to be used as features. If None, uses defaults.
        """
        pass
    @abstractmethod
    def predict(self, new_data: pd.DataFrame, feature_columns: Optional[list] = None) -> Union[pd.Series, float, None]:
        """
        Makes predictions on new data.
        Args:
            new_data (pd.DataFrame): DataFrame with the latest data for prediction.
                                     For bar-by-bar, this might be a single row or a lookback window.
            feature_columns (Optional[list]): List of column names to be used as features, must match training.
        Returns:
            Union[pd.Series, float, None]: Predicted value(s) or None if prediction fails.
                                           Could be a series for multi-step or single float for next step.
        """
        pass
    @abstractmethod
    def save_model(self, path: str):
        """
        Saves the trained model to the specified path.
        """
        pass
    @abstractmethod
    def load_model(self, path: str):
        """
        Loads a trained model from the specified path.
        """
        pass
    @abstractmethod
    def create_features(self, data: pd.DataFrame, feature_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Creates features for the model from raw data.
        """
        pass