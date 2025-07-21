# FILE: kamikaze_komodo/strategy_framework/strategy_manager.py
from typing import List, Dict, Any, Type, Optional
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings

# Import all strategy classes to build a registry
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.strategy_framework.strategies.bollinger_band_breakout_strategy import BollingerBandBreakoutStrategy
from kamikaze_komodo.strategy_framework.strategies.volatility_squeeze_breakout_strategy import VolatilitySqueezeBreakoutStrategy
from kamikaze_komodo.strategy_framework.strategies.funding_rate_strategy import FundingRateStrategy
from kamikaze_komodo.strategy_framework.strategies.ensemble_ml_strategy import EnsembleMLStrategy
from kamikaze_komodo.strategy_framework.strategies.regime_switching_strategy import RegimeSwitchingStrategy
from kamikaze_komodo.strategy_framework.strategies.bollinger_band_mean_reversion_strategy import BollingerBandMeanReversionStrategy
from kamikaze_komodo.strategy_framework.strategies.ehlers_instantaneous_trendline import EhlersInstantaneousTrendlineStrategy
from kamikaze_komodo.strategy_framework.strategies.ml_forecaster_strategy import MLForecasterStrategy
from kamikaze_komodo.strategy_framework.strategies.composite_strategy import CompositeStrategy


logger = get_logger(__name__)

# A registry to map strategy names to their classes
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "EWMACStrategy": EWMACStrategy,
    "BollingerBandBreakoutStrategy": BollingerBandBreakoutStrategy,
    "VolatilitySqueezeBreakoutStrategy": VolatilitySqueezeBreakoutStrategy,
    "FundingRateStrategy": FundingRateStrategy,
    "EnsembleMLStrategy": EnsembleMLStrategy,
    "RegimeSwitchingStrategy": RegimeSwitchingStrategy,
    "BollingerBandMeanReversionStrategy": BollingerBandMeanReversionStrategy,
    "EhlersInstantaneousTrendlineStrategy": EhlersInstantaneousTrendlineStrategy,
    "MLForecasterStrategy": MLForecasterStrategy,
    "CompositeStrategy": CompositeStrategy,
}

MODEL_CACHE = {}

class StrategyManager:
    """
    Manages the loading, initialization, and execution of trading strategies.
    Now includes a factory method to create strategies by name and a cache for ML models.
    """
    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        logger.info("StrategyManager initialized.")

    @staticmethod
    def create_strategy(
        strategy_name: str,
        symbol: str,
        timeframe: str,
        params: Optional[Dict[str, Any]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseStrategy]:
        """
        Factory method to create a strategy instance from its name.
        Handles regular, composite, and configuration variants (e.g., MyStrategy_Variant).
        """
        strategy_class = STRATEGY_REGISTRY.get(strategy_name)
        config_section_name = strategy_name

        if not strategy_class:
            base_strategy_name = strategy_name.split('_')[0]
            strategy_class = STRATEGY_REGISTRY.get(base_strategy_name)
            if not strategy_class:
                logger.error(f"Strategy '{strategy_name}' not found in registry.")
                return None
         
        strategy_params = params or (settings.get_strategy_params(config_section_name) if settings else {})
        init_kwargs = init_kwargs or {}
         
        if strategy_class == CompositeStrategy:
            if not settings:
                logger.error("Settings not available, cannot create CompositeStrategy.")
                return None
            
            comp_params = settings.get_strategy_params('CompositeStrategy')
            components = []
            weights = {}
            i = 1
            while True:
                comp_class_name = comp_params.get(f'component_{i}_class')
                if not comp_class_name: break
                
                comp_weight = float(comp_params.get(f'component_{i}_weight', 1.0))
                
                component_instance = StrategyManager.create_strategy(
                    strategy_name=comp_class_name, symbol=symbol, timeframe=timeframe
                )
                
                if component_instance:
                    components.append(component_instance)
                    weights[component_instance.name] = comp_weight
                else:
                    logger.error(f"Failed to create composite component: {comp_class_name}")
                i += 1
            
            if not components:
                logger.error("CompositeStrategy defined but no components could be created.")
                return None
                
            init_kwargs['components'] = components
            init_kwargs['method'] = comp_params.get('method', 'weighted_vote')
            init_kwargs['weights'] = weights

        if strategy_class == MLForecasterStrategy:
            model_config_section = strategy_params.get('modelconfigsection')
            if model_config_section in MODEL_CACHE:
                init_kwargs['inference_engine'] = MODEL_CACHE[model_config_section]
                logger.debug(f"Reusing cached model for '{model_config_section}'.")

        try:
            instance = strategy_class(symbol, timeframe, params=strategy_params, **init_kwargs)
             
            if strategy_class == MLForecasterStrategy and hasattr(instance, 'inference_engine') and instance.inference_engine:
                model_config_section = strategy_params.get('modelconfigsection')
                if model_config_section not in MODEL_CACHE:
                    MODEL_CACHE[model_config_section] = instance.inference_engine
                    logger.info(f"Cached new ML model instance for '{model_config_section}'.")

            if strategy_name != strategy_class.__name__:
                instance.name = strategy_name
            return instance

        except Exception as e:
            logger.error(f"Failed to create strategy '{strategy_name}': {e}", exc_info=True)
            return None