# kamikaze_komodo/backtesting_engine/optimizer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
import itertools
import optuna # For more advanced optimization like TPE
import asyncio

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.orchestration.portfolio_manager import PortfolioManager
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.config.settings import settings as app_settings
from kamikaze_komodo.portfolio_constructor.asset_allocator import FixedWeightAssetAllocator
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, ATRStopManager, PercentageStopManager, TripleBarrierStopManager
from kamikaze_komodo.risk_control_module.volatility_band_stop_manager import VolatilityBandStopManager
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class StrategyOptimizer:
    """
    Optimizes strategy parameters using various methods like Grid Search or Optuna.
    Supports walk-forward optimization.
    """
    def __init__(
        self,
        strategy_class: type,
        data_feed_df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        optimization_metric: str = 'total_net_profit',
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        position_sizer_class_name: Optional[str] = None, # Note: This is now legacy. Sizing is handled by AssetAllocator.
        position_sizer_params: Optional[Dict[str, Any]] = None,
        stop_manager_class_name: Optional[str] = None,
        stop_manager_params: Optional[Dict[str, Any]] = None,
        sentiment_data_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ):
        self.strategy_class = strategy_class
        self.data_feed_df = data_feed_df
        self.param_grid = param_grid
        self.optimization_metric = optimization_metric
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

        # Legacy parameter, will be ignored by the new engine structure.
        self.position_sizer_class_name = position_sizer_class_name
        if self.position_sizer_class_name:
            logger.warning(f"position_sizer_class_name ('{position_sizer_class_name}') is a legacy parameter and will not be used in the unified backtesting engine.")

        self.position_sizer_params = position_sizer_params if position_sizer_params else {}
        self.stop_manager_class_name = stop_manager_class_name
        self.stop_manager_params = stop_manager_params if stop_manager_params else {}

        self.sentiment_data_df = sentiment_data_df
        self.symbol = symbol if symbol else (app_settings.default_symbol if app_settings else "OPTIMIZE_SYMBOL")
        self.timeframe = timeframe if timeframe else (app_settings.default_timeframe if app_settings else "OPTIMIZE_TF")

        logger.info(f"StrategyOptimizer initialized for {strategy_class.__name__} on {self.symbol} ({self.timeframe}). Optimizing for: {self.optimization_metric}")


    async def _run_backtest_for_params(self, params_set: Dict[str, Any], current_data_feed: pd.DataFrame) -> float:
        """Runs a single backtest for a given set of parameters."""
        try:
            # 1. Create the single strategy instance with the current parameter set
            strategy_instance = self.strategy_class(
                symbol=self.symbol,
                timeframe=self.timeframe,
                params=params_set
            )

            # 2. For single-asset optimization, asset allocation is simply 100% to the asset if a signal exists.
            # We use a FixedWeightAssetAllocator to achieve this. The PortfolioManager will use it.
            asset_allocator = FixedWeightAssetAllocator(target_weights={self.symbol: 1.0})

            # 3. Create a PortfolioManager configured for this single strategy run.
            portfolio_manager = PortfolioManager(
                trading_universe=[self.symbol],
                strategy_instances=[strategy_instance],
                asset_allocator=asset_allocator,
            )

            # 4. Create the stop manager for this run, using parameters from the optimization set.
            stop_manager_instance = None
            if self.stop_manager_class_name:
                stop_module_map = {
                    "ATRStopManager": ATRStopManager,
                    "PercentageStopManager": PercentageStopManager,
                    "TripleBarrierStopManager": TripleBarrierStopManager,
                    "VolatilityBandStopManager": VolatilityBandStopManager,
                }
                StopManagerClass = stop_module_map.get(self.stop_manager_class_name)
                if StopManagerClass:
                    stop_manager_instance = StopManagerClass(params=params_set)
                else:
                    logger.error(f"Could not find stop manager class: {self.stop_manager_class_name}")


            # 5. The data feed for the engine needs to be a dictionary
            data_feeds = {self.symbol: current_data_feed}

            # 6. Instantiate the BacktestingEngine with the specific PM, data, and stop manager
            engine = BacktestingEngine(
                portfolio_manager=portfolio_manager,
                data_feeds=data_feeds,
                initial_capital=self.initial_capital,
                commission_bps=self.commission_bps,
                slippage_bps=self.slippage_bps,
                stop_manager=stop_manager_instance,
            )

            trades_log, final_portfolio, equity_curve = await engine.run()

            # 7. Calculate performance metrics
            risk_free_rate = float(app_settings.config.get('BacktestingPerformance', 'RiskFreeRateAnnual', fallback=0.02)) if app_settings else 0.02
            annual_factor = int(app_settings.config.get('BacktestingPerformance', 'AnnualizationFactor', fallback=252)) if app_settings else 252

            analyzer = PerformanceAnalyzer(
                trades=trades_log,
                initial_capital=self.initial_capital,
                final_capital=final_portfolio['final_portfolio_value'],
                equity_curve_df=equity_curve,
                risk_free_rate_annual=risk_free_rate,
                annualization_factor=annual_factor
            )
            metrics = analyzer.calculate_metrics()
            metric_value = metrics.get(self.optimization_metric, -float('inf'))

            if pd.isna(metric_value):
                metric_value = -float('inf') if self.optimization_metric != 'max_drawdown_pct' else float('inf')

            return float(metric_value)

        except Exception as e:
            logger.error(f"Error during backtest for params {params_set}: {e}", exc_info=True)
            return -float('inf') if self.optimization_metric != 'max_drawdown_pct' else float('inf')


    async def grid_search(self) -> Tuple[Optional[Dict[str, Any]], float, pd.DataFrame]:
        param_names = list(self.param_grid.keys())
        param_value_combinations = list(itertools.product(*self.param_grid.values()))

        results = []
        best_metric = -float('inf')
        if self.optimization_metric == 'max_drawdown_pct':
            best_metric = float('inf')

        best_params = None
        logger.info(f"Starting Grid Search with {len(param_value_combinations)} combinations.")

        for i, combo in enumerate(param_value_combinations):
            current_params = dict(zip(param_names, combo))
            logger.debug(f"Grid Search - Combo {i+1}/{len(param_value_combinations)}: {current_params}")
            metric_value = await self._run_backtest_for_params(current_params, self.data_feed_df)
            results.append({**current_params, 'metric_value': metric_value})

            if self.optimization_metric == 'max_drawdown_pct':
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_params = current_params
            elif metric_value > best_metric:
                best_metric = metric_value
                best_params = current_params

        results_df = pd.DataFrame(results)
        if best_params:
            logger.info(f"Grid Search completed. Best params: {best_params}, Best {self.optimization_metric}: {best_metric:.4f}")
        else:
            logger.warning("Grid Search completed but no best parameters found (all trials might have failed or yielded non-comparable results).")
        return best_params, best_metric, results_df.sort_values(by='metric_value', ascending=(self.optimization_metric == 'max_drawdown_pct'))


    def optuna_optimize(self, n_trials: int = 100, study_name: Optional[str] = None, storage_url: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], float, optuna.study.Study]:
        if not study_name:
            study_name = f"{self.strategy_class.__name__}_{self.symbol.replace('/', '')}_{self.timeframe}_Optimization"

        direction = 'minimize' if self.optimization_metric == 'max_drawdown_pct' else 'maximize'
        study = optuna.create_study(study_name=study_name, storage=storage_url, load_if_exists=True, direction=direction)

        # Optuna's objective function is synchronous. We need to run our async backtest from within it.
        def objective(trial: optuna.trial.Trial) -> float:
            params_set = {}
            for param_name, values in self.param_grid.items():
                if not values:
                    logger.warning(f"Parameter '{param_name}' in grid has empty values. Skipping for Optuna.")
                    continue
                if isinstance(values[0], bool):
                    params_set[param_name] = trial.suggest_categorical(param_name, [True, False])
                elif isinstance(values[0], int) and len(values) >= 2:
                    step = values[2] if len(values) > 2 else 1
                    params_set[param_name] = trial.suggest_int(param_name, values[0], values[1], step=step)
                elif isinstance(values[0], float) and len(values) >= 2:
                    params_set[param_name] = trial.suggest_float(param_name, values[0], values[1])
                elif isinstance(values, list):
                    params_set[param_name] = trial.suggest_categorical(param_name, values)
                else:
                    logger.warning(f"Parameter '{param_name}' in grid has unsupported format for Optuna. Values: {values}. Skipping.")

            # Get the running asyncio event loop and run the async backtest function
            try:
                loop = asyncio.get_running_loop()
                metric_value = loop.run_until_complete(self._run_backtest_for_params(params_set, self.data_feed_df))
            except RuntimeError: # If no loop is running (e.g., script is purely sync)
                metric_value = asyncio.run(self._run_backtest_for_params(params_set, self.data_feed_df))

            return metric_value

        logger.info(f"Starting Optuna optimization with {n_trials} trials. Optimizing for {self.optimization_metric} ({direction}).")
        study.optimize(objective, n_trials=n_trials, timeout=None)

        best_params = None
        best_metric_value = study.best_value if study.best_trial else (-float('inf') if direction == 'maximize' else float('inf'))
        if study.best_trial:
            best_params = study.best_params
            logger.info(f"Optuna optimization completed. Best params: {best_params}, Best {self.optimization_metric}: {best_metric_value:.4f}")
        else:
            logger.warning("Optuna optimization completed but no best trial found (all trials might have failed or yielded non-comparable results).")
        return best_params, best_metric_value, study