# FILE: kamikaze_komodo/backtesting_engine/optimizer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import itertools

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.config.settings import settings as app_settings
from kamikaze_komodo.risk_control_module.position_sizer import FixedFractionalPositionSizer
from kamikaze_komodo.risk_control_module.stop_manager import PercentageStopManager
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class StrategyOptimizer:
    """
    Optimizes strategy parameters using Grid Search for Walk-Forward Optimization.
    Includes a warmup period to ensure indicators are ready.
    """
    def __init__(
        self,
        strategy_class: type,
        data_feed_df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        optimization_metric: str = 'sortino_ratio',
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        slippage_model_type: str = 'fixed', # <-- ADDED
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        warmup_period: int = 200,
        strategy_init_kwargs: Optional[Dict[str, Any]] = None # <-- ADDED
    ):
        self.strategy_class = strategy_class
        self.data_feed_df = data_feed_df
        self.param_grid = param_grid if param_grid else {}
        self.optimization_metric = optimization_metric
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.slippage_model_type = slippage_model_type # <-- ADDED
        self.warmup_period = warmup_period
        self.symbol = symbol or (app_settings.default_symbol if app_settings else "OPTIMIZE_SYMBOL")
        self.timeframe = timeframe or (app_settings.default_timeframe if app_settings else "OPTIMIZE_TF")
        self.strategy_init_kwargs = strategy_init_kwargs or {} # <-- ADDED
        logger.info(f"StrategyOptimizer initialized for {strategy_class.__name__}. Optimizing for: {optimization_metric}")

    def _run_backtest_for_params(self, params_set: Dict[str, Any], data_feed: pd.DataFrame) -> Dict[str, Any]:
        """Runs a single backtest and returns a dictionary of all performance metrics."""
        try:
            base_params = app_settings.get_strategy_params(self.strategy_class.__name__) if app_settings else {}
            combined_params = {**base_params, **params_set}

            # MODIFIED: Pass strategy_init_kwargs to the strategy constructor
            strategy_instance = self.strategy_class(
                symbol=self.symbol,
                timeframe=self.timeframe,
                params=combined_params,
                **self.strategy_init_kwargs
            )
            
            # MODIFIED: Pass the slippage model type to the engine
            engine = BacktestingEngine(
                data_feed_df=data_feed,
                strategy=strategy_instance,
                initial_capital=self.initial_capital,
                commission_bps=self.commission_bps,
                slippage_bps=self.slippage_bps,
                slippage_model_type=self.slippage_model_type
            )
            trades_log, final_portfolio, equity_curve = engine.run()
            
            analyzer = PerformanceAnalyzer(
                trades=trades_log,
                initial_capital=self.initial_capital,
                final_capital=final_portfolio['final_portfolio_value'],
                equity_curve_df=equity_curve
            )
            return analyzer.calculate_metrics()
        except Exception as e:
            logger.error(f"Error during backtest for params {params_set}: {e}", exc_info=True)
            return {}

    def grid_search(self) -> Tuple[Optional[Dict[str, Any]], float, pd.DataFrame]:
        """Performs grid search on the entire data_feed_df provided to the optimizer instance."""
        param_names = list(self.param_grid.keys())
        
        if not self.param_grid or not all(self.param_grid.values()):
            combinations = [{}]
        else:
            combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*self.param_grid.values())]

        results = []
        best_metric = -float('inf')
        best_params = None

        logger.info(f"Starting Grid Search with {len(combinations)} combinations.")

        for i, params in enumerate(combinations):
            logger.debug(f"Grid Search - Combo {i+1}/{len(combinations)}: {params}")
            
            metrics = self._run_backtest_for_params(params, self.data_feed_df)
            metric_value = metrics.get(self.optimization_metric)
            
            if metric_value is None or pd.isna(metric_value):
                metric_value = -float('inf')

            results.append({**params, 'metric_value': metric_value})

            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params

        results_df = pd.DataFrame(results)
        if best_params is not None:
            logger.info(f"Grid Search completed. Best params: {best_params}, Best {self.optimization_metric}: {best_metric:.4f}")
        else:
            logger.warning("Grid Search completed but no best parameters found.")
        
        return best_params, best_metric, results_df

    def walk_forward_optimization(
        self,
        training_period_bars: int,
        testing_period_bars: int,
        step_size_bars: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(self.data_feed_df.index, pd.DatetimeIndex):
            raise ValueError("data_feed_df must have a DatetimeIndex for WFO.")

        full_data = self.data_feed_df
        n_total_bars = len(full_data)

        if self.warmup_period + training_period_bars + testing_period_bars > n_total_bars:
            logger.error(f"Not enough data for even one WFO window. Required: {self.warmup_period + training_period_bars + testing_period_bars}, Available: {n_total_bars}")
            return []

        results_over_time = []
        start_idx = self.warmup_period

        logger.info(f"Starting WFO. Train: {training_period_bars}, Test: {testing_period_bars}, Step: {step_size_bars}, Warmup: {self.warmup_period}")

        while start_idx + training_period_bars <= n_total_bars:
            train_slice_start = start_idx - self.warmup_period
            train_slice_end = start_idx + training_period_bars
            
            test_slice_start = train_slice_end - self.warmup_period
            test_slice_end = min(test_slice_start + self.warmup_period + testing_period_bars, n_total_bars)

            if test_slice_start >= test_slice_end:
                break

            training_data = full_data.iloc[train_slice_start:train_slice_end]
            testing_data = full_data.iloc[test_slice_start:test_slice_end]
            
            logger.info(f"WFO Step: Training from {training_data.index[self.warmup_period]} to {training_data.index[-1]}")
            
            step_optimizer = StrategyOptimizer(
                strategy_class=self.strategy_class,
                data_feed_df=training_data,
                param_grid=self.param_grid,
                optimization_metric=self.optimization_metric,
                initial_capital=self.initial_capital,
                commission_bps=self.commission_bps,
                slippage_bps=self.slippage_bps,
                slippage_model_type=self.slippage_model_type, # Pass it down
                symbol=self.symbol,
                timeframe=self.timeframe,
                strategy_init_kwargs=self.strategy_init_kwargs # Pass it down
            )
            
            best_params_for_step, train_metric_value, _ = step_optimizer.grid_search()

            if best_params_for_step is not None:
                logger.info(f"WFO Step: Best params from training: {best_params_for_step} (Metric: {train_metric_value:.4f})")
                logger.info(f"WFO Step: Testing from {testing_data.index[self.warmup_period]} to {testing_data.index[-1]}")
                
                test_metrics = self._run_backtest_for_params(best_params_for_step, testing_data)
                
                if test_metrics:
                    result_entry = {
                        'train_start_date': training_data.index[self.warmup_period],
                        'train_end_date': training_data.index[-1],
                        'test_start_date': testing_data.index[self.warmup_period],
                        'test_end_date': testing_data.index[-1],
                        'best_params': best_params_for_step,
                        f'train_{self.optimization_metric}': train_metric_value,
                    }
                    result_entry.update({f'test_{k}': v for k, v in test_metrics.items()})
                    results_over_time.append(result_entry)
                    logger.info(f"WFO Step: Test period {self.optimization_metric}: {test_metrics.get(self.optimization_metric):.4f}")
                else:
                    logger.warning("WFO Step: Backtest on testing data failed or produced no metrics.")
            else:
                logger.warning("WFO Step: No best parameters found in training phase. Skipping test for this step.")

            start_idx += step_size_bars

        logger.info("Walk-Forward Optimization completed.")
        return results_over_time