# kamikaze_komodo/backtesting_engine/optimizer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
import itertools
import optuna # For more advanced optimization like TPE

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.config.settings import settings as app_settings # Renamed to avoid conflict
from kamikaze_komodo.risk_control_module.position_sizer import BasePositionSizer, FixedFractionalPositionSizer, ATRBasedPositionSizer # Add more as needed
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, PercentageStopManager, ATRStopManager # Add more as needed
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class StrategyOptimizer:
    """
    Optimizes strategy parameters using various methods like Grid Search or Optuna.
    Supports walk-forward optimization.
    """
    def __init__(
        self,
        strategy_class: type, # The class of the strategy to optimize (e.g., EWMACStrategy)
        data_feed_df: pd.DataFrame,
        param_grid: Dict[str, List[Any]], # e.g., {'short_window': [10, 12, 15], 'long_window': [20, 26, 30]}
        optimization_metric: str = 'total_net_profit', # Metric to optimize (from PerformanceAnalyzer)
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        position_sizer_class_name: Optional[str] = None, # e.g., "FixedFractionalPositionSizer"
        position_sizer_params: Optional[Dict[str, Any]] = None,
        stop_manager_class_name: Optional[str] = None, # e.g., "PercentageStopManager"
        stop_manager_params: Optional[Dict[str, Any]] = None,
        sentiment_data_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None, # Pass symbol if not in strategy params
        timeframe: Optional[str] = None # Pass timeframe if not in strategy params
    ):
        self.strategy_class = strategy_class
        self.data_feed_df = data_feed_df
        self.param_grid = param_grid
        self.optimization_metric = optimization_metric
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

        self.position_sizer_class_name = position_sizer_class_name
        self.position_sizer_params = position_sizer_params if position_sizer_params else {}
        self.stop_manager_class_name = stop_manager_class_name
        self.stop_manager_params = stop_manager_params if stop_manager_params else {}

        self.sentiment_data_df = sentiment_data_df
        self.symbol = symbol if symbol else (app_settings.default_symbol if app_settings else "OPTIMIZE_SYMBOL")
        self.timeframe = timeframe if timeframe else (app_settings.default_timeframe if app_settings else "OPTIMIZE_TF")

        logger.info(f"StrategyOptimizer initialized for {strategy_class.__name__} on {self.symbol} ({self.timeframe}). Optimizing for: {optimization_metric}")

    def _get_risk_module_instance(self, class_name_str: Optional[str], base_module, params_to_pass: Dict):
        if not class_name_str:
            return None
        try:
            ClassReference = getattr(base_module, class_name_str)
            return ClassReference(params=params_to_pass) # Assuming constructors accept 'params' dict
        except AttributeError:
            logger.error(f"Could not find class {class_name_str} in {base_module.__name__}")
        except Exception as e:
            logger.error(f"Error instantiating {class_name_str}: {e}")
        return None


    def _run_backtest_for_params(self, params_set: Dict[str, Any], current_data_feed: pd.DataFrame) -> float:
        """Runs a single backtest for a given set of parameters."""
        try:
            # Ensure strategy parameters are correctly passed, including any being optimized
            strategy_instance = self.strategy_class(
                symbol=self.symbol,
                timeframe=self.timeframe,
                params=params_set # Pass the current combination of parameters being tested
            )
    
            # Instantiate PositionSizer and StopManager if class names are provided
            # Merging fixed params with optimized params (if any sizer/stop params are in param_grid)
            combined_sizer_params = {**self.position_sizer_params, **params_set}
            combined_stop_params = {**self.stop_manager_params, **params_set}

            sizer_instance = self._get_risk_module_instance(self.position_sizer_class_name, pdm, combined_sizer_params) if self.position_sizer_class_name else FixedFractionalPositionSizer()
            stop_instance = self._get_risk_module_instance(self.stop_manager_class_name, smm, combined_stop_params) if self.stop_manager_class_name else PercentageStopManager()
    
            # Dynamically import position_sizer_module and stop_manager_module
            import kamikaze_komodo.risk_control_module.position_sizer as pdm
            import kamikaze_komodo.risk_control_module.stop_manager as smm
            from kamikaze_komodo.risk_control_module.volatility_band_stop_manager import VolatilityBandStopManager # if used by name

            if self.position_sizer_class_name:
                sizer_instance = self._get_risk_module_instance(self.position_sizer_class_name, pdm, combined_sizer_params)
            else: # Default sizer if none specified for optimization run
                sizer_instance = FixedFractionalPositionSizer(params=combined_sizer_params)


            if self.stop_manager_class_name:
                if self.stop_manager_class_name == "VolatilityBandStopManager":
                    # VolatilityBandStopManager is in a separate file
                    from kamikaze_komodo.risk_control_module import volatility_band_stop_manager as vbsm
                    stop_instance = self._get_risk_module_instance(self.stop_manager_class_name, vbsm, combined_stop_params)
                else:
                    stop_instance = self._get_risk_module_instance(self.stop_manager_class_name, smm, combined_stop_params)
            else: # Default stop manager
                stop_instance = PercentageStopManager(params=combined_stop_params)


            engine = BacktestingEngine(
                data_feed_df=current_data_feed,
                strategy=strategy_instance,
                initial_capital=self.initial_capital,
                commission_bps=self.commission_bps,
                slippage_bps=self.slippage_bps,
                position_sizer=sizer_instance,
                stop_manager=stop_instance,
                sentiment_data_df=self.sentiment_data_df # Assumed to cover the current_data_feed period
            )
            trades_log, final_portfolio, equity_curve = engine.run()
    
            # Metrics calculation
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

            # For Optuna, which maximizes by default. If metric is drawdown, we want to minimize, so return negative.
            # However, Optuna's direction is set in create_study. So, return the actual metric value.
            return float(metric_value)

        except Exception as e:
            logger.error(f"Error during backtest for params {params_set}: {e}", exc_info=True)
            return -float('inf') if self.optimization_metric != 'max_drawdown_pct' else float('inf')


    def grid_search(self) -> Tuple[Optional[Dict[str, Any]], float, pd.DataFrame]:
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
            metric_value = self._run_backtest_for_params(current_params, self.data_feed_df)
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

        def objective(trial: optuna.trial.Trial) -> float:
            params_set = {}
            for param_name, values in self.param_grid.items():
                if not values: # Skip if param list is empty
                    logger.warning(f"Parameter '{param_name}' in grid has empty values. Skipping for Optuna.")
                    continue
                if isinstance(values[0], bool): # Categorical for bool
                    params_set[param_name] = trial.suggest_categorical(param_name, [True, False])
                elif isinstance(values[0], int) and len(values) >= 2: # Treat as int range [min, max, step(optional)]
                    step = values[2] if len(values) > 2 else 1
                    params_set[param_name] = trial.suggest_int(param_name, values[0], values[1], step=step)
                elif isinstance(values[0], float) and len(values) >= 2: # Treat as float range [min, max, step(optional for log)]
                    # Optuna's suggest_float doesn't have a direct step like suggest_int.
                    # If discrete floats are needed, use suggest_categorical or round after suggestion.
                    params_set[param_name] = trial.suggest_float(param_name, values[0], values[1])
                elif isinstance(values, list): # Categorical for other types
                    params_set[param_name] = trial.suggest_categorical(param_name, values)
                else:
                    logger.warning(f"Parameter '{param_name}' in grid has unsupported format for Optuna. Values: {values}. Skipping.")
    
            metric_value = self._run_backtest_for_params(params_set, self.data_feed_df)
            return metric_value

        logger.info(f"Starting Optuna optimization with {n_trials} trials. Optimizing for {self.optimization_metric} ({direction}).")
        study.optimize(objective, n_trials=n_trials, timeout=None) # Add timeout if needed

        best_params = None
        best_metric_value = study.best_value if study.best_trial else (-float('inf') if direction == 'maximize' else float('inf'))
        if study.best_trial:
            best_params = study.best_params
            logger.info(f"Optuna optimization completed. Best params: {best_params}, Best {self.optimization_metric}: {best_metric_value:.4f}")
        else:
            logger.warning("Optuna optimization completed but no best trial found (all trials might have failed or yielded non-comparable results).")
        return best_params, best_metric_value, study


    def walk_forward_optimization(
        self,
        training_period_bars: int,
        testing_period_bars: int,
        step_size_bars: int,
        optimization_method: str = 'grid_search',
        optuna_n_trials_per_step: int = 50
    ) -> List[Dict[str, Any]]:
        if not isinstance(self.data_feed_df.index, pd.DatetimeIndex):
            raise ValueError("data_feed_df must have a DatetimeIndex for walk-forward optimization.")

        full_data = self.data_feed_df.copy()
        n_total_bars = len(full_data)

        if training_period_bars + testing_period_bars > n_total_bars:
            logger.error("Not enough data for even one training/testing period in WFO.")
            return []

        results_over_time = []
        start_idx = 0

        logger.info(f"Starting Walk-Forward Optimization. Train: {training_period_bars}, Test: {testing_period_bars}, Step: {step_size_bars}")

        original_full_data_feed = self.data_feed_df # Store the original full data feed reference

        while start_idx + training_period_bars <= n_total_bars: # Ensure training period is within bounds
            train_end_idx = start_idx + training_period_bars
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + testing_period_bars, n_total_bars) # Don't go beyond total bars

            if test_start_idx >= test_end_idx : # No testing data left
                break

            training_data = full_data.iloc[start_idx:train_end_idx]
            testing_data = full_data.iloc[test_start_idx:test_end_idx]
    
            logger.info(f"WFO Step: Training from {training_data.index[0]} to {training_data.index[-1]} ({len(training_data)} bars)")
    
            self.data_feed_df = training_data # Temporarily set data_feed_df for optimization methods

            best_params_for_step: Optional[Dict[str, Any]] = None
            train_metric_value = -float('inf')

            if optimization_method == 'grid_search':
                best_params_for_step, train_metric_value, _ = self.grid_search()
            elif optimization_method == 'optuna':
                study_name_wfo = f"{self.strategy_class.__name__}_WFO_Step_{start_idx}"
                best_params_for_step, train_metric_value, _ = self.optuna_optimize(n_trials=optuna_n_trials_per_step, study_name=study_name_wfo)
            else:
                logger.error(f"Unsupported optimization_method: {optimization_method}")
                self.data_feed_df = original_full_data_feed # Restore
                return results_over_time

            if best_params_for_step:
                logger.info(f"WFO Step: Best params from training: {best_params_for_step} (Metric: {train_metric_value:.4f})")
                logger.info(f"WFO Step: Testing from {testing_data.index[0]} to {testing_data.index[-1]} ({len(testing_data)} bars) with best params.")
        
                test_metric_value = self._run_backtest_for_params(best_params_for_step, testing_data)
        
                results_over_time.append({
                    'train_start_date': training_data.index[0],
                    'train_end_date': training_data.index[-1],
                    'test_start_date': testing_data.index[0],
                    'test_end_date': testing_data.index[-1],
                    'best_params': best_params_for_step,
                    f'train_{self.optimization_metric}': train_metric_value,
                    f'test_{self.optimization_metric}': test_metric_value
                })
                logger.info(f"WFO Step: Test period {self.optimization_metric}: {test_metric_value:.4f}")
            else:
                logger.warning("WFO Step: No best parameters found in training phase. Skipping test for this step.")

            start_idx += step_size_bars

        self.data_feed_df = original_full_data_feed # Restore original full data feed
        logger.info("Walk-Forward Optimization completed.")
        return results_over_time