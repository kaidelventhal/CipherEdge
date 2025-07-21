# FILE: kamikaze_komodo/backtesting_engine/optimizer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Type
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.config.settings import settings as app_settings, PROJECT_ROOT
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.ml_models.regime_detection.kmeans_regime_model import KMeansRegimeModel

# Import all strategy and component classes for instantiation by name
from kamikaze_komodo.strategy_framework.strategy_manager import STRATEGY_REGISTRY
from kamikaze_komodo.backtesting_engine.engine import POSITION_SIZER_REGISTRY, STOP_MANAGER_REGISTRY
from kamikaze_komodo.core.models import Trade

logger = get_logger(__name__)

# Define which strategies belong to which regime type
TREND_STRATEGIES = ["EWMACStrategy", "EhlersInstantaneousTrendlineStrategy", "VolatilitySqueezeBreakoutStrategy", "BollingerBandBreakoutStrategy"]
MEAN_REVERSION_STRATEGIES = ["BollingerBandMeanReversionStrategy"]

def run_single_trial(trial_def: Dict[str, Any], data_feed: pd.DataFrame, initial_capital: float, commission_bps: float, slippage_bps: float) -> Optional[Tuple[int, List[Trade], Dict, pd.DataFrame]]:
    """
    Worker function to run a single backtest trial. Designed to be called by a ProcessPoolExecutor.
    """
    trial_id = trial_def.get('id', -1)
    try:
        engine = BacktestingEngine(
            data_feed_df=data_feed,
            strategy_name=trial_def['strategy_name'],
            strategy_params=trial_def['strategy_params'],
            position_sizer_name=trial_def['position_sizer_name'],
            stop_manager_name=trial_def['stop_manager_name'],
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            symbol=trial_def['symbol'],
            timeframe=trial_def['timeframe']
        )
        trades_log, final_portfolio, equity_curve = engine.run()
        return trial_id, trades_log, final_portfolio, equity_curve
    except Exception as e:
        logger.error(f"[Worker PID: {os.getpid()}] FATAL ERROR running trial {trial_id} for {trial_def['symbol']}|{trial_def['strategy_name']}: {e}", exc_info=True)
        return None


class StrategyOptimizer:
    """
    Phase 3: Enhanced to run large-scale discovery of (symbol, strategy, risk, stop) combinations
    using parallel processing to improve performance.
    """
    def __init__(
        self,
        data_feeds: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0
    ):
        self.data_feeds = data_feeds
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        logger.info(f"StrategyOptimizer initialized for Phase 3 discovery.")

    def _get_dominant_regime(self, symbol: str) -> Optional[int]:
        """Identifies the dominant market regime over the backtest period."""
        if symbol not in self.data_feeds:
            return None
         
        df = self.data_feeds[symbol]
        timeframe = app_settings.default_timeframe
        kmeans_config = app_settings.get_strategy_params("KMeans_Regime_Model")
        kmeans_path = os.path.join(PROJECT_ROOT, kmeans_config.get('modelsavepath', 'ml_models/trained_models'), f"kmeans_regime_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
         
        if not os.path.exists(kmeans_path):
            logger.warning(f"KMeans model not found at {kmeans_path}. Cannot perform regime-aware filtering for {symbol}.")
            return None
             
        kmeans_model = KMeansRegimeModel(model_path=kmeans_path, params=kmeans_config)
        if not kmeans_model.model:
            return None

        df['market_regime'] = kmeans_model.predict_regimes_for_dataframe(df)
        df['market_regime'] = df['market_regime'].ffill()
        dominant_regime = df['market_regime'].mode()[0] if not df['market_regime'].dropna().empty else None
         
        if dominant_regime is not None:
            regime_name = kmeans_model.regime_labels.get(int(dominant_regime), f"Regime {dominant_regime}")
            logger.info(f"Dominant regime for {symbol} is: {regime_name} ({dominant_regime})")
            return int(dominant_regime)
        return None

    def run_phase3_discovery(self) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
        """
        Orchestrates the entire Phase 3 discovery process using a process pool for parallel backtesting.
        """
        trials = self._generate_trials()
        if not trials:
            logger.error("No trials were generated. Aborting discovery.")
            return pd.DataFrame(), {}

        all_metrics = []
        equity_curves = {}
         
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures_to_trials = {
                executor.submit(
                    run_single_trial, 
                    trial, 
                    self.data_feeds[trial['symbol']],
                    self.initial_capital,
                    self.commission_bps,
                    self.slippage_bps
                ): trial for trial in trials
            }

            pbar = tqdm(as_completed(futures_to_trials), total=len(trials), desc="Running Backtest Trials")
            for future in pbar:
                trial_def = futures_to_trials[future]
                pbar.set_description(f"Processing {trial_def['symbol']} | {trial_def['strategy_name']}")
                 
                result = future.result()
                if result:
                    trial_id, trades_log, final_portfolio, equity_curve = result
                     
                    analyzer = PerformanceAnalyzer(
                        trades=trades_log,
                        initial_capital=self.initial_capital,
                        final_capital=final_portfolio['final_portfolio_value'],
                        equity_curve_df=equity_curve
                    )
                    metrics = analyzer.calculate_metrics()
                    
                    metrics.update(trial_def)
                    all_metrics.append(metrics)
                    equity_curves[trial_id] = equity_curve['total_value_usd']

        if not all_metrics:
            logger.error("No metrics were generated from the backtest run.")
            return pd.DataFrame(), {}
             
        results_df = pd.DataFrame(all_metrics).set_index('id')
        sharpe_series = results_df['sharpe_ratio'].dropna()
        num_bars = len(next(iter(self.data_feeds.values())))

        results_df['deflated_sharpe_ratio'] = results_df['sharpe_ratio'].apply(
            lambda x: PerformanceAnalyzer.calculate_deflated_sharpe_ratio(
                sharpe_ratios_series=sharpe_series,
                num_bars_in_backtest=num_bars,
                selected_sharpe=x
            ) if pd.notna(x) else np.nan
        )

        results_for_csv = results_df.copy().reset_index().rename(columns={'id': 'trial_id'})
        results_for_csv['id'] = results_for_csv['trial_id']

        csv_columns = [
            'trial_id', 'initial_capital', 'final_capital', 'total_net_profit',
            'total_return_pct', 'total_trades', 'winning_trades', 'losing_trades',
            'breakeven_trades', 'win_rate_pct', 'loss_rate_pct',
            'average_pnl_per_trade', 'average_win_pnl', 'average_loss_pnl',
            'profit_factor', 'max_drawdown_pct', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'total_fees_paid', 'average_holding_period_hours',
            'longest_win_streak', 'longest_loss_streak', 'time_in_market_pct',
            'turnover_rate', 'id', 'symbol', 'timeframe', 'strategy_name',
            'position_sizer_name', 'stop_manager_name', 'strategy_params',
            'deflated_sharpe_ratio'
        ]

        for col in csv_columns:
            if col not in results_for_csv.columns:
                results_for_csv[col] = pd.NA

        results_for_csv[csv_columns].to_csv("trial_results.csv", index=False)
        logger.info("Phase 3 discovery finished. Results saved to 'trial_results.csv'.")
         
        return results_df, equity_curves

    def _generate_trials(self) -> List[Dict[str, Any]]:
        """Generates a list of all trial definitions based on Phase 3 settings and parameter grids."""
        symbols = app_settings.PHASE3_SYMBOLS
        strategy_names = app_settings.PHASE3_STRATEGIES
        sizer_names = app_settings.PHASE3_RISK_MODULES
        stop_names = app_settings.PHASE3_STOP_MANAGERS
        param_grids = app_settings.PHASE3_GRID_SEARCH
         
        trials = []
        trial_id_counter = 0

        for symbol in symbols:
            dominant_regime = self._get_dominant_regime(symbol)
             
            for strat_name in strategy_names:
                #if dominant_regime is not None:
                #    is_trending_regime = dominant_regime == 1 
                #    if strat_name in TREND_STRATEGIES and not is_trending_regime: continue
                #    if strat_name in MEAN_REVERSION_STRATEGIES and is_trending_regime: continue

                base_params = app_settings.get_strategy_params(strat_name)
                grid_key = strat_name.split('_')[0]
                grid = param_grids.get(grid_key, {})
                 
                param_combinations = []
                if not grid:
                    param_combinations.append(base_params)
                else:
                    keys, values = zip(*grid.items())
                    for v in itertools.product(*values):
                        combo = dict(zip(keys, v))
                        full_params = {**base_params, **combo}
                        param_combinations.append(full_params)
                 
                for params in param_combinations:
                    for sizer_name in sizer_names:
                        for stop_name in stop_names:
                            trial_def = {
                                'id': trial_id_counter,
                                'symbol': symbol,
                                'timeframe': app_settings.default_timeframe,
                                'strategy_name': strat_name,
                                'position_sizer_name': sizer_name,
                                'stop_manager_name': stop_name,
                                'strategy_params': params,
                            }
                            trials.append(trial_def)
                            trial_id_counter += 1
         
        logger.info(f"Generated {len(trials)} trials for discovery with parameter grid search.")
        return trials