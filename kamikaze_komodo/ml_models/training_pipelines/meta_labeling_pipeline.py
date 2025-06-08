# FILE: kamikaze_komodo/ml_models/training_pipelines/meta_labeling_pipeline.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier # Example meta-model
import joblib
import os
from typing import Tuple, Dict, Any, Optional

from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from datetime import datetime, timedelta, timezone

logger = get_logger(__name__)

def get_triple_barrier_events(
    close_prices: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float],
    target_vol: pd.Series,
    min_ret: float,
    num_threads: int,
    vertical_barrier_times: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generates labels for the triple-barrier method.
    This is a simplified adaptation of De Prado's method.
    """
    if vertical_barrier_times is None:
        raise ValueError("vertical_barrier_times must be provided.")
        
    out = pd.DataFrame(index=t_events)
    out['t1'] = vertical_barrier_times
    out['trgt'] = target_vol

    for loc, t1 in vertical_barrier_times.items():
        df0 = close_prices[loc:t1]
        df0 = (df0 / close_prices[loc] - 1) * (1 if side is None or side.loc[loc] == 1 else -1)
        
        out.loc[loc, 'sl'] = df0[df0 < -pt_sl[1]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt_sl[0]].index.min()

    # Determine first touch
    df1 = out.dropna(subset=['sl', 'pt'])
    df1 = df1.copy()
    df1['t1'] = pd.to_datetime(df1['t1'])
    df1['sl'] = pd.to_datetime(df1['sl'])
    df1['pt'] = pd.to_datetime(df1['pt'])
    
    first_touch = df1[['t1', 'sl', 'pt']].min(axis=1)
    
    events = pd.DataFrame(index=t_events)
    events['t1'] = first_touch
    events['trgt'] = out['trgt']
    if side is not None:
        events['side'] = side
    
    return events

def get_bins(events: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
    """
    Computes labels {0, 1} from events.
    1 = take profit was hit, 0 = stop loss or time barrier was hit.
    """
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close_prices.reindex(px, method='bfill')
    
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    
    if 'side' in events:
        out['ret'] *= events_['side']
    
    # Simple binary outcome: 1 for positive return, 0 for non-positive
    out['bin'] = np.sign(out['ret'])
    out.loc[out['bin'] <= 0, 'bin'] = 0
    
    return out


class MetaLabelingTrainingPipeline:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_params = settings.get_strategy_params('MLForecaster_Strategy')
        
        # Paths for saving models
        base_path = os.path.join(PROJECT_ROOT, "ml_models/trained_models")
        os.makedirs(base_path, exist_ok=True)
        self.primary_model_path = os.path.join(base_path, f"primary_{symbol.replace('/', '_')}.joblib")
        self.meta_model_path = os.path.join(base_path, f"meta_{symbol.replace('/', '_')}.joblib")

    async def fetch_data(self, days: int = 1000) -> pd.DataFrame:
        db_manager = DatabaseManager()
        data_fetcher = DataFetcher()
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        bars = db_manager.retrieve_bar_data(self.symbol, self.timeframe, start_date=start_date)
        if not bars or len(bars) < 200:
            bars = await data_fetcher.fetch_historical_data_for_period(self.symbol, self.timeframe, start_date)
            if bars: db_manager.store_bar_data(bars)
        await data_fetcher.close()
        db_manager.close()
        
        if not bars: return pd.DataFrame()
        
        df = pd.DataFrame([b.model_dump() for b in bars])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for lag in [1, 3, 5, 10]:
            df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        df['volatility_20'] = df['log_return_lag_1'].rolling(window=20).std()
        return df.dropna()

    async def run_training(self):
        logger.info(f"Starting meta-labeling training pipeline for {self.symbol}...")
        
        # 1. Fetch and prepare data
        data = await self.fetch_data()
        if data.empty:
            logger.error("No data available for training.")
            return
        
        features_df = self.create_features(data)
        X = features_df
        
        # 2. Define barriers and generate labels
        daily_vol = X['volatility_20'] # Simplified volatility measure
        pt_sl_multipliers = (1.0, 1.0) # Symmetric 1:1 risk-reward
        vertical_barrier_bars = 10 # 10 bars hold time
        
        vertical_barrier_times = pd.Series(X.index, index=X.index).shift(-vertical_barrier_bars)
        
        # Primary model will predict side, let's assume it predicts return > 0 or < 0
        # For simplicity, we'll generate side predictions based on a simple momentum indicator
        side = pd.Series(np.nan, index=X.index)
        side[X['log_return_lag_1'] > 0] = 1
        side[X['log_return_lag_1'] < 0] = -1
        side = side.ffill().dropna()
        
        # Align indexes
        aligned_idx = side.index.intersection(vertical_barrier_times.index).intersection(daily_vol.index)
        
        events = get_triple_barrier_events(
            close_prices=X['close'],
            t_events=aligned_idx,
            pt_sl=pt_sl_multipliers,
            target_vol=daily_vol.loc[aligned_idx],
            min_ret=0,
            num_threads=1,
            vertical_barrier_times=vertical_barrier_times.loc[aligned_idx],
            side=side.loc[aligned_idx]
        )
        
        labels = get_bins(events, X['close'])
        
        # Align features with labels
        X = X.loc[labels.index]
        
        # 3. Train Primary Model (predicts side)
        # Using a simple LightGBM model to predict the sign of the next return
        y_primary = np.sign(X['close'].pct_change(1).shift(-1).dropna())
        X_primary = X.loc[y_primary.index]
        
        primary_model = lgb.LGBMClassifier()
        primary_model.fit(X_primary, y_primary)
        joblib.dump(primary_model, self.primary_model_path)
        logger.info(f"Primary model trained and saved to {self.primary_model_path}")
        
        # 4. Get Primary Model Predictions (as feature for meta model)
        primary_predictions_proba = primary_model.predict_proba(X)[:, 1] # Probability of class 1 (up)
        
        # 5. Train Meta Model (predicts success based on primary model's confidence)
        X_meta = pd.DataFrame({'primary_pred_proba': primary_predictions_proba}, index=X.index)
        y_meta = labels['bin']
        
        # Using a calibrated classifier for the meta model
        meta_model_base = RandomForestClassifier(n_estimators=100, max_depth=4)
        meta_model = CalibratedClassifierCV(meta_model_base, method='isotonic', cv=5)
        meta_model.fit(X_meta, y_meta)
        joblib.dump(meta_model, self.meta_model_path)
        logger.info(f"Meta model trained and saved to {self.meta_model_path}")

        logger.info("Meta-labeling training pipeline finished.")