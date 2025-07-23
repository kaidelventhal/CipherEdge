import pandas as pd
from datetime import datetime
from typing import Optional, List
import os

from .data_fetcher import DataFetcher
from .database_manager import DatabaseManager
from ..core.models import BarData
from ..app_logger import get_logger
from ..config.settings import settings

logger = get_logger(__name__)

class DataHandler:
    """
    Handles fetching, preparation, and enrichment of market data for backtesting and live trading.
    Encapsulates logic for merging different data sources like funding rates and sentiment,
    preventing data collision issues in the main application logic.
    """

    def __init__(self):
        self.fetcher = DataFetcher()
        self.db_manager = DatabaseManager()

    async def get_prepared_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        needs_funding_rate: bool = False,
        needs_sentiment: bool = False
    ) -> pd.DataFrame:
        """
        Fetches, merges, and prepares data, returning a clean DataFrame with a DatetimeIndex.
        This is the primary method to get data for any backtest or analysis.
        """
        bars = self.db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
        if not bars:
            logger.info(f"No data in DB for {symbol}/{timeframe}, fetching from exchange...")
            bars = await self.fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
            if bars:
                self.db_manager.store_bar_data(bars)
        
        if not bars:
            logger.error(f"Could not retrieve or fetch any data for {symbol}/{timeframe}.")
            return pd.DataFrame()

        data_df = pd.DataFrame([bar.model_dump() for bar in bars])
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df.sort_values('timestamp', inplace=True)

        if needs_funding_rate:
            if 'funding_rate' in data_df.columns:
                data_df = data_df.drop(columns=['funding_rate'])
            
            logger.info(f"Attempting to retrieve funding rates for {symbol} from cache...")
            funding_rates_raw = self.db_manager.retrieve_funding_rates(symbol, start_date, end_date)
            
            if not funding_rates_raw:
                logger.info(f"Funding rates for {symbol} not in cache for the requested period. Fetching from exchange...")
                funding_rates_raw = await self.fetcher.fetch_funding_rate_history(symbol, since=start_date)
                if funding_rates_raw:
                    logger.info(f"Caching {len(funding_rates_raw)} new funding rate entries.")
                    self.db_manager.store_funding_rates(funding_rates_raw)

            if funding_rates_raw:
                fr_df = pd.DataFrame(funding_rates_raw)
                fr_df['timestamp'] = pd.to_datetime(fr_df['timestamp'], unit='ms', utc=True)
                fr_df = fr_df[['timestamp', 'fundingRate']].sort_values('timestamp')
                
                data_df = pd.merge_asof(
                    left=data_df,
                    right=fr_df,
                    on='timestamp',
                    direction='backward'
                )
                data_df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
                logger.info("Funding rates merged.")
            else:
                logger.warning(f"Could not fetch or retrieve funding rates for {symbol}. Column will be filled with 0.0.")
                data_df['funding_rate'] = 0.0
        
        if needs_sentiment:
            if 'sentiment_score' in data_df.columns:
                data_df = data_df.drop(columns=['sentiment_score'])
            
            if settings and settings.simulated_sentiment_data_path and os.path.exists(settings.simulated_sentiment_data_path):
                sentiment_df = pd.read_csv(settings.simulated_sentiment_data_path, parse_dates=['timestamp'])
                
                data_df = pd.merge_asof(
                    left=data_df.sort_values('timestamp'),
                    right=sentiment_df[['timestamp', 'sentiment_score']].sort_values('timestamp'),
                    on='timestamp',
                    direction='backward'
                )
                logger.info("Sentiment data merged.")
            else:
                logger.warning("Simulated sentiment data path not found. 'sentiment_score' column will be filled with 0.0.")
                data_df['sentiment_score'] = 0.0
        
        data_df.set_index('timestamp', inplace=True)

        if 'funding_rate' in data_df.columns:
            data_df['funding_rate'] = data_df['funding_rate'].ffill().fillna(0.0)
        
        if 'sentiment_score' in data_df.columns:
            data_df['sentiment_score'] = data_df['sentiment_score'].ffill().fillna(0.0)

        core_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in core_cols:
            if col not in data_df.columns:
                logger.error(f"Core column '{col}' is missing from the final DataFrame.")
                return pd.DataFrame()
        
        logger.info(f"Prepared data for {symbol}/{timeframe} with {len(data_df)} bars.")
        return data_df

    async def close(self):
        """Closes underlying connections."""
        await self.fetcher.close()
        self.db_manager.close()