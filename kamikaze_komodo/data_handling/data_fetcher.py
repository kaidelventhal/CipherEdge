# kamikaze_komodo/data_handling/data_fetcher.py
import ccxt.async_support as ccxt # Use async version for future compatibility
import asyncio
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, timezone
# Assuming these are correctly located relative to this file for your project structure
from kamikaze_komodo.core.models import BarData, FundingRate
from kamikaze_komodo.core.utils import ohlcv_to_bardata
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings # Ensure settings is loaded globally

logger = get_logger(__name__)

class DataFetcher:
    """
    Fetches historical and real-time market data using CCXT.
    """
    def __init__(self):
        if not settings:
            logger.critical("Settings not loaded. DataFetcher cannot be initialized.")
            raise ValueError("Settings not loaded. Ensure config files are present and correct.")

        self.exchange_id = settings.exchange_id_to_use
        exchange_class = getattr(ccxt, self.exchange_id, None)
        
        if not exchange_class:
            logger.error(f"Exchange '{self.exchange_id}' is not supported by CCXT.")
            raise ValueError(f"Exchange '{self.exchange_id}' is not supported by CCXT.")

        config = {
            'apiKey': settings.kraken_api_key,
            'secret': settings.kraken_secret_key,
            'enableRateLimit': True,
        }
        
        self.exchange = exchange_class(config)
        logger.info(f"Instantiated CCXT exchange class: {self.exchange_id}")

        if settings.kraken_testnet:
            if hasattr(self.exchange, 'set_sandbox_mode') and callable(self.exchange.set_sandbox_mode):
                try:
                    self.exchange.set_sandbox_mode(True)
                    logger.info(f"CCXT sandbox mode successfully enabled for {self.exchange_id}.")
                except ccxt.NotSupported:
                    logger.warning(f"{self.exchange_id} does not support unified set_sandbox_mode via method.")
                except Exception as e:
                    logger.error(f"An error occurred while trying to set sandbox mode for {self.exchange_id}: {e}", exc_info=True)
            else:
                logger.warning(f"{self.exchange_id} CCXT class does not have a 'set_sandbox_mode' method.")
        
        # self.exchange.verbose = True # Uncomment for debugging API calls
        logger.info(f"Initialized DataFetcher for '{self.exchange_id}'. Configured Testnet (Sandbox) from settings: {settings.kraken_testnet}")

    async def fetch_historical_ohlcv(self, symbol: str, timeframe: str, since: Optional[datetime] = None, limit: Optional[int] = None, params: Optional[dict] = None) -> List[BarData]:
        if not self.exchange.has['fetchOHLCV']:
            logger.error(f"{self.exchange_id} does not support fetchOHLCV.")
            return []

        since_timestamp_ms = int(since.timestamp() * 1000) if since else None

        ohlcv_data_list: List[BarData] = []
        try:
            logger.info(f"Fetching historical OHLCV for {symbol} ({timeframe}) since {since} with limit {limit}")
            raw_ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since_timestamp_ms, limit, params or {})
            
            if raw_ohlcv:
                for entry in raw_ohlcv:
                    try:
                        bar = ohlcv_to_bardata(entry, symbol, timeframe)
                        ohlcv_data_list.append(bar)
                    except ValueError as e_bar:
                        logger.warning(f"Skipping invalid OHLCV entry for {symbol} ({timeframe}): {entry}. Error: {e_bar}")
                logger.info(f"Successfully fetched {len(ohlcv_data_list)} candles for {symbol} ({timeframe}).")
        except Exception as e:
            logger.error(f"An unexpected error occurred in fetch_historical_ohlcv for {symbol}: {e}", exc_info=True)
        return ohlcv_data_list

    async def fetch_historical_data_for_period(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime = datetime.now(timezone.utc)) -> List[BarData]:
        all_bars: List[BarData] = []
        current_start_date = start_date
        
        timeframe_duration_seconds = self.exchange.parse_timeframe(timeframe)
        if timeframe_duration_seconds is None:
            logger.error(f"Could not parse timeframe: {timeframe} for {self.exchange_id}.")
            return []

        logger.info(f"Fetching historical period data for {symbol} ({timeframe}) from {start_date} to {end_date}")

        while current_start_date < end_date:
            limit_per_call = 500
            bars = await self.fetch_historical_ohlcv(symbol, timeframe, since=current_start_date, limit=limit_per_call)
            
            if not bars: break 
            
            relevant_bars = [b for b in bars if b.timestamp < end_date]
            if not relevant_bars: break

            all_bars.extend(relevant_bars)
            last_fetched_timestamp = relevant_bars[-1].timestamp
            current_start_date = last_fetched_timestamp + timedelta(seconds=timeframe_duration_seconds)
            
            if current_start_date >= end_date: break
            
            await asyncio.sleep(self.exchange.rateLimit / 1000.0 if isinstance(self.exchange.rateLimit, (int, float)) and self.exchange.rateLimit > 0 else 0.2)

        if all_bars:
            unique_bars_dict = {bar.timestamp: bar for bar in all_bars}
            all_bars = sorted(list(unique_bars_dict.values()), key=lambda b: b.timestamp)
            logger.info(f"Total unique historical bars fetched for {symbol} ({timeframe}) in period: {len(all_bars)}")
        
        return all_bars

    async def fetch_funding_rate_history(self, symbol: str, since: Optional[datetime] = None, limit: Optional[int] = 100) -> List[FundingRate]:
        """Fetches historical funding rates for a perpetual futures symbol."""
        if not self.exchange.has['fetchFundingRateHistory']:
            logger.warning(f"Exchange {self.exchange_id} does not support fetchFundingRateHistory.")
            return []
        
        since_ms = int(since.timestamp() * 1000) if since else None
        logger.info(f"Fetching funding rate history for {symbol} since {since}.")
        try:
            raw_rates = await self.exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
            funding_rates = [
                FundingRate(
                    symbol=rate['symbol'],
                    timestamp=datetime.fromtimestamp(rate['timestamp'] / 1000, tz=timezone.utc),
                    funding_rate=rate['fundingRate'],
                    mark_price=rate.get('markPrice')
                ) for rate in raw_rates
            ]
            logger.info(f"Fetched {len(funding_rates)} funding rate entries for {symbol}.")
            return funding_rates
        except Exception as e:
            logger.error(f"Error fetching funding rate history for {symbol}: {e}", exc_info=True)
            return []


    async def close(self):
        """Closes the CCXT exchange connection."""
        try:
            if hasattr(self.exchange, 'close') and callable(self.exchange.close):
                await self.exchange.close()
            logger.info(f"CCXT exchange connection for {self.exchange_id} closed.")
        except Exception as e:
            logger.error(f"Error closing CCXT exchange connection for {self.exchange_id}: {e}", exc_info=True)