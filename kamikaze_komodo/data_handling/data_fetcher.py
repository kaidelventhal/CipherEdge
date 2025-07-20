# kamikaze_komodo/data_handling/data_fetcher.py
import ccxt.async_support as ccxt # Use async version for future compatibility
import asyncio
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta, timezone
# Assuming these are correctly located relative to this file for your project structure
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.core.utils import ohlcv_to_bardata
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings # Ensure settings is loaded globally

logger = get_logger(__name__)

class DataFetcher:
    """
    Fetches historical and real-time market data using CCXT.
    Phase 6: Added fetch_historical_data_for_pair for pair trading strategies.
    """
    def __init__(self): # MODIFIED: No longer takes exchange_id as an argument
        if not settings:
            logger.critical("Settings not loaded. DataFetcher cannot be initialized.")
            raise ValueError("Settings not loaded. Ensure config files are present and correct.")

        self.exchange_id = settings.exchange_id_to_use # MODIFIED: Get from global settings
        exchange_class = getattr(ccxt, self.exchange_id, None)
        
        if not exchange_class:
            logger.error(f"Exchange '{self.exchange_id}' is not supported by CCXT.")
            raise ValueError(f"Exchange '{self.exchange_id}' is not supported by CCXT.")

        # API keys should be specific to the selected exchange_id 
        # (e.g., Kraken Spot keys for 'kraken', Kraken Futures Demo keys for 'krakenfutures')
        config = {
            'apiKey': settings.kraken_api_key, # This assumes kraken_api_key holds the relevant key
            'secret': settings.kraken_secret_key, # This assumes kraken_secret_key holds the relevant secret
            'enableRateLimit': True, # Recommended by CCXT
        }
        
        # Example: If your settings had distinct keys for different exchanges:
        # if self.exchange_id == 'krakenfutures':
        #     config['apiKey'] = settings.kraken_futures_api_key 
        #     config['secret'] = settings.kraken_futures_secret_key
        # elif self.exchange_id == 'kraken':
        #     config['apiKey'] = settings.kraken_spot_api_key
        #     config['secret'] = settings.kraken_spot_secret_key
        # For now, we use the general kraken_api_key/secret from settings.

        self.exchange = exchange_class(config)
        logger.info(f"Instantiated CCXT exchange class: {self.exchange_id}")

        if settings.kraken_testnet: # This flag now controls sandbox mode for the selected exchange
            if hasattr(self.exchange, 'set_sandbox_mode') and callable(self.exchange.set_sandbox_mode):
                try:
                    self.exchange.set_sandbox_mode(True)
                    logger.info(f"CCXT sandbox mode successfully enabled for {self.exchange_id}.")
                    # You can log the API URL to verify it changed, e.g.:
                    # logger.info(f"Using API URLs: {self.exchange.urls['api']}")
                except ccxt.NotSupported:
                    logger.warning(f"{self.exchange_id} does not support unified set_sandbox_mode via method. Testnet functionality might depend on specific API keys or default URLs for this exchange class.")
                except Exception as e:
                    logger.error(f"An error occurred while trying to set sandbox mode for {self.exchange_id}: {e}", exc_info=True)
            else:
                logger.warning(f"{self.exchange_id} CCXT class does not have a 'set_sandbox_mode' method. Testnet operation relies on correct API keys for the test environment and default URLs.")
        
        self.exchange.verbose = False # Set to True for debugging API calls
        logger.info(f"Initialized DataFetcher for '{self.exchange_id}'. Configured Testnet (Sandbox) from settings: {settings.kraken_testnet}")

    async def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None
    ) -> List[BarData]:
        if not self.exchange.has['fetchOHLCV']:
            logger.error(f"{self.exchange_id} does not support fetchOHLCV.")
            # await self.close() # Closing here might be premature if other operations are pending
            return []

        since_timestamp_ms = None
        if since:
            if since.tzinfo is None: 
                since = since.replace(tzinfo=timezone.utc)
            since_timestamp_ms = int(since.timestamp() * 1000)

        ohlcv_data_list: List[BarData] = []
        try:
            logger.info(f"Fetching historical OHLCV for {symbol} ({timeframe}) from exchange {self.exchange_id} since {since} with limit {limit}")
            raw_ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since_timestamp_ms, limit, params or {})            
            # More robust check for raw_ohlcv
            if raw_ohlcv is not None and isinstance(raw_ohlcv, list):
                if not raw_ohlcv: # Empty list
                    logger.info(f"No OHLCV data returned (empty list) for {symbol} ({timeframe}) from {self.exchange_id} with the given parameters.")
                else:
                    for entry in raw_ohlcv:
                        try:
                            bar = ohlcv_to_bardata(entry, symbol, timeframe)
                            ohlcv_data_list.append(bar)
                        except ValueError as e_bar:
                            logger.warning(f"Skipping invalid OHLCV entry for {symbol} ({timeframe}): {entry}. Error: {e_bar}")
                    logger.info(f"Successfully fetched {len(ohlcv_data_list)} candles for {symbol} ({timeframe}) from {self.exchange_id}.")
            elif raw_ohlcv is None:
                logger.info(f"No OHLCV data returned (got None) for {symbol} ({timeframe}) from {self.exchange_id} with the given parameters.")
            else: # It's something else, not None and not a list
                logger.warning(f"Unexpected data type received for OHLCV for {symbol} ({timeframe}) from {self.exchange_id}: {type(raw_ohlcv)}. Data: {str(raw_ohlcv)[:200]}")
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching OHLCV for {symbol} from {self.exchange_id}: {e}", exc_info=True)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching OHLCV for {symbol} from {self.exchange_id}: {e}", exc_info=True)
        except Exception as e: # Generic catch-all
            logger.error(f"An unexpected error occurred in fetch_historical_ohlcv for {symbol} from {self.exchange_id}: {e}", exc_info=True)
        
        return ohlcv_data_list

    async def fetch_historical_data_for_period(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime = datetime.now(timezone.utc)
    ) -> List[BarData]:
        all_bars: List[BarData] = []
        current_start_date = start_date
        
        try:
            timeframe_duration_seconds = self.exchange.parse_timeframe(timeframe)
        except Exception as e_tf:
            logger.error(f"Failed to parse timeframe '{timeframe}' using CCXT for {self.exchange_id}: {e_tf}")
            timeframe_duration_seconds = None # Fallback if parse_timeframe itself errors

        if timeframe_duration_seconds is None: # Still None after try-except
            logger.error(f"Could not parse timeframe: {timeframe} for {self.exchange_id}. Cannot paginate effectively. Attempting single fetch.")
            return await self.fetch_historical_ohlcv(symbol, timeframe, since=start_date, limit=1000) # Example limit

        logger.info(f"Fetching historical period data for {symbol} ({timeframe}) on {self.exchange_id} from {start_date} to {end_date}")

        while current_start_date < end_date:
            limit_per_call = 500 # Adjust as needed
            
            logger.debug(f"Fetching batch for {symbol} from {current_start_date} with limit {limit_per_call}")
            bars = await self.fetch_historical_ohlcv(symbol, timeframe, since=current_start_date, limit=limit_per_call)
            
            if not bars: # Includes None or empty list after fetch_historical_ohlcv's logging
                logger.info(f"No more data found for {symbol} ({timeframe}) starting {current_start_date}, or an error occurred during fetch.")
                break 
            
            # Filter bars that are strictly before the overall end_date
            # The timestamp from OHLCV is the start of the candle.
            # If a candle's start is >= end_date, we don't need it or subsequent ones.
            relevant_bars = [b for b in bars if b.timestamp < end_date]
            
            if not relevant_bars:
                if bars and bars[0].timestamp >= end_date: # First fetched bar is already past our period
                    logger.debug(f"First bar fetched ({bars[0].timestamp}) is already at or after end_date ({end_date}). Stopping pagination.")
                break # No relevant bars in this batch

            all_bars.extend(relevant_bars)
            
            # Move to the next period: start after the last fetched relevant candle
            last_fetched_timestamp = relevant_bars[-1].timestamp
            # To get the start of the *next* candle, add the timeframe duration
            current_start_date = last_fetched_timestamp + timedelta(seconds=timeframe_duration_seconds)
            
            if current_start_date >= end_date: # Optimization: if next fetch starts at or after end_date
                logger.debug("Next calculated start_date is at or after end_date. Concluding pagination.")
                break
            
            logger.debug(f"Fetched {len(relevant_bars)} relevant bars. Next fetch for {symbol} will start from {current_start_date}. Total collected: {len(all_bars)}")
            
            # Respect rate limits (ensure rateLimit is a number)
            if isinstance(self.exchange.rateLimit, (int, float)) and self.exchange.rateLimit > 0:
                await asyncio.sleep(self.exchange.rateLimit / 1000.0) 
            else:
                await asyncio.sleep(0.2) # Default small delay if rateLimit is not standard

        # Remove duplicates (if any from overlapping fetches, though logic above tries to avoid it) and sort
        if all_bars:
            unique_bars_dict = {bar.timestamp: bar for bar in all_bars}
            all_bars = sorted(list(unique_bars_dict.values()), key=lambda b: b.timestamp)
            logger.info(f"Total unique historical bars fetched for {symbol} ({timeframe}) in period: {len(all_bars)}")
        
        return all_bars

    async def fetch_historical_data_for_pair(
        self,
        symbol1: str,
        symbol2: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime = datetime.now(timezone.utc)
    ) -> Tuple[Optional[List[BarData]], Optional[List[BarData]]]:
        """
        Fetches historical data for two symbols forming a pair.
        Returns a tuple of (data_symbol1, data_symbol2).
        Data is attempted to be synchronized by timestamp, but perfect sync is not guaranteed
        if one asset has missing bars where the other doesn't.
        Further alignment might be needed in the strategy.
        """
        logger.info(f"Fetching historical data for pair: {symbol1} and {symbol2} ({timeframe}) from {start_date} to {end_date}")
        
        data_symbol1 = await self.fetch_historical_data_for_period(symbol1, timeframe, start_date, end_date)
        data_symbol2 = await self.fetch_historical_data_for_period(symbol2, timeframe, start_date, end_date)

        if not data_symbol1:
            logger.warning(f"No data fetched for {symbol1} in the pair.")
        if not data_symbol2:
            logger.warning(f"No data fetched for {symbol2} in the pair.")
        
        # Basic check for data presence
        if not data_symbol1 or not data_symbol2:
            logger.warning(f"Could not fetch data for one or both assets in the pair ({symbol1}, {symbol2}).")
            return None, None # Indicate failure to fetch for one or both

        # Strategies will need to handle potential misalignments or use pandas to merge/align.
        logger.info(f"Fetched {len(data_symbol1)} bars for {symbol1} and {len(data_symbol2)} bars for {symbol2} for pair trading.")
        return data_symbol1, data_symbol2

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetches historical funding rate data for a perpetual futures symbol.
        """
        if not self.exchange.has.get('fetchFundingRateHistory'):
            logger.error(f"{self.exchange_id} does not support fetchFundingRateHistory.")
            return []

        since_timestamp_ms = int(since.timestamp() * 1000) if since else None
        funding_rates = []
        try:
            logger.info(f"Fetching funding rate history for {symbol} from {self.exchange_id} since {since}.")
            funding_rates = await self.exchange.fetch_funding_rate_history(symbol, since_timestamp_ms, limit)
            logger.info(f"Successfully fetched {len(funding_rates)} funding rate entries for {symbol}.")
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching funding rates for {symbol}: {e}", exc_info=True)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching funding rates for {symbol}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching funding rates for {symbol}: {e}", exc_info=True)
        return funding_rates


    async def subscribe_to_realtime_trades(self, symbol: str):
        # (Your existing placeholder code for this method)
        if not self.exchange.has['watchTrades']:
            logger.warning(f"{self.exchange_id} does not support real-time trade watching via WebSockets in CCXT.")
            # await self.close() # Consider if closing here is always appropriate
            return

        logger.info(f"Attempting to subscribe to real-time trades for {symbol} on {self.exchange_id}...")
        logger.warning("Real-time data subscription is a placeholder and not fully implemented.")
        pass

    async def close(self):
        """Closes the CCXT exchange connection."""
        try:
            if hasattr(self.exchange, 'close') and callable(self.exchange.close):
                await self.exchange.close()
            logger.info(f"CCXT exchange connection for {self.exchange_id} closed.")
        except Exception as e:
            logger.error(f"Error closing CCXT exchange connection for {self.exchange_id}: {e}", exc_info=True)