# kamikaze_komodo/exchange_interaction/exchange_api.py
import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Optional, List
from kamikaze_komodo.core.enums import OrderType, OrderSide
from kamikaze_komodo.core.models import Order
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings # Import global settings
from datetime import datetime, timezone # Ensure timezone is imported

logger = get_logger(__name__)

class ExchangeAPI:
    """
    Handles interactions with the cryptocurrency exchange.
    Manages order placement, cancellation, and fetching account information.
    Phase 6: Added explicit check for short selling capability (though CCXT often handles this implicitly for derivative exchanges).
    """
    def __init__(self, exchange_id: Optional[str] = None): # exchange_id is now optional
        if not settings:
            logger.critical("Settings not loaded. ExchangeAPI cannot be initialized.")
            raise ValueError("Settings not loaded.")

        self.exchange_id = exchange_id if exchange_id else settings.exchange_id_to_use
        exchange_class = getattr(ccxt, self.exchange_id, None)

        if not exchange_class:
            logger.error(f"Exchange {self.exchange_id} is not supported by CCXT.")
            raise ValueError(f"Exchange {self.exchange_id} is not supported by CCXT.")

        # Determine API keys based on the exchange_id
        # This example assumes a single set of keys in settings (e.g., KRAKEN_API)
        # For a multi-exchange system, you'd fetch keys specific to self.exchange_id
        api_key = settings.kraken_api_key # Defaulting to Kraken keys for now
        secret_key = settings.kraken_secret_key # Defaulting to Kraken keys
        use_testnet = settings.kraken_testnet # Defaulting to Kraken testnet setting

        # Example for specific exchange key loading (if settings were structured differently)
        # if self.exchange_id == 'binance':
        #     api_key = settings.binance_api_key
        #     secret_key = settings.binance_secret_key
        #     use_testnet = settings.binance_testnet
        # elif self.exchange_id == 'krakenfutures':
        #     api_key = settings.kraken_futures_api_key # etc.

        config = {
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
        }
        self.exchange = exchange_class(config)
        logger.info(f"Initialized ExchangeAPI for {self.exchange_id}.")

        if use_testnet:
            if hasattr(self.exchange, 'set_sandbox_mode') and callable(self.exchange.set_sandbox_mode):
                try:
                    self.exchange.set_sandbox_mode(True)
                    logger.info(f"Sandbox mode enabled for {self.exchange_id}.")
                except ccxt.NotSupported:
                    logger.warning(f"{self.exchange_id} does not support unified set_sandbox_mode. Testnet functionality depends on API keys/URL.")
                except Exception as e_sandbox:
                    logger.error(f"Error setting sandbox mode for {self.exchange_id}: {e_sandbox}")
            else:
                logger.warning(f"{self.exchange_id} does not have set_sandbox_mode. Testnet relies on specific API keys or default URL pointing to sandbox.")
        else:
            logger.info(f"Running in live mode for {self.exchange_id}.")

        if not api_key or "YOUR_API_KEY" in str(api_key).upper() or (isinstance(api_key, str) and "D27PYGI95TLS" in api_key.upper()): # Check specific placeholder
            logger.warning(f"API key for {self.exchange_id} appears to be a placeholder or is not configured. Authenticated calls may fail.")

    async def fetch_balance(self) -> Optional[Dict]:
        if not self.exchange.has['fetchBalance']:
            logger.error(f"{self.exchange_id} does not support fetchBalance.")
            return None
        try:
            balance = await self.exchange.fetch_balance()
            logger.info(f"Successfully fetched balance from {self.exchange_id}.")
            return balance
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching balance: {e}", exc_info=True)
        except ccxt.AuthenticationError as e_auth:
            logger.error(f"Authentication error fetching balance from {self.exchange_id}. Check API keys and permissions: {e_auth}", exc_info=True)
            return None # Explicitly return None on auth error
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching balance: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching balance: {e}", exc_info=True)
        return None

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Optional[Order]:
        if order_type == OrderType.LIMIT and price is None:
            logger.error("Price must be specified for a LIMIT order.")
            return None

        if not self.exchange.has['createOrder']:
            logger.error(f"{self.exchange_id} does not support createOrder.")
            return None

        order_type_str = order_type.value
        side_str = side.value

        # Phase 6: Check for short selling specific capabilities (conceptual for CCXT futures)
        if side == OrderSide.SELL: # This could be opening a short or closing a long
            # For many futures exchanges, 'sell' with no existing position implies short.
            # CCXT often handles this implicitly. Some exchanges might need specific params for short.
            # e.g., params = {'reduceOnly': False} if it was to ensure opening a new position.
            # We assume for now that a simple SELL order will open a short if no long position exists.
            # If the exchange has explicit shorting methods (less common in CCXT unified API), that'd be different.
            logger.info(f"Preparing to place a SELL order for {symbol}. This may open a short position.")

        try:
            logger.info(f"Attempting to place {side_str} {order_type_str} order for {amount} {symbol} at price {price if price else 'market'} on {self.exchange_id}")
            
            # Check for placeholder API keys again before actual call
            is_placeholder_key = not self.exchange.apiKey or "YOUR_API_KEY" in self.exchange.apiKey.upper() or "D27PYGI95TLS" in self.exchange.apiKey.upper()
            if settings.kraken_testnet and is_placeholder_key : # Use general testnet flag
                logger.warning(f"Simulating order creation for {self.exchange_id} due to testnet mode and placeholder API keys.")
                simulated_order_id = f"sim_{self.exchange_id}_{ccxt.Exchange.uuid()}"
                return Order(
                    id=simulated_order_id,
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price if order_type == OrderType.LIMIT else None,
                    timestamp=datetime.now(timezone.utc), # Use timezone.utc
                    status="open", # Simulate as open
                    exchange_id=simulated_order_id
                )

            exchange_order_response = await self.exchange.create_order(symbol, order_type_str, side_str, amount, price, params or {})
            logger.info(f"Successfully placed order on {self.exchange_id}. Order ID: {exchange_order_response.get('id')}")
            
            created_order = Order(
                id=str(exchange_order_response.get('id')),
                symbol=exchange_order_response.get('symbol'),
                type=OrderType(exchange_order_response.get('type', order_type_str).lower()),
                side=OrderSide(exchange_order_response.get('side', side_str).lower()),
                amount=float(exchange_order_response.get('amount', amount)),
                price=float(exchange_order_response['price']) if exchange_order_response.get('price') else None,
                timestamp=datetime.fromtimestamp(exchange_order_response['timestamp'] / 1000, tz=timezone.utc) if exchange_order_response.get('timestamp') else datetime.now(timezone.utc),
                status=exchange_order_response.get('status', 'open'),
                filled_amount=float(exchange_order_response.get('filled', 0.0)),
                average_fill_price=float(exchange_order_response.get('average')) if exchange_order_response.get('average') else None,
                exchange_id=str(exchange_order_response.get('id'))
            )
            return created_order

        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds to place order for {symbol} on {self.exchange_id}: {e}", exc_info=True)
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order parameters for {symbol} on {self.exchange_id}: {e}", exc_info=True)
        except ccxt.AuthenticationError as e_auth:
            logger.error(f"Authentication error placing order for {symbol} on {self.exchange_id}. Check API keys: {e_auth}", exc_info=True)
        except ccxt.NetworkError as e:
            logger.error(f"Network error placing order for {symbol} on {self.exchange_id}: {e}", exc_info=True)
        except ccxt.ExchangeError as e: # Catch specific exchange errors before generic Exception
            logger.error(f"Exchange error placing order for {symbol} on {self.exchange_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred placing order for {symbol} on {self.exchange_id}: {e}", exc_info=True)
        return None

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None, params: Optional[Dict] = None) -> bool:
        if not self.exchange.has['cancelOrder']:
            logger.error(f"{self.exchange_id} does not support cancelOrder.")
            return False
        try:
            await self.exchange.cancel_order(order_id, symbol, params or {})
            logger.info(f"Successfully requested cancellation for order ID {order_id} on {self.exchange_id}.")
            return True
        except ccxt.OrderNotFound as e:
            logger.error(f"Order ID {order_id} not found for cancellation on {self.exchange_id}: {e}", exc_info=True)
        except ccxt.NetworkError as e:
            logger.error(f"Network error canceling order {order_id} on {self.exchange_id}: {e}", exc_info=True)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error canceling order {order_id} on {self.exchange_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred canceling order {order_id} on {self.exchange_id}: {e}", exc_info=True)
        return False

    async def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        if not self.exchange.has['fetchOrder']:
            logger.warning(f"{self.exchange_id} does not support fetching individual orders directly.")
            return None
        try:
            exchange_order_response = await self.exchange.fetch_order(order_id, symbol)
            fetched_order = Order(
                id=str(exchange_order_response.get('id')),
                symbol=exchange_order_response.get('symbol'),
                type=OrderType(exchange_order_response.get('type').lower()),
                side=OrderSide(exchange_order_response.get('side').lower()),
                amount=float(exchange_order_response.get('amount')),
                price=float(exchange_order_response['price']) if exchange_order_response.get('price') else None,
                timestamp=datetime.fromtimestamp(exchange_order_response['timestamp'] / 1000, tz=timezone.utc) if exchange_order_response.get('timestamp') else datetime.now(timezone.utc),
                status=exchange_order_response.get('status'),
                filled_amount=float(exchange_order_response.get('filled', 0.0)),
                average_fill_price=float(exchange_order_response.get('average')) if exchange_order_response.get('average') else None,
                exchange_id=str(exchange_order_response.get('id'))
            )
            return fetched_order
        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found on {self.exchange_id}.")
        except Exception as e:
            logger.error(f"Error fetching order {order_id} on {self.exchange_id}: {e}", exc_info=True)
        return None

    async def fetch_open_orders(self, symbol: Optional[str] = None, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[Order]: # Changed since to datetime
        open_orders_list = []
        if not self.exchange.has['fetchOpenOrders']:
            logger.warning(f"{self.exchange_id} does not support fetching open orders.")
            return open_orders_list

        try:
            since_timestamp_ms = int(since.timestamp() * 1000) if since else None
            raw_orders = await self.exchange.fetch_open_orders(symbol, since_timestamp_ms, limit)
            for ex_order in raw_orders:
                order = Order(
                    id=str(ex_order.get('id')),
                    symbol=ex_order.get('symbol'),
                    type=OrderType(ex_order.get('type').lower()),
                    side=OrderSide(ex_order.get('side').lower()),
                    amount=float(ex_order.get('amount')),
                    price=float(ex_order['price']) if ex_order.get('price') else None,
                    timestamp=datetime.fromtimestamp(ex_order['timestamp'] / 1000, tz=timezone.utc) if ex_order.get('timestamp') else datetime.now(timezone.utc),
                    status=ex_order.get('status', 'open'),
                    filled_amount=float(ex_order.get('filled', 0.0)),
                    average_fill_price=float(ex_order.get('average')) if ex_order.get('average') else None,
                    exchange_id=str(ex_order.get('id'))
                )
                open_orders_list.append(order)
            logger.info(f"Fetched {len(open_orders_list)} open orders for symbol {symbol if symbol else 'all'} on {self.exchange_id}.")
        except Exception as e:
            logger.error(f"Error fetching open orders on {self.exchange_id}: {e}", exc_info=True)
        return open_orders_list

    async def close(self):
        try:
            if hasattr(self.exchange, 'close') and callable(self.exchange.close):
                await self.exchange.close()
                logger.info(f"CCXT exchange connection for {self.exchange_id} closed.")
        except Exception as e:
            logger.error(f"Error closing CCXT exchange connection for {self.exchange_id}: {e}", exc_info=True)

# Example Usage (run within an asyncio event loop):
async def main_exchange_api_example():
    if not settings:
        print("Settings could not be loaded. Exiting example.")
        return

    exchange_api = ExchangeAPI() # Uses exchange_id from settings
    balance = await exchange_api.fetch_balance()
    if balance:
        logger.info(f"Free USD Balance: {balance.get('USD', {}).get('free', 'N/A')}")
        logger.info(f"Free {settings.default_symbol.split('/')[0]} Balance: {balance.get(settings.default_symbol.split('/')[0], {}).get('free', 'N/A')}")

    # target_symbol = settings.default_symbol
    # order_to_place = await exchange_api.create_order(
    #     symbol=target_symbol,
    #     order_type=OrderType.LIMIT,
    #     side=OrderSide.BUY,
    #     amount=0.0001,
    #     price=15000.0
    # )
    # if order_to_place:
    #     logger.info(f"Practice order placed/simulated: ID {order_to_place.id}, Status {order_to_place.status}")
    # else:
    #     logger.warning("Practice order placement failed or was not attempted.")

    await exchange_api.close()