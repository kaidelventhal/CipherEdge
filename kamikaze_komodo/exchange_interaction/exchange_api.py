# kamikaze_komodo/exchange_interaction/exchange_api.py
import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Optional, List
from kamikaze_komodo.core.enums import OrderType, OrderSide
from kamikaze_komodo.core.models import Order
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings
from datetime import datetime

logger = get_logger(__name__)

class ExchangeAPI:
    """
    Handles interactions with the cryptocurrency exchange (e.g., Kraken).
    Manages order placement, cancellation, and fetching account information.
    """
    def __init__(self, exchange_id: str = 'kraken'):
        if not settings:
            logger.critical("Settings not loaded. ExchangeAPI cannot be initialized.")
            raise ValueError("Settings not loaded.")
            
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, self.exchange_id, None)
        if not exchange_class:
            logger.error(f"Exchange {self.exchange_id} is not supported by CCXT.")
            raise ValueError(f"Exchange {self.exchange_id} is not supported by CCXT.")

        config = {
            'apiKey': settings.kraken_api_key,
            'secret': settings.kraken_secret_key,
            'enableRateLimit': True,
        }
        # Add testnet/sandbox configuration if applicable and supported by CCXT for the exchange
        # if settings.kraken_testnet and self.exchange_id == 'binance':
        #     config['options'] = {'defaultType': 'future', 'test': True}
        # Kraken's sandbox typically uses different API keys for a demo account.

        self.exchange = exchange_class(config)
        logger.info(f"Initialized ExchangeAPI for {self.exchange_id}. Testnet mode: {settings.kraken_testnet}")

        if not settings.kraken_api_key or settings.kraken_api_key == "YOUR_API_KEY_REPLACE_ME":
            logger.warning(f"API keys for {self.exchange_id} are not properly set. Authenticated calls will fail.")


    async def fetch_balance(self) -> Optional[Dict]:
        """
        Fetches the account balance from the exchange.
        Returns:
            Optional[Dict]: A dictionary representing the account balance, or None on error.
                            The structure is defined by CCXT's fetch_balance method.
        """
        if not self.exchange.has['fetchBalance']:
            logger.error(f"{self.exchange_id} does not support fetchBalance.")
            return None
        try:
            balance = await self.exchange.fetch_balance()
            logger.info(f"Successfully fetched balance from {self.exchange_id}.")
            # logger.debug(f"Balance details: {balance}") # Can be very verbose
            return balance
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching balance: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching balance: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching balance: {e}")
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
        """
        Places an order on the exchange.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USD').
            order_type (OrderType): Type of order (market, limit).
            side (OrderSide): 'buy' or 'sell'.
            amount (float): Quantity of the asset to trade.
            price (Optional[float]): Price for limit orders. Required if order_type is LIMIT.
            params (Optional[Dict]): Additional parameters for the exchange.

        Returns:
            Optional[Order]: An Order object representing the placed order, or None on error.
        """
        if order_type == OrderType.LIMIT and price is None:
            logger.error("Price must be specified for a LIMIT order.")
            return None
        
        if not self.exchange.has['createOrder']:
            logger.error(f"{self.exchange_id} does not support createOrder.")
            return None

        order_type_str = order_type.value
        side_str = side.value

        try:
            logger.info(f"Attempting to place {side_str} {order_type_str} order for {amount} {symbol} at price {price if price else 'market'}")
            
            # Placeholder: Simulate order creation if in testnet and keys are dummy
            if settings.kraken_testnet and (not settings.kraken_api_key or "YOUR_API_KEY" in settings.kraken_api_key):
                logger.warning("Simulating order creation due to testnet mode and dummy API keys.")
                simulated_order_id = f"sim_{self.exchange_id}_{ccxt.Exchange.uuid()}"
                return Order(
                    id=simulated_order_id,
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price if order_type == OrderType.LIMIT else None, # Market orders might not have a 'creation' price
                    timestamp=datetime.utcnow(),
                    status="open", # Simulated as open initially
                    exchange_id=simulated_order_id
                )

            # Actual order creation
            exchange_order = await self.exchange.create_order(symbol, order_type_str, side_str, amount, price, params or {})
            logger.info(f"Successfully placed order on {self.exchange_id}. Order ID: {exchange_order.get('id')}")
            
            # Map CCXT order response to our Order model
            # This mapping can be quite detailed depending on the exchange response.
            created_order = Order(
                id=str(exchange_order.get('id')),
                symbol=exchange_order.get('symbol'),
                type=OrderType(exchange_order.get('type', order_type_str).lower()),
                side=OrderSide(exchange_order.get('side', side_str).lower()),
                amount=float(exchange_order.get('amount', amount)),
                price=float(exchange_order['price']) if exchange_order.get('price') else None,
                timestamp=datetime.fromtimestamp(exchange_order['timestamp'] / 1000, tz=timezone.utc) if exchange_order.get('timestamp') else datetime.utcnow(),
                status=exchange_order.get('status', 'open'),
                filled_amount=float(exchange_order.get('filled', 0.0)),
                average_fill_price=float(exchange_order.get('average')) if exchange_order.get('average') else None,
                exchange_id=str(exchange_order.get('id'))
            )
            return created_order

        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds to place order for {symbol}: {e}")
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order parameters for {symbol}: {e}")
        except ccxt.NetworkError as e:
            logger.error(f"Network error placing order for {symbol}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error placing order for {symbol}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred placing order for {symbol}: {e}")
        return None

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None, params: Optional[Dict] = None) -> bool:
        """
        Cancels an open order on the exchange.

        Args:
            order_id (str): The ID of the order to cancel.
            symbol (Optional[str]): The trading symbol (required by some exchanges for cancelOrder).
            params (Optional[Dict]): Additional parameters for the exchange.

        Returns:
            bool: True if the order was successfully canceled, False otherwise.
        """
        if not self.exchange.has['cancelOrder']:
            logger.error(f"{self.exchange_id} does not support cancelOrder.")
            return False
        try:
            # Some exchanges require symbol for cancelOrder, others don't.
            # CCXT unified API for cancel_order(id, symbol=None, params={})
            await self.exchange.cancel_order(order_id, symbol, params or {})
            logger.info(f"Successfully requested cancellation for order ID {order_id} on {self.exchange_id}.")
            return True # Note: Cancellation might be a request; status needs to be confirmed by fetch_order.
        except ccxt.OrderNotFound as e:
            logger.error(f"Order ID {order_id} not found for cancellation: {e}")
        except ccxt.NetworkError as e:
            logger.error(f"Network error canceling order {order_id}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error canceling order {order_id}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred canceling order {order_id}: {e}")
        return False

    async def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Order]:
        """
        Fetches information about a specific order.
        """
        if not self.exchange.has['fetchOrder']:
            logger.warning(f"{self.exchange_id} does not support fetching individual orders directly.")
            return None
        try:
            exchange_order = await self.exchange.fetch_order(order_id, symbol)
            # Map to internal Order model
            fetched_order = Order(
                id=str(exchange_order.get('id')),
                symbol=exchange_order.get('symbol'),
                type=OrderType(exchange_order.get('type').lower()),
                side=OrderSide(exchange_order.get('side').lower()),
                amount=float(exchange_order.get('amount')),
                price=float(exchange_order['price']) if exchange_order.get('price') else None,
                timestamp=datetime.fromtimestamp(exchange_order['timestamp'] / 1000, tz=timezone.utc) if exchange_order.get('timestamp') else datetime.utcnow(),
                status=exchange_order.get('status'),
                filled_amount=float(exchange_order.get('filled', 0.0)),
                average_fill_price=float(exchange_order.get('average')) if exchange_order.get('average') else None,
                exchange_id=str(exchange_order.get('id'))
            )
            return fetched_order
        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found on {self.exchange_id}.")
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
        return None

    async def fetch_open_orders(self, symbol: Optional[str] = None, since: Optional[int] = None, limit: Optional[int] = None) -> List[Order]:
        """
        Fetches all open orders for a given symbol or all symbols.
        """
        open_orders_list = []
        if not self.exchange.has['fetchOpenOrders']:
            logger.warning(f"{self.exchange_id} does not support fetching open orders.")
            return open_orders_list
        try:
            raw_orders = await self.exchange.fetch_open_orders(symbol, since, limit)
            for ex_order in raw_orders:
                order = Order(
                    id=str(ex_order.get('id')),
                    symbol=ex_order.get('symbol'),
                    type=OrderType(ex_order.get('type').lower()),
                    side=OrderSide(ex_order.get('side').lower()),
                    amount=float(ex_order.get('amount')),
                    price=float(ex_order['price']) if ex_order.get('price') else None,
                    timestamp=datetime.fromtimestamp(ex_order['timestamp'] / 1000, tz=timezone.utc) if ex_order.get('timestamp') else datetime.utcnow(),
                    status=ex_order.get('status', 'open'),
                    filled_amount=float(ex_order.get('filled', 0.0)),
                    average_fill_price=float(ex_order.get('average')) if ex_order.get('average') else None,
                    exchange_id=str(ex_order.get('id'))
                )
                open_orders_list.append(order)
            logger.info(f"Fetched {len(open_orders_list)} open orders for symbol {symbol if symbol else 'all'}.")
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
        return open_orders_list


    async def close(self):
        """Closes the CCXT exchange connection."""
        try:
            if hasattr(self.exchange, 'close') and callable(self.exchange.close):
                await self.exchange.close()
            logger.info(f"CCXT exchange connection for {self.exchange_id} closed.")
        except Exception as e:
            logger.error(f"Error closing CCXT exchange connection: {e}")

# Example Usage (run within an asyncio event loop):
async def main_exchange_api_example():
    if not settings:
        print("Settings could not be loaded. Exiting example.")
        return

    exchange_api = ExchangeAPI(exchange_id='kraken')

    # Fetch balance
    balance = await exchange_api.fetch_balance()
    if balance:
        # Log specific free balances if needed, e.g., USD, BTC
        logger.info(f"Free USD Balance: {balance.get('USD', {}).get('free', 0)}")
        logger.info(f"Free BTC Balance: {balance.get('BTC', {}).get('free', 0)}")
    
    # Example: Place a practice order (simulated if keys are dummy)
    # Ensure the symbol and amounts are reasonable for testing.
    # Kraken typically requires minimum order sizes. E.g., BTC/USD might need > 0.0001 BTC
    # This will likely fail with "InsufficientFunds" or "InvalidOrder" on a live account without funds or proper setup.
    
    # For actual testing with dummy keys, the simulation path in create_order will be hit.
    # If you have paper trading keys, replace the placeholders in secrets.ini.
    
    # target_symbol = settings.default_symbol # e.g. 'BTC/USD'
    # order_to_place = await exchange_api.create_order(
    #     symbol=target_symbol,
    #     order_type=OrderType.LIMIT, # Use LIMIT to avoid unexpected fills with market orders
    #     side=OrderSide.BUY,
    #     amount=0.0001,  # Example small amount, check Kraken's minimums
    #     price=15000.0   # Example very low price for a limit buy to ensure it doesn't fill immediately
    # )

    # if order_to_place:
    #     logger.info(f"Practice order placed/simulated: ID {order_to_place.id}, Status {order_to_place.status}")
        
    #     # Fetch the status of this specific order
    #     await asyncio.sleep(2) # Give some time for order to register (if live)
    #     fetched_order_status = await exchange_api.fetch_order(order_to_place.id, order_to_place.symbol)
    #     if fetched_order_status:
    #         logger.info(f"Fetched status for order {fetched_order_status.id}: {fetched_order_status.status}")

    #     # Try to cancel the order
    #     # if order_to_place.status == 'open' or "sim_" in order_to_place.id: # Only cancel if open or simulated
    #     #     logger.info(f"Attempting to cancel order: {order_to_place.id}")
    #     #     cancel_success = await exchange_api.cancel_order(order_to_place.id, order_to_place.symbol)
    #     #     logger.info(f"Cancellation request for order {order_to_place.id} successful: {cancel_success}")
    # else:
    #     logger.warning("Practice order placement failed or was not attempted.")

    await exchange_api.close()

if __name__ == '__main__':
    # Standalone test
    # from kamikaze_komodo.config.settings import Config
    # settings_instance = Config(config_file='../config/config.ini', secrets_file='../config/secrets.ini')
    # global settings
    # settings = settings_instance
    # if settings:
    #    asyncio.run(main_exchange_api_example())
    # else:
    #    print("Failed to load settings for standalone exchange_api example.")
    pass