# kamikaze_komodo/core/utils.py
from datetime import datetime, timezone

from kamikaze_komodo.core.models import BarData

def format_timestamp(ts: datetime, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Formats a datetime object into a string.
    Ensures timezone awareness, defaulting to UTC if naive.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.strftime(fmt)

def current_timestamp_ms() -> int:
    """
    Returns the current UTC timestamp in milliseconds.
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def ohlcv_to_bardata(ohlcv: list, symbol: str, timeframe: str) -> BarData:
    """
    Converts a CCXT OHLCV list [timestamp_ms, open, high, low, close, volume]
    to a BarData object.
    """
    from kamikaze_komodo.core.models import BarData # Local import to avoid circular dependency
    
    if len(ohlcv) != 6:
        raise ValueError("OHLCV list must contain 6 elements: timestamp, open, high, low, close, volume")

    dt_object = datetime.fromtimestamp(ohlcv[0] / 1000, tz=timezone.utc)
    return BarData(
        timestamp=dt_object,
        open=float(ohlcv[1]),
        high=float(ohlcv[2]),
        low=float(ohlcv[3]),
        close=float(ohlcv[4]),
        volume=float(ohlcv[5]),
        symbol=symbol,
        timeframe=timeframe
    )

# Add other utility functions as needed, e.g.,
# - Mathematical helpers not in TA-Lib
# - Data validation functions
# - etc.