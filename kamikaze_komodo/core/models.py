# kamikaze_komodo/core/models.py
# Added fields to NewsArticle and Trade as potentially needed by new modules.
# Added prediction_value and prediction_confidence to BarData for Phase 5 ML strategy.
from typing import Optional, List, Dict, Any # Added Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import pydantic # Added timezone
from kamikaze_komodo.core.enums import OrderType, OrderSide, SignalType, TradeResult
class BarData(BaseModel):
    """
    Represents OHLCV market data for a specific time interval.
    """
    timestamp: datetime = Field(..., description="The start time of the candle, expected to be timezone-aware (UTC)")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
    symbol: Optional[str] = Field(None, description="Trading symbol, e.g., BTC/USD")
    timeframe: Optional[str] = Field(None, description="Candle timeframe, e.g., 1h")
    
    # Optional fields for indicators or sentiment
    atr: Optional[float] = Field(None, description="Average True Range at this bar")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score associated with this bar's timestamp")
    
    # Fields for ML predictions (Phase 5)
    prediction_value: Optional[float] = Field(None, description="Predicted value by an ML model (e.g., future price, return)")
    prediction_confidence: Optional[float] = Field(None, description="Confidence of the ML prediction (0.0 to 1.0)")
    class Config:
        frozen = False # Allow modification by strategies/engine (e.g. to add ATR or predictions)
        
class Order(BaseModel):
    # ... (no changes from existing)
    id: str = Field(..., description="Unique order identifier (from exchange or internal)")
    symbol: str = Field(..., description="Trading symbol, e.g., BTC/USD")
    type: OrderType = Field(..., description="Type of order (market, limit, etc.)")
    side: OrderSide = Field(..., description="Order side (buy or sell)")
    amount: float = Field(..., gt=0, description="Quantity of the asset to trade")
    price: Optional[float] = Field(None, gt=0, description="Price for limit or stop orders")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Time the order was created")
    status: str = Field("open", description="Current status of the order (e.g., open, filled, canceled)")
    filled_amount: float = Field(0.0, ge=0, description="Amount of the order that has been filled")
    average_fill_price: Optional[float] = Field(None, description="Average price at which the order was filled")
    exchange_id: Optional[str] = Field(None, description="Order ID from the exchange")
class Trade(BaseModel):
    """
    Represents an executed trade.
    """
    id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol, e.g., BTC/USD")
    entry_order_id: str = Field(..., description="ID of the order that opened the trade")
    exit_order_id: Optional[str] = Field(None, description="ID of the order that closed the trade")
    side: OrderSide = Field(..., description="Trade side (buy/long or sell/short)")
    entry_price: float = Field(..., gt=0, description="Price at which the trade was entered")
    exit_price: Optional[float] = Field(None, description="Price at which the trade was exited (must be >0 if set)")
    amount: float = Field(..., gt=0, description="Quantity of the asset traded")
    entry_timestamp: datetime = Field(..., description="Time the trade was entered")
    exit_timestamp: Optional[datetime] = Field(None, description="Time the trade was exited")
    pnl: Optional[float] = Field(None, description="Profit or Loss for the trade")
    pnl_percentage: Optional[float] = Field(None, description="Profit or Loss percentage for the trade")
    commission: float = Field(0.0, ge=0, description="Trading commission paid")
    result: Optional[TradeResult] = Field(None, description="Outcome of the trade (Win/Loss/Breakeven)")
    notes: Optional[str] = Field(None, description="Any notes related to the trade")
    # Added for ATR based stops or other context
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields for additional trade data, e.g., atr_at_entry")
    @pydantic.field_validator('exit_price')
    def exit_price_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('exit_price must be positive if set')
        return v
class NewsArticle(BaseModel):
    """
    Represents a news article relevant to market analysis.
    """
    id: str = Field(..., description="Unique identifier for the news article (e.g., URL hash or URL itself)")
    url: str = Field(..., description="Source URL of the article")
    title: str = Field(..., description="Headline or title of the article")
    publication_date: Optional[datetime] = Field(None, description="Date the article was published (UTC)")
    retrieval_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Date the article was retrieved (UTC)")
    source: str = Field(..., description="Source of the news (e.g., CoinDesk, CoinTelegraph, RSS feed name)")
    content: Optional[str] = Field(None, description="Full text content of the article")
    summary: Optional[str] = Field(None, description="AI-generated or scraped summary")
    
    # Sentiment related fields
    sentiment_score: Optional[float] = Field(None, description="Overall sentiment score (-1.0 to 1.0)")
    sentiment_label: Optional[str] = Field(None, description="Sentiment label (e.g., positive, negative, neutral, bullish, bearish)")
    sentiment_confidence: Optional[float] = Field(None, description="Confidence of the sentiment analysis (0.0 to 1.0)")
    key_themes: Optional[List[str]] = Field(default_factory=list, description="Key themes identified by sentiment analysis")
    
    related_symbols: Optional[List[str]] = Field(default_factory=list, description="Cryptocurrencies mentioned or related")
    raw_llm_response: Optional[Dict[str, Any]] = Field(None, description="Raw response from LLM for sentiment if available")
class PortfolioSnapshot(BaseModel):
    # ... (no changes from existing)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_value_usd: float = Field(..., description="Total portfolio value in USD")
    cash_balance_usd: float = Field(..., description="Available cash in USD")
    positions: Dict[str, float] = Field(default_factory=dict, description="Asset quantities, e.g., {'BTC': 0.5, 'ETH': 10}") # symbol: quantity
    open_pnl_usd: float = Field(0.0, description="Total open Profit/Loss in USD for current positions")