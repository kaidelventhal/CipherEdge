# kamikaze_komodo/data_handling/database_manager.py
# Updated to include store/retrieve for NewsArticle
# Phase 6: Minor modification to bar_data table to include market_regime.
import sqlite3
from typing import List, Optional, Dict, Any # Added Dict, Any
from kamikaze_komodo.core.models import BarData, NewsArticle # Added NewsArticle
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone, UTC 
import json # For storing dicts/lists like related_symbols or key_themes

logger = get_logger(__name__)

class DatabaseManager:
    """
    Manages local storage of data (initially SQLite).
    Timestamps are stored as ISO 8601 TEXT.
    Lists/Dicts are stored as JSON TEXT.
    """
    def __init__(self, db_name: str = "kamikaze_komodo_data.db"):
        self.db_name = db_name
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_COLNAMES)
            self.conn.row_factory = sqlite3.Row 
            logger.info(f"Successfully connected to database: {self.db_name}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_name}: {e}")
            self.conn = None

    def _create_tables(self):
        if not self.conn:
            logger.error("Cannot create tables, no database connection.")
            return
        try:
            cursor = self.conn.cursor()
            # BarData Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bar_data (
                    timestamp TEXT NOT NULL, 
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    atr REAL, 
                    sentiment_score REAL,
                    prediction_value REAL, -- Phase 5
                    prediction_confidence REAL, -- Phase 5
                    market_regime INTEGER, -- Phase 6
                    PRIMARY KEY (timestamp, symbol, timeframe)
                )
            """)
            # NewsArticle Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    publication_date TEXT, 
                    retrieval_date TEXT NOT NULL, 
                    source TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    sentiment_confidence REAL,
                    key_themes TEXT,
                    related_symbols TEXT,
                    raw_llm_response TEXT 
                )
            """)
            # Trades Table (Example for future use)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, entry_order_id TEXT, exit_order_id TEXT,
                    side TEXT NOT NULL, entry_price REAL NOT NULL, exit_price REAL, amount REAL NOT NULL,
                    entry_timestamp TEXT NOT NULL, exit_timestamp TEXT, pnl REAL, pnl_percentage REAL,
                    commission REAL DEFAULT 0.0, result TEXT, notes TEXT, custom_fields TEXT
                )
            """)
            self.conn.commit()
            logger.info("Tables checked/created successfully (timestamps as TEXT, complex fields as JSON TEXT).")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")

    def _to_iso_format(self, dt: Optional[datetime]) -> Optional[str]:
        if dt is None: return None
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.isoformat()

    def _from_iso_format(self, iso_str: Optional[str]) -> Optional[datetime]:
        if iso_str is None: return None
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            logger.warning(f"Could not parse ISO timestamp string: {iso_str}")
            return None

    def store_bar_data(self, bar_data_list: List[BarData]):
        if not self.conn: logger.error("No DB connection for bar data."); return False
        if not bar_data_list: logger.info("No bar data to store."); return True
        try:
            cursor = self.conn.cursor()
            data_to_insert = [
                (
                    self._to_iso_format(bd.timestamp), bd.symbol, bd.timeframe,
                    bd.open, bd.high, bd.low, bd.close, bd.volume,
                    bd.atr, bd.sentiment_score,
                    bd.prediction_value, bd.prediction_confidence, # Phase 5
                    bd.market_regime # Phase 6
                ) for bd in bar_data_list
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO bar_data 
                (timestamp, symbol, timeframe, open, high, low, close, volume, atr, sentiment_score,
                 prediction_value, prediction_confidence, market_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
            """, data_to_insert) 
            self.conn.commit()
            logger.info(f"Stored/Replaced {len(data_to_insert)} bar data entries. ({cursor.rowcount} affected)")
            return True
        except Exception as e:
            logger.error(f"Error storing bar data: {e}", exc_info=True); return False

    def retrieve_bar_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[BarData]:
        if not self.conn: logger.error("No DB connection for bar data."); return []
        try:
            cursor = self.conn.cursor()
            query = "SELECT timestamp, open, high, low, close, volume, symbol, timeframe, atr, sentiment_score, prediction_value, prediction_confidence, market_regime FROM bar_data WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]
            if start_date: query += " AND timestamp >= ?"; params.append(self._to_iso_format(start_date))
            if end_date: query += " AND timestamp <= ?"; params.append(self._to_iso_format(end_date))
            query += " ORDER BY timestamp ASC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            bar_data_list = []
            for row in rows:
                dt_object = self._from_iso_format(row['timestamp'])
                if not dt_object: continue
                bar_data_list.append(BarData(
                    timestamp=dt_object, open=row['open'], high=row['high'], low=row['low'],
                    close=row['close'], volume=row['volume'], symbol=row['symbol'], timeframe=row['timeframe'],
                    atr=row['atr'], sentiment_score=row['sentiment_score'],
                    prediction_value=row['prediction_value'], prediction_confidence=row['prediction_confidence'], # Phase 5
                    market_regime=row['market_regime'] # Phase 6
                ))
            logger.info(f"Retrieved {len(bar_data_list)} bar data entries for {symbol} ({timeframe}).")
            return bar_data_list
        except Exception as e:
            logger.error(f"Error retrieving bar data for {symbol} ({timeframe}): {e}", exc_info=True); return []

    def store_news_articles(self, articles: List[NewsArticle]):
        if not self.conn: logger.error("No DB connection for news articles."); return False
        if not articles: logger.info("No news articles to store."); return True
        try:
            cursor = self.conn.cursor()
            data_to_insert = []
            for article in articles:
                data_to_insert.append((
                    article.id, article.url, article.title,
                    self._to_iso_format(article.publication_date),
                    self._to_iso_format(article.retrieval_date),
                    article.source, article.content, article.summary,
                    article.sentiment_score, article.sentiment_label,
                    article.sentiment_confidence,
                    json.dumps(article.key_themes) if article.key_themes else None,
                    json.dumps(article.related_symbols) if article.related_symbols else None,
                    json.dumps(article.raw_llm_response) if article.raw_llm_response else None
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO news_articles
                (id, url, title, publication_date, retrieval_date, source, content, summary,
                 sentiment_score, sentiment_label, sentiment_confidence, key_themes, related_symbols, raw_llm_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            self.conn.commit()
            logger.info(f"Stored/Replaced {len(data_to_insert)} news articles. ({cursor.rowcount} affected)")
            return True
        except Exception as e:
            logger.error(f"Error storing news articles: {e}", exc_info=True); return False

    def retrieve_news_articles(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, source: Optional[str] = None, limit: int = 100) -> List[NewsArticle]:
        if not self.conn: logger.error("No DB connection for news articles."); return []
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM news_articles WHERE 1=1"
            params = []

            if symbol: # Search in related_symbols (requires LIKE or a better FTS setup)
                query += " AND related_symbols LIKE ?"
                params.append(f'%"{symbol}"%') # Simple JSON array search, not very efficient
            if start_date: # Based on publication_date
                query += " AND publication_date >= ?"
                params.append(self._to_iso_format(start_date))
            if end_date:
                query += " AND publication_date <= ?"
                params.append(self._to_iso_format(end_date))
            if source:
                query += " AND source = ?"
                params.append(source)
            
            query += " ORDER BY publication_date DESC, retrieval_date DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            articles_list = []
            for row in rows:
                articles_list.append(NewsArticle(
                    id=row['id'], url=row['url'], title=row['title'],
                    publication_date=self._from_iso_format(row['publication_date']),
                    retrieval_date=self._from_iso_format(row['retrieval_date']),
                    source=row['source'], content=row['content'], summary=row['summary'],
                    sentiment_score=row['sentiment_score'], sentiment_label=row['sentiment_label'],
                    sentiment_confidence=row['sentiment_confidence'],
                    key_themes=json.loads(row['key_themes']) if row['key_themes'] else [],
                    related_symbols=json.loads(row['related_symbols']) if row['related_symbols'] else [],
                    raw_llm_response=json.loads(row['raw_llm_response']) if row['raw_llm_response'] else None
                ))
            logger.info(f"Retrieved {len(articles_list)} news articles with given criteria.")
            return articles_list
        except Exception as e:
            logger.error(f"Error retrieving news articles: {e}", exc_info=True); return []

    def close(self):
        if self.conn: self.conn.close(); logger.info("Database connection closed."); self.conn = None

    def __del__(self): self.close()