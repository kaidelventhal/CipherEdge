# kamikaze_komodo/data_handling/database_manager.py
# Updated to include store/retrieve for NewsArticle
# Phase 6: Minor modification to bar_data table to include market_regime.
import sqlite3
from typing import List, Optional, Dict, Any # Added Dict, Any
from kamikaze_komodo.core.models import BarData, NewsArticle, FundingRate # Added FundingRate
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
            # Add funding_rate column
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
                    prediction_value REAL,
                    prediction_confidence REAL,
                    market_regime INTEGER,
                    funding_rate REAL,
                    PRIMARY KEY (timestamp, symbol, timeframe)
                )
            """)
            # Funding Rate Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL NOT NULL,
                    mark_price REAL,
                    PRIMARY KEY (timestamp, symbol)
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
            self.conn.commit()
            logger.info("Tables checked/created successfully.")
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
                    bd.prediction_value, bd.prediction_confidence,
                    bd.market_regime, bd.funding_rate
                ) for bd in bar_data_list
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO bar_data 
                (timestamp, symbol, timeframe, open, high, low, close, volume, atr, sentiment_score,
                 prediction_value, prediction_confidence, market_regime, funding_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
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
            query = "SELECT * FROM bar_data WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]
            if start_date: query += " AND timestamp >= ?"; params.append(self._to_iso_format(start_date))
            if end_date: query += " AND timestamp <= ?"; params.append(self._to_iso_format(end_date))
            query += " ORDER BY timestamp ASC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            bar_data_list = [BarData(**row) for row in (dict(row) for row in rows)]
            logger.info(f"Retrieved {len(bar_data_list)} bar data entries for {symbol} ({timeframe}).")
            return bar_data_list
        except Exception as e:
            logger.error(f"Error retrieving bar data for {symbol} ({timeframe}): {e}", exc_info=True); return []

    def store_funding_rates(self, funding_rates: List[FundingRate]):
        if not self.conn: logger.error("No DB connection for funding rates."); return False
        if not funding_rates: logger.info("No funding rates to store."); return True
        try:
            cursor = self.conn.cursor()
            data_to_insert = [
                (self._to_iso_format(fr.timestamp), fr.symbol, fr.funding_rate, fr.mark_price) for fr in funding_rates
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO funding_rates
                (timestamp, symbol, funding_rate, mark_price)
                VALUES (?, ?, ?, ?)
            """, data_to_insert)
            self.conn.commit()
            logger.info(f"Stored/Replaced {len(data_to_insert)} funding rate entries.")
            return True
        except Exception as e:
            logger.error(f"Error storing funding rates: {e}", exc_info=True); return False

    def retrieve_funding_rates(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[FundingRate]:
        if not self.conn: logger.error("No DB connection for funding rates."); return []
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM funding_rates WHERE symbol = ?"
            params = [symbol]
            if start_date: query += " AND timestamp >= ?"; params.append(self._to_iso_format(start_date))
            if end_date: query += " AND timestamp <= ?"; params.append(self._to_iso_format(end_date))
            query += " ORDER BY timestamp ASC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            rates_list = [FundingRate(**row) for row in (dict(row) for row in rows)]
            logger.info(f"Retrieved {len(rates_list)} funding rate entries for {symbol}.")
            return rates_list
        except Exception as e:
            logger.error(f"Error retrieving funding rates for {symbol}: {e}", exc_info=True); return []
    
    def store_news_articles(self, articles: List[NewsArticle]):
        if not self.conn: logger.error("No DB connection for news articles."); return False
        if not articles: logger.info("No news articles to store."); return True
        try:
            cursor = self.conn.cursor()
            data_to_insert = [
                (
                    article.id, article.url, article.title,
                    self._to_iso_format(article.publication_date),
                    self._to_iso_format(article.retrieval_date),
                    article.source, article.content, article.summary,
                    article.sentiment_score, article.sentiment_label,
                    article.sentiment_confidence,
                    json.dumps(article.key_themes) if article.key_themes else None,
                    json.dumps(article.related_symbols) if article.related_symbols else None,
                    json.dumps(article.raw_llm_response) if article.raw_llm_response else None
                ) for article in articles
            ]
            
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

            if symbol:
                query += " AND related_symbols LIKE ?"
                params.append(f'%"{symbol}"%')
            if start_date:
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
            articles_list = [NewsArticle(**dict(row)) for row in rows]
            logger.info(f"Retrieved {len(articles_list)} news articles with given criteria.")
            return articles_list
        except Exception as e:
            logger.error(f"Error retrieving news articles: {e}", exc_info=True); return []

    def close(self):
        if self.conn: self.conn.close(); logger.info("Database connection closed."); self.conn = None

    def __del__(self): self.close()