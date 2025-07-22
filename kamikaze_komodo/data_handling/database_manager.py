# kamikaze_komodo/data_handling/database_manager.py
import sqlite3
from typing import List, Optional, Dict, Any
from kamikaze_komodo.core.models import BarData, NewsArticle
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone, UTC 
import json

logger = get_logger(__name__)

class DatabaseManager:
    """
    Manages local storage of core data (OHLCV, News).
    Indicators are not stored here; they are calculated on-the-fly for backtests.
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
                    funding_rate REAL,
                    PRIMARY KEY (timestamp, symbol, timeframe)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL,
                    PRIMARY KEY (timestamp, symbol)
                )
            """)
            self.conn.commit()
            logger.info("Core tables checked/created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")

    def _to_iso_format(self, dt: Optional[datetime]) -> Optional[str]:
        if dt is None: return None
        if dt.tzinfo is None: dt = dt.replace(tzinfo=UTC)
        else: dt = dt.astimezone(UTC)
        return dt.isoformat()

    def _from_iso_format(self, iso_str: Optional[str]) -> Optional[datetime]:
        if iso_str is None: return None
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None: return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            return None

    def store_bar_data(self, bar_data_list: List[BarData]):
        if not self.conn: return
        if not bar_data_list: return
        try:
            cursor = self.conn.cursor()
            data_to_insert = [
                (
                    self._to_iso_format(bd.timestamp), bd.symbol, bd.timeframe,
                    bd.open, bd.high, bd.low, bd.close, bd.volume,
                    bd.funding_rate
                ) for bd in bar_data_list
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO bar_data 
                (timestamp, symbol, timeframe, open, high, low, close, volume, funding_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) 
            """, data_to_insert) 
            self.conn.commit()
            logger.info(f"Stored/Replaced {len(data_to_insert)} bar data entries.")
        except Exception as e:
            logger.error(f"Error storing bar data: {e}", exc_info=True)

    def retrieve_bar_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[BarData]:
        if not self.conn: return []
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM bar_data WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]
            if start_date: query += " AND timestamp >= ?"; params.append(self._to_iso_format(start_date))
            if end_date: query += " AND timestamp <= ?"; params.append(self._to_iso_format(end_date))
            query += " ORDER BY timestamp ASC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [
                BarData(
                    timestamp=self._from_iso_format(row['timestamp']),
                    **{k: row[k] for k in row.keys() if k != 'timestamp'}
                ) for row in rows if self._from_iso_format(row['timestamp'])
            ]
        except Exception as e:
            logger.error(f"Error retrieving bar data: {e}", exc_info=True)
            return []

    def store_funding_rates(self, funding_rate_list: List[Dict[str, Any]]):
        if not self.conn or not funding_rate_list:
            return
        try:
            cursor = self.conn.cursor()
            data_to_insert = [
                (
                    self._to_iso_format(datetime.fromtimestamp(fr['timestamp'] / 1000, tz=timezone.utc)),
                    fr['symbol'],
                    fr['fundingRate']
                ) for fr in funding_rate_list
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO funding_rates (timestamp, symbol, funding_rate)
                VALUES (?, ?, ?)
            """, data_to_insert)
            self.conn.commit()
            logger.info(f"Stored/Replaced {len(data_to_insert)} funding rate entries.")
        except Exception as e:
            logger.error(f"Error storing funding rates: {e}", exc_info=True)

    def retrieve_funding_rates(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        if not self.conn: return []
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM funding_rates WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
            params = (symbol, self._to_iso_format(start_date), self._to_iso_format(end_date))
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                {
                    "timestamp": self._from_iso_format(row['timestamp']).timestamp() * 1000,
                    "symbol": row['symbol'],
                    "fundingRate": row['funding_rate']
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error retrieving funding rates: {e}", exc_info=True)
            return []

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
            self.conn = None

    def __del__(self):
        self.close()