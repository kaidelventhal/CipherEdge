# kamikaze_komodo/data_handling/database_manager.py
import sqlite3
from typing import List, Optional
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
# from kamikaze_komodo.config.settings import settings # settings is not used directly in this file
from datetime import datetime, timezone, UTC # UTC is preferred for Python 3.11+

logger = get_logger(__name__)

class DatabaseManager:
    """
    Manages local storage of data (initially SQLite).
    Timestamps are stored as ISO 8601 TEXT to ensure reliable conversion.
    """
    def __init__(self, db_name: str = "kamikaze_komodo_data.db"):
        self.db_name = db_name
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establishes a connection to the SQLite database."""
        try:
            # MODIFIED: Using PARSE_COLNAMES only, to rely on Python for type conversion from TEXT
            self.conn = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_COLNAMES)
            self.conn.row_factory = sqlite3.Row 
            logger.info(f"Successfully connected to database: {self.db_name}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_name}: {e}")
            self.conn = None

    def _create_tables(self):
        """Creates necessary tables if they don't exist."""
        if not self.conn:
            logger.error("Cannot create tables, no database connection.")
            return

        try:
            cursor = self.conn.cursor()
            # BarData Table
            # MODIFIED: timestamp column is now TEXT
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
                    PRIMARY KEY (timestamp, symbol, timeframe)
                )
            """)
            # NewsArticle Table (Example for future use - timestamp columns should also be TEXT if added)
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
                    sentiment_label TEXT
                )
            """)
            # Trades Table (Example for future use - timestamp columns should also be TEXT if added)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_order_id TEXT,
                    exit_order_id TEXT,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    amount REAL NOT NULL,
                    entry_timestamp TEXT NOT NULL, 
                    exit_timestamp TEXT, 
                    pnl REAL,
                    pnl_percentage REAL,
                    commission REAL DEFAULT 0.0,
                    result TEXT,
                    notes TEXT
                )
            """)
            self.conn.commit()
            logger.info("Tables checked/created successfully (timestamps as TEXT).")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")

    def store_bar_data(self, bar_data_list: List[BarData]):
        """
        Stores a list of BarData objects into the database.
        Timestamps are converted to ISO 8601 strings.
        """
        if not self.conn:
            logger.error("Cannot store bar data, no database connection.")
            return False
        if not bar_data_list:
            logger.info("No bar data provided to store.")
            return True

        try:
            cursor = self.conn.cursor()
            data_to_insert = []
            for bd in bar_data_list:
                # Ensure timestamp is UTC and then convert to ISO 8601 string
                ts_aware = bd.timestamp
                if ts_aware.tzinfo is None or ts_aware.tzinfo.utcoffset(ts_aware) is None:
                    # If somehow naive, assume UTC (though BarData should be UTC from fetcher)
                    ts_aware = ts_aware.replace(tzinfo=UTC) 
                else:
                    ts_aware = ts_aware.astimezone(UTC) # Convert to UTC if it's some other timezone

                data_to_insert.append((
                    ts_aware.isoformat(), # MODIFIED: Store as ISO string
                    bd.symbol, bd.timeframe,
                    bd.open, bd.high, bd.low, bd.close, bd.volume
                ))
            
            cursor.executemany("""
                INSERT OR IGNORE INTO bar_data 
                (timestamp, symbol, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            self.conn.commit()
            logger.info(f"Stored/Ignored {len(data_to_insert)} bar data entries. ({cursor.rowcount} actually inserted/replaced)")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing bar data: {e}")
            return False
        except Exception as e_gen: # Catch other potential errors like during isoformat
            logger.error(f"Generic error storing bar data: {e_gen}", exc_info=True)
            return False


    def retrieve_bar_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[BarData]:
        """
        Retrieves BarData objects from the database for a given symbol and timeframe.
        Timestamps are converted from ISO 8601 strings back to datetime objects.
        """
        if not self.conn:
            logger.error("Cannot retrieve bar data, no database connection.")
            return []

        try:
            cursor = self.conn.cursor()
            # MODIFIED: Parameters for timestamp comparison need to be ISO strings
            query = "SELECT timestamp, open, high, low, close, volume, symbol, timeframe FROM bar_data WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]

            if start_date:
                query += " AND timestamp >= ?"
                # Ensure start_date is UTC then format to ISO string for comparison
                start_date_aware = start_date
                if start_date_aware.tzinfo is None or start_date_aware.tzinfo.utcoffset(start_date_aware) is None:
                    start_date_aware = start_date_aware.replace(tzinfo=UTC)
                else:
                    start_date_aware = start_date_aware.astimezone(UTC)
                params.append(start_date_aware.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                # Ensure end_date is UTC then format to ISO string for comparison
                end_date_aware = end_date
                if end_date_aware.tzinfo is None or end_date_aware.tzinfo.utcoffset(end_date_aware) is None:
                    end_date_aware = end_date_aware.replace(tzinfo=UTC)
                else:
                    end_date_aware = end_date_aware.astimezone(UTC)
                params.append(end_date_aware.isoformat())
            
            query += " ORDER BY timestamp ASC"

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            
            bar_data_list = []
            for row in rows:
                # MODIFIED: Parse timestamp string using fromisoformat and ensure it's UTC
                # datetime.fromisoformat correctly handles strings like 'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'
                # or 'YYYY-MM-DD HH:MM:SS.ffffff' (if it was stored without TZ, though we store with)
                # If it was stored with +00:00, fromisoformat makes it timezone-aware.
                # If it was stored as naive UTC, we'll make it aware.
                iso_timestamp_str = row['timestamp']
                try:
                    # fromisoformat handles timezone information like +00:00 if present
                    dt_object = datetime.fromisoformat(iso_timestamp_str)
                    # Ensure it is UTC. If it was parsed as naive, assume UTC.
                    # If it was parsed with offset, convert to UTC for consistency.
                    if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
                        dt_object = dt_object.replace(tzinfo=UTC)
                    else:
                        dt_object = dt_object.astimezone(UTC)
                except ValueError as ve:
                    logger.error(f"Could not parse timestamp '{iso_timestamp_str}' from DB: {ve}")
                    continue # Skip this problematic row

                bar_data_list.append(BarData(
                    timestamp=dt_object,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    symbol=row['symbol'],
                    timeframe=row['timeframe']
                ))
            logger.info(f"Retrieved {len(bar_data_list)} bar data entries for {symbol} ({timeframe}).")
            return bar_data_list
        except sqlite3.Error as e:
            logger.error(f"Error retrieving bar data for {symbol} ({timeframe}): {e}")
            return []
        except Exception as e_gen: # Catch other potential errors
            logger.error(f"Generic error retrieving bar data for {symbol} ({timeframe}): {e_gen}", exc_info=True)
            return []

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
            self.conn = None

    def __del__(self):
        """Ensures the database connection is closed when the object is garbage collected."""
        self.close()