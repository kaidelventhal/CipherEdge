# kamikaze_komodo/config/settings.py
import configparser
import os
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class Config:
    """
    Manages application configuration using config.ini and secrets.ini.
    """
    def __init__(self, config_file='config/config.ini', secrets_file='config/secrets.ini'):
        self.config = configparser.ConfigParser()
        self.secrets = configparser.ConfigParser()

        # Determine absolute paths for config files
        # Assumes this settings.py is in kamikaze_komodo/config/
        # So base_dir should be kamikaze_komodo/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        self.config_file_path = os.path.join(base_dir, config_file)
        self.secrets_file_path = os.path.join(base_dir, secrets_file)
        
        if not os.path.exists(self.config_file_path):
            logger.error(f"Config file not found: {self.config_file_path}")
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")
        if not os.path.exists(self.secrets_file_path):
            logger.warning(f"Secrets file not found: {self.secrets_file_path}. Some features might be unavailable.")
            # Not raising FileNotFoundError for secrets to allow operation with public endpoints
            # if API keys are not strictly necessary for all operations.

        self.config.read(self.config_file_path)
        self.secrets.read(self.secrets_file_path)

        # General Settings
        self.log_level: str = self.config.get('General', 'LogLevel', fallback='INFO')
        self.log_file_path: str = self.config.get('General', 'LogFilePath', fallback='logs/kamikaze_komodo.log')

        # API Settings
        self.exchange_id_to_use: str = self.config.get('API', 'ExchangeID', fallback='kraken') # <-- MODIFIED/ADDED
        self.kraken_api_key: str | None = self.secrets.get('KRAKEN_API', 'API_KEY', fallback=None)
        self.kraken_secret_key: str | None = self.secrets.get('KRAKEN_API', 'SECRET_KEY', fallback=None)
        self.kraken_testnet: bool = self.config.getboolean('API', 'KrakenTestnet', fallback=True)


        # Data Fetching Settings
        self.default_symbol: str = self.config.get('DataFetching', 'DefaultSymbol', fallback='BTC/USD')
        self.default_timeframe: str = self.config.get('DataFetching', 'DefaultTimeframe', fallback='1h')
        self.historical_data_days: int = self.config.getint('DataFetching', 'HistoricalDataDays', fallback=365)

        # Trading Settings
        self.max_portfolio_risk: float = self.config.getfloat('Trading', 'MaxPortfolioRisk', fallback=0.02)
        self.default_leverage: float = self.config.getfloat('Trading', 'DefaultLeverage', fallback=1.0)
        
        # Strategy Specific Settings (Example for EWMAC)
        self.ewmac_short_window: int = self.config.getint('EWMAC_Strategy', 'ShortWindow', fallback=12)
        self.ewmac_long_window: int = self.config.getint('EWMAC_Strategy', 'LongWindow', fallback=26)


    def get_strategy_params(self, strategy_name: str) -> dict:
        """
        Retrieves parameters for a specific strategy section from config.ini.
        """
        params = {}
        if self.config.has_section(strategy_name):
            params = dict(self.config.items(strategy_name))
            # Convert to appropriate types if necessary, e.g., int, float
            for key, value in params.items():
                # Attempt to convert to int if all digits
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    params[key] = int(value)
                else:
                    # Attempt to convert to float
                    try:
                        params[key] = float(value)
                    except ValueError:
                        # If not int or float, keep as string
                        pass 
        else:
            logger.warning(f"No configuration section found for strategy: {strategy_name}")
        return params

# Global config instance
try:
    settings = Config()
except FileNotFoundError as e:
    logger.critical(f"Could not initialize settings due to missing configuration file: {e}")
    settings = None 
except Exception as e_global: # Catch any other parsing errors during Config init
    logger.critical(f"Failed to initialize Config object: {e_global}", exc_info=True)
    settings = None


if settings and (not settings.kraken_api_key or "YOUR_API_KEY" in str(settings.kraken_api_key).upper() or "M9EXKSAN" in str(settings.kraken_api_key).upper()):
    # Check for common placeholders, adjust "M9EXKSAN" if your placeholder is different
    logger.warning(f"API Key for '{settings.exchange_id_to_use}' appears to be a placeholder or is not configured in secrets.ini. Authenticated interaction will be limited.")