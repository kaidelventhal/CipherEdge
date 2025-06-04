# kamikaze_komodo/config/settings.py
# Updated to include Phase 3 and Phase 4 settings

import configparser
import os
from kamikaze_komodo.app_logger import get_logger
from typing import Dict, List, Optional, Any # Added Any for type hinting

logger = get_logger(__name__)

class Config:
    """
    Manages application configuration using config.ini and secrets.ini.
    """
    def __init__(self, config_file_rel_path='config/config.ini', secrets_file_rel_path='config/secrets.ini'):
        self.config = configparser.ConfigParser()
        self.secrets = configparser.ConfigParser()

        script_dir = os.path.dirname(os.path.abspath(__file__)) # .../kamikaze_komodo/config
        project_module_dir = os.path.dirname(script_dir) # .../kamikaze_komodo

        self.config_file_path = os.path.join(project_module_dir, config_file_rel_path)
        self.secrets_file_path = os.path.join(project_module_dir, secrets_file_rel_path)

        if not os.path.exists(self.config_file_path):
            logger.error(f"Config file not found: {self.config_file_path}")
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")
        if not os.path.exists(self.secrets_file_path):
            logger.warning(f"Secrets file not found: {self.secrets_file_path}. Some features might be unavailable.")

        self.config.read(self.config_file_path)
        self.secrets.read(self.secrets_file_path)

        # General Settings
        self.log_level: str = self.config.get('General', 'LogLevel', fallback='INFO')
        self.log_file_path: str = self.config.get('General', 'LogFilePath', fallback='logs/kamikaze_komodo.log')

        # API Settings
        self.exchange_id_to_use: str = self.config.get('API', 'ExchangeID', fallback='kraken')
        self.kraken_api_key: Optional[str] = self.secrets.get('KRAKEN_API', 'API_KEY', fallback=None)
        self.kraken_secret_key: Optional[str] = self.secrets.get('KRAKEN_API', 'SECRET_KEY', fallback=None)
        self.kraken_testnet: bool = self.config.getboolean('API', 'KrakenTestnet', fallback=True)

        # Data Fetching Settings
        self.default_symbol: str = self.config.get('DataFetching', 'DefaultSymbol', fallback='BTC/USD')
        self.default_timeframe: str = self.config.get('DataFetching', 'DefaultTimeframe', fallback='1h')
        self.historical_data_days: int = self.config.getint('DataFetching', 'HistoricalDataDays', fallback=365)
        self.data_fetch_limit_per_call: int = self.config.getint('DataFetching', 'DataFetchLimitPerCall', fallback=500)

        # Trading Settings
        self.max_portfolio_risk: float = self.config.getfloat('Trading', 'MaxPortfolioRisk', fallback=0.02)
        self.default_leverage: float = self.config.getfloat('Trading', 'DefaultLeverage', fallback=1.0)
        self.commission_bps: float = self.config.getfloat('Trading', 'CommissionBPS', fallback=10.0)

        # EWMAC Strategy Settings
        self.ewmac_short_window: int = self.config.getint('EWMAC_Strategy', 'ShortWindow', fallback=12)
        self.ewmac_long_window: int = self.config.getint('EWMAC_Strategy', 'LongWindow', fallback=26)
        self.ewmac_signal_window: int = self.config.getint('EWMAC_Strategy', 'SignalWindow', fallback=9) # For MACD part of EWMAC if used

        # --- Phase 3: Risk Management Settings ---
        self.position_sizer_type: str = self.config.get('RiskManagement', 'PositionSizer', fallback='FixedFractional')
        self.fixed_fractional_allocation_fraction: float = self.config.getfloat('RiskManagement', 'FixedFractional_AllocationFraction', fallback=0.10)
        self.atr_based_risk_per_trade_fraction: float = self.config.getfloat('RiskManagement', 'ATRBased_RiskPerTradeFraction', fallback=0.01)
        self.atr_based_atr_multiple_for_stop: float = self.config.getfloat('RiskManagement', 'ATRBased_ATRMultipleForStop', fallback=2.0)

        self.stop_manager_type: str = self.config.get('RiskManagement', 'StopManager_Default', fallback='PercentageBased')
        _sl_pct_str = self.config.get('RiskManagement', 'PercentageStop_LossPct', fallback='0.02')
        self.percentage_stop_loss_pct: Optional[float] = float(_sl_pct_str) if _sl_pct_str and _sl_pct_str.lower() not in ['none', '0', '0.0'] else None

        _tp_pct_str = self.config.get('RiskManagement', 'PercentageStop_TakeProfitPct', fallback='0.05')
        self.percentage_stop_take_profit_pct: Optional[float] = float(_tp_pct_str) if _tp_pct_str and _tp_pct_str.lower() not in ['none', '0', '0.0'] else None

        self.atr_stop_atr_multiple: float = self.config.getfloat('RiskManagement', 'ATRStop_ATRMultiple', fallback=2.0)

        # --- Phase 3: Portfolio Constructor Settings ---
        self.asset_allocator_type: str = self.config.get('PortfolioConstructor', 'AssetAllocator', fallback='FixedWeight')
        self.default_allocation_btcusd: float = self.config.getfloat('PortfolioConstructor', f'DefaultAllocation_{self.default_symbol.replace("/","").replace(":","")}', fallback=1.0)
        self.rebalancer_deviation_threshold: float = self.config.getfloat('PortfolioConstructor', 'Rebalancer_DeviationThreshold', fallback=0.05)

        # --- Phase 4: AI News Analysis Settings ---
        self.enable_sentiment_analysis: bool = self.config.getboolean('AI_NewsAnalysis', 'EnableSentimentAnalysis', fallback=True)
        self.sentiment_llm_model: str = self.config.get('AI_NewsAnalysis', 'SentimentLLMModel', fallback='gemma:7b')
        self.ollama_base_url: Optional[str] = self.config.get('AI_NewsAnalysis', 'OllamaBaseURL', fallback="http://localhost:11434")
        if self.ollama_base_url and self.ollama_base_url.lower() in ['none', '']: self.ollama_base_url = None

        self.sentiment_filter_threshold_long: float = self.config.getfloat('AI_NewsAnalysis', 'SentimentFilter_Threshold_Long', fallback=0.1)
        self.sentiment_filter_threshold_short: float = self.config.getfloat('AI_NewsAnalysis', 'SentimentFilter_Threshold_Short', fallback=-0.1)
        self.simulated_sentiment_data_path: Optional[str] = self.config.get('AI_NewsAnalysis', 'SimulatedSentimentDataPath', fallback=None)
        if self.simulated_sentiment_data_path and self.simulated_sentiment_data_path.lower() in ['none', '']: self.simulated_sentiment_data_path = None
        # Resolve simulated sentiment data path relative to project module dir if it's a relative path
        if self.simulated_sentiment_data_path and not os.path.isabs(self.simulated_sentiment_data_path):
            self.simulated_sentiment_data_path = os.path.join(project_module_dir, self.simulated_sentiment_data_path)


        self.news_scraper_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NewsScraper_Enable', fallback=True)
        self.rss_feed_cointelegraph: str = self.config.get('AI_NewsAnalysis', 'RSSFeed_CoinTelegraph', fallback='https://cointelegraph.com/rss')
        self.rss_feed_seeking_alpha: str = self.config.get('AI_NewsAnalysis', 'RSSFeed_SeekingAlpha', fallback='https://seekingalpha.com/market_currents.xml')

        self.notification_listener_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NotificationListener_Enable', fallback=False)
        self.browser_agent_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'BrowserAgent_Enable', fallback=False)
        self.browser_agent_llm_model: str = self.config.get('AI_NewsAnalysis', 'BrowserAgent_LLMModel', fallback='gemma:7b')
        self.browser_agent_max_steps: int = self.config.getint('AI_NewsAnalysis', 'BrowserAgent_MaxSteps', fallback=15)

    def get_strategy_params(self, strategy_name: str) -> dict:
        params = {}
        section_name = f"{strategy_name}_Strategy" # Standardize section names like EWMAC_Strategy
        if self.config.has_section(section_name):
            params = dict(self.config.items(section_name))
            for key, value in params.items():
                # Convert to int, float, bool, None where appropriate
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    params[key] = int(value)
                else:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        if value.lower() == 'true': params[key] = True
                        elif value.lower() == 'false': params[key] = False
                        elif value.lower() == 'none': params[key] = None
        else:
            logger.warning(f"No specific configuration section found for strategy: {section_name}. Using defaults or globally passed params.")
        return params

    def get_news_scraper_config(self) -> Dict[str, Any]:
        cfg = {"rss_feeds": [], "websites": []}
        if self.news_scraper_enable:
            if self.rss_feed_cointelegraph:
                cfg["rss_feeds"].append({"name": "CoinTelegraph RSS", "url": self.rss_feed_cointelegraph})
            if self.rss_feed_seeking_alpha:
                cfg["rss_feeds"].append({"name": "SeekingAlpha Market Currents", "url": self.rss_feed_seeking_alpha})
            # Example to read additional websites if configured in a more generic way in config.ini
            for key, value in self.config.items('AI_NewsAnalysis'):
                if key.startswith("websitescrape_") and key.endswith("_url"):
                    site_id = key.replace("websitescrape_", "").replace("_url", "")
                    site_name = self.config.get('AI_NewsAnalysis', f'WebsiteScrape_{site_id}_Name', fallback=site_id.capitalize())
                    cfg["websites"].append({"name": site_name, "url": value})
        return cfg

try:
    settings = Config(config_file_rel_path='config/config.ini', secrets_file_rel_path='config/secrets.ini')
except FileNotFoundError as e:
    logger.critical(f"Could not initialize settings due to missing configuration file: {e}")
    settings = None
except Exception as e_global:
    logger.critical(f"Failed to initialize Config object: {e_global}", exc_info=True)
    settings = None

if settings and (not settings.kraken_api_key or "YOUR_API_KEY" in str(settings.kraken_api_key).upper() or "D27PYGI95TLS" in str(settings.kraken_api_key).upper()): # Updated placeholder check
    logger.warning(f"API Key for '{settings.exchange_id_to_use}' appears to be a placeholder or is not configured in secrets.ini. Authenticated interaction will be limited.")