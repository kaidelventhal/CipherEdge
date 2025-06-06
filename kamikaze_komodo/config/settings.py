# FILE: kamikaze_komodo/config/settings.py
import configparser
import os
from kamikaze_komodo.app_logger import get_logger
from typing import Dict, List, Optional, Any

logger = get_logger(__name__)

# Define a single, reliable project root that other modules can import.
# This file is in .../kamikaze_komodo/config/, so THREE levels up is the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    """
    Manages application configuration using config.ini and secrets.ini.
    """
    def __init__(self, config_file_rel_path='kamikaze_komodo/config/config.ini', secrets_file_rel_path='kamikaze_komodo/config/secrets.ini'):
        self.config = configparser.ConfigParser()
        self.secrets = configparser.ConfigParser()

        self.config_file_path = os.path.join(PROJECT_ROOT, config_file_rel_path)
        self.secrets_file_path = os.path.join(PROJECT_ROOT, secrets_file_rel_path)

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
        self.exchange_id_to_use: str = self.config.get('API', 'ExchangeID', fallback='krakenfutures')
        self.kraken_api_key: Optional[str] = self.secrets.get('KRAKEN_API', 'API_KEY', fallback=None)
        self.kraken_secret_key: Optional[str] = self.secrets.get('KRAKEN_API', 'SECRET_KEY', fallback=None)
        self.kraken_testnet: bool = self.config.getboolean('API', 'KrakenTestnet', fallback=True)

        # Data Fetching Settings
        self.default_symbol: str = self.config.get('DataFetching', 'DefaultSymbol', fallback='PF_XBTUSD')
        self.default_timeframe: str = self.config.get('DataFetching', 'DefaultTimeframe', fallback='4h')
        self.historical_data_days: int = self.config.getint('DataFetching', 'HistoricalDataDays', fallback=365)
        self.data_fetch_limit_per_call: int = self.config.getint('DataFetching', 'DataFetchLimitPerCall', fallback=500)

        # Trading Settings
        self.max_portfolio_risk: float = self.config.getfloat('Trading', 'MaxPortfolioRisk', fallback=0.02)
        self.default_leverage: float = self.config.getfloat('Trading', 'DefaultLeverage', fallback=1.0)
        self.commission_bps: float = self.config.getfloat('Trading', 'CommissionBPS', fallback=10.0)

        # EWMAC Strategy Settings (Example, specific strategies below)
        self.ewmac_short_window: int = self.config.getint('EWMAC_Strategy', 'ShortWindow', fallback=12)
        self.ewmac_long_window: int = self.config.getint('EWMAC_Strategy', 'LongWindow', fallback=26)
        self.ewmac_signal_window: int = self.config.getint('EWMAC_Strategy', 'SignalWindow', fallback=9)
        self.ewmac_atr_period: int = self.config.getint('EWMAC_Strategy', 'atr_period', fallback=14)


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
        default_symbol_config_key = f'DefaultAllocation_{self.default_symbol.replace("/", "").replace(":", "")}'
        self.default_allocation_for_symbol: float = self.config.getfloat('PortfolioConstructor', default_symbol_config_key, fallback=1.0)
        self.rebalancer_deviation_threshold: float = self.config.getfloat('PortfolioConstructor', 'Rebalancer_DeviationThreshold', fallback=0.05)

        # --- Phase 4: AI News Analysis Settings ---
        self.enable_sentiment_analysis: bool = self.config.getboolean('AI_NewsAnalysis', 'EnableSentimentAnalysis', fallback=True)
        self.sentiment_llm_provider: str = self.config.get('AI_NewsAnalysis', 'SentimentLLMProvider', fallback='VertexAI')
        self.browser_agent_llm_provider: str = self.config.get('AI_NewsAnalysis', 'BrowserAgent_LLMProvider', fallback='VertexAI')
        self.browser_agent_max_steps: int = self.config.getint('AI_NewsAnalysis', 'BrowserAgent_Max_Steps', fallback=20)


        self.sentiment_filter_threshold_long: float = self.config.getfloat('AI_NewsAnalysis', 'SentimentFilter_Threshold_Long', fallback=0.1)
        self.sentiment_filter_threshold_short: float = self.config.getfloat('AI_NewsAnalysis', 'SentimentFilter_Threshold_Short', fallback=-0.1)
        
        self.simulated_sentiment_data_path: Optional[str] = self.config.get('AI_NewsAnalysis', 'SimulatedSentimentDataPath', fallback=None)
        if self.simulated_sentiment_data_path and self.simulated_sentiment_data_path.lower() in ['none', '']:
            self.simulated_sentiment_data_path = None
        
        if self.simulated_sentiment_data_path and not os.path.isabs(self.simulated_sentiment_data_path):
            self.simulated_sentiment_data_path = os.path.join(PROJECT_ROOT, self.simulated_sentiment_data_path)


        self.news_scraper_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NewsScraper_Enable', fallback=True)
        self.notification_listener_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NotificationListener_Enable', fallback=False)
        self.browser_agent_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'BrowserAgent_Enable', fallback=False)
        
        # VertexAI Settings
        self.vertex_ai_project_id: Optional[str] = self.config.get('VertexAI', 'ProjectID', fallback=None)
        self.vertex_ai_location: Optional[str] = self.config.get('VertexAI', 'Location', fallback=None)
        self.vertex_ai_sentiment_model_name: str = self.config.get('VertexAI', 'SentimentModelName', fallback='gemini-2.5-flash-preview-05-20')
        self.vertex_ai_browser_agent_model_name: str = self.config.get('VertexAI', 'BrowserAgentModelName', fallback='gemini-2.5-flash-preview-05-20')

        if self.vertex_ai_project_id and self.vertex_ai_project_id.lower() == 'your-gcp-project-id':
            logger.warning("Vertex AI ProjectID is set to 'your-gcp-project-id'. Please update it in config.ini.")
            self.vertex_ai_project_id = None

        self.rss_feeds: List[Dict[str, str]] = []
        if self.config.has_section('AI_NewsAnalysis'):
            for key, value in self.config.items('AI_NewsAnalysis'):
                clean_key = key.strip().lower()
                if clean_key.startswith("rssfeed_"):
                    feed_name_part = clean_key.replace("rssfeed_", "")
                    feed_name = feed_name_part.replace("_", " ").title()
                    self.rss_feeds.append({"name": feed_name, "url": value})
        if not self.rss_feeds:
            logger.warning("No RSS feeds configured in config.ini under [AI_NewsAnalysis] with 'RSSFeed_' prefix.")


    def get_strategy_params(self, strategy_or_component_name: str) -> dict:
        """
        Retrieves parameters for a given strategy or component section name.
        Example section names: EWMAC_Strategy, LightGBM_Forecaster, MLForecaster_Strategy
        """
        params = {}
        found_section = None
        for section in self.config.sections():
            if section.lower() == strategy_or_component_name.lower():
                found_section = section
                break
        
        if found_section and self.config.has_section(found_section):
            params = dict(self.config.items(found_section))
            for key, value in params.items():
                original_value = value
                try:
                    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        params[key] = int(value)
                    else:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            if value.lower() == 'true': params[key] = True
                            elif value.lower() == 'false': params[key] = False
                            elif value.lower() in ['none', '']: params[key] = None
                            else:
                                params[key] = original_value
                except Exception as e:
                    logger.debug(f"Could not auto-convert param '{key}' with value '{original_value}' in section '{found_section}'. Kept as string. Error: {e}")
                    params[key] = original_value
        else:
            logger.warning(f"No specific configuration section found for: {strategy_or_component_name}. Using defaults or globally passed params.")
        
        # FIX: Correct attribute names to match what's defined in __init__
        if 'sentimentfilter_long_threshold' not in params:
            params['sentimentfilter_long_threshold'] = self.sentiment_filter_threshold_long
        if 'sentimentfilter_short_threshold' not in params:
            params['sentimentfilter_short_threshold'] = self.sentiment_filter_threshold_short
            
        return params

    def get_news_scraper_config(self) -> Dict[str, Any]:
        cfg = {"rss_feeds": self.rss_feeds, "websites": []}
        return cfg


try:
    settings = Config()
except FileNotFoundError as e:
    logger.critical(f"Could not initialize settings due to missing configuration file: {e}")
    settings = None # type: ignore
except Exception as e_global:
    logger.critical(f"Failed to initialize Config object: {e_global}", exc_info=True)
    settings = None # type: ignore

if settings and (not settings.kraken_api_key or "YOUR_API_KEY" in str(settings.kraken_api_key).upper() or "D27PYGI95TLS" in str(settings.kraken_api_key).upper()):
    logger.warning(f"API Key for '{settings.exchange_id_to_use}' appears to be a placeholder or is not configured in secrets.ini. Authenticated interaction will be limited/simulated.")