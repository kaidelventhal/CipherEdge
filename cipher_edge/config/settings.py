import configparser
import os
import json
from cipher_edge.app_logger import get_logger
from typing import Dict, List, Optional, Any

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """
    Manages application configuration using config.ini and secrets.ini.
    """
    def __init__(self, config_file_rel_path='config/config.ini', secrets_file_rel_path='config/secrets.ini'):
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
        self.log_file_path: str = self.config.get('General', 'LogFilePath', fallback='logs/cipher_edge.log')

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
        self.slippage_bps: float = self.config.getfloat('Trading', 'SlippageBPS', fallback=2.0)
        
        # Phase 1: Slippage and Precision Settings
        self.BASE_SLIPPAGE_BPS: float = self.config.getfloat('Trading', 'BASE_SLIPPAGE_BPS', fallback=1.0)
        self.AVERAGE_DAILY_VOLUME_FACTOR: float = self.config.getfloat('Trading', 'AVERAGE_DAILY_VOLUME_FACTOR', fallback=0.02)
        self.VOLATILITY_SLIPPAGE_FACTOR: float = self.config.getfloat('Trading', 'VOLATILITY_SLIPPAGE_FACTOR', fallback=0.1)
        self.MIN_TICK_SIZE: float = self.config.getfloat('Trading', 'MIN_TICK_SIZE', fallback=0.5)
        self.PRICE_PRECISION: int = self.config.getint('Trading', 'PRICE_PRECISION', fallback=1)

        # EWMAC Strategy Settings (Example, specific strategies below)
        self.ewmac_short_window: int = self.config.getint('EWMAC_Strategy', 'ShortWindow', fallback=12)
        self.ewmac_long_window: int = self.config.getint('EWMAC_Strategy', 'LongWindow', fallback=26)
        self.ewmac_signal_window: int = self.config.getint('EWMAC_Strategy', 'SignalWindow', fallback=9)
        self.ewmac_atr_period: int = self.config.getint('EWMAC_Strategy', 'atr_period', fallback=14)


        # --- Phase 2: Risk Management Settings (Updated & New) ---
        self.max_portfolio_drawdown_pct: float = self.config.getfloat('RiskManagement', 'MaxPortfolioDrawdownPct', fallback=0.20)
        
        self.position_sizer_type: str = self.config.get('RiskManagement', 'PositionSizer', fallback='FixedFractional')
        self.fixed_fractional_allocation_fraction: float = self.config.getfloat('RiskManagement', 'FixedFractional_AllocationFraction', fallback=0.10)
        self.atr_based_risk_per_trade_fraction: float = self.config.getfloat('RiskManagement', 'ATRBased_RiskPerTradeFraction', fallback=0.01)
        self.atr_based_atr_multiple_for_stop: float = self.config.getfloat('RiskManagement', 'ATRBased_ATRMultipleForStop', fallback=2.0)

        # New Position Sizer Params
        self.optimal_f_win_rate_estimate: float = self.config.getfloat('RiskManagement', 'OptimalF_WinRateEstimate', fallback=0.51)
        self.optimal_f_avg_win_loss_ratio_estimate: float = self.config.getfloat('RiskManagement', 'OptimalF_AvgWinLossRatioEstimate', fallback=1.1)
        self.optimal_f_kelly_fraction: float = self.config.getfloat('RiskManagement', 'OptimalF_KellyFraction', fallback=0.5)

        self.ml_confidence_min_size_factor: float = self.config.getfloat('RiskManagement', 'MLConfidence_MinSizeFactor', fallback=0.5)
        self.ml_confidence_max_size_factor: float = self.config.getfloat('RiskManagement', 'MLConfidence_MaxSizeFactor', fallback=1.5)
        self.ml_confidence_base_allocation_fraction: float = self.config.getfloat('RiskManagement', 'MLConfidence_BaseAllocationFraction', fallback=0.05)


        self.stop_manager_type: str = self.config.get('RiskManagement', 'StopManager_Default', fallback='PercentageBased')
        _sl_pct_str = self.config.get('RiskManagement', 'PercentageStop_LossPct', fallback='0.02')
        self.percentage_stop_loss_pct: Optional[float] = float(_sl_pct_str) if _sl_pct_str and _sl_pct_str.lower() not in ['none', '0', '0.0'] else None
        _tp_pct_str = self.config.get('RiskManagement', 'PercentageStop_TakeProfitPct', fallback='0.05')
        self.percentage_stop_take_profit_pct: Optional[float] = float(_tp_pct_str) if _tp_pct_str and _tp_pct_str.lower() not in ['none', '0', '0.0'] else None
        self.atr_stop_atr_multiple: float = self.config.getfloat('RiskManagement', 'ATRStop_ATRMultiple', fallback=2.0)

        # New Stop Manager Params
        self.parabolic_sar_acceleration_factor: float = self.config.getfloat('RiskManagement', 'ParabolicSAR_AccelerationFactor', fallback=0.02)
        self.parabolic_sar_max_acceleration: float = self.config.getfloat('RiskManagement', 'ParabolicSAR_MaxAcceleration', fallback=0.2)
        
        self.triple_barrier_profit_multiplier: float = self.config.getfloat('RiskManagement', 'TripleBarrier_ProfitMultiplier', fallback=1.5)
        self.triple_barrier_loss_multiplier: float = self.config.getfloat('RiskManagement', 'TripleBarrier_LossMultiplier', fallback=1.0)
        self.triple_barrier_time_limit_days: int = self.config.getint('RiskManagement', 'TripleBarrier_TimeLimitDays', fallback=10)

        # Existing VolatilityBandStopManager params
        self.volatility_band_stop_band_type: str = self.config.get('RiskManagement', 'VolatilityBandStop_BandType', fallback='bollinger')
        self.volatility_band_stop_bb_period: int = self.config.getint('RiskManagement', 'VolatilityBandStop_BB_Period', fallback=20)
        self.volatility_band_stop_bb_std_dev: float = self.config.getfloat('RiskManagement', 'VolatilityBandStop_BB_StdDev', fallback=2.0)
        self.volatility_band_stop_kc_period: int = self.config.getint('RiskManagement', 'VolatilityBandStop_KC_Period', fallback=20)
        self.volatility_band_stop_kc_atr_period: int = self.config.getint('RiskManagement', 'VolatilityBandStop_KC_ATR_Period', fallback=10)
        self.volatility_band_stop_kc_atr_multiplier: float = self.config.getfloat('RiskManagement', 'VolatilityBandStop_KC_ATR_Multiplier', fallback=1.5)
        self.volatility_band_stop_trail_type: str = self.config.get('RiskManagement', 'VolatilityBandStop_TrailType', fallback='none')

        # Pair Trading Sizer (existing)
        self.pair_trading_position_sizer_dollar_neutral: bool = self.config.getboolean('RiskManagement', 'PairTradingPositionSizer_DollarNeutral', fallback=True)


        # --- Phase 2: Portfolio Constructor Settings (Updated & New) ---
        self.portfolio_constructor_type: str = self.config.get('PortfolioConstructor', 'ConstructorType', fallback='Default') # Not used yet, but good to have
        self.asset_allocator_type: str = self.config.get('PortfolioConstructor', 'AssetAllocator', fallback='FixedWeight')
        default_symbol_config_key = f'DefaultAllocation_{self.default_symbol.replace("/", "").replace(":", "")}'
        self.default_allocation_for_symbol: float = self.config.getfloat('PortfolioConstructor', default_symbol_config_key, fallback=1.0)
        
        # New Rebalancing Triggers
        self.rebalance_threshold_pct: float = self.config.getfloat('PortfolioConstructor', 'Rebalance_Threshold_Pct', fallback=0.05)
        
        # New Volatility Targeting Parameters
        self.volatility_targeting_enable: bool = self.config.getboolean('PortfolioConstructor', 'Volatility_Targeting_Enable', fallback=False)
        self.target_portfolio_volatility: float = self.config.getfloat('PortfolioConstructor', 'Target_Portfolio_Volatility', fallback=0.15) # e.g., 15% annual vol target
        self.volatility_targeting_lookback_period: int = self.config.getint('PortfolioConstructor', 'Volatility_Targeting_Lookback_Period', fallback=60) # e.g., 60 bars


        # Optimal F Allocator (existing as BaseAssetAllocator, not specific here)
        self.optimalf_default_win_probability: float = self.config.getfloat('PortfolioConstructor', 'OptimalF_Default_Win_Probability', fallback=0.51)
        self.optimalf_default_payoff_ratio: float = self.config.getfloat('PortfolioConstructor', 'OptimalF_Default_Payoff_Ratio', fallback=1.1)
        self.optimalf_kelly_fraction: float = self.config.getfloat('PortfolioConstructor', 'OptimalF_Kelly_Fraction', fallback=0.25)


        # --- Phase 4: AI News Analysis Settings ---
        self.enable_sentiment_analysis: bool = self.config.getboolean('AI_NewsAnalysis', 'EnableSentimentAnalysis', fallback=True)
        self.use_sentiment_in_models: bool = self.config.getboolean('AI_NewsAnalysis', 'UseSentimentInModels', fallback=True)
        self.sentiment_llm_provider: str = self.config.get('AI_NewsAnalysis', 'SentimentLLMProvider', fallback='VertexAI')
        self.browser_agent_llm_provider: str = self.config.get('AI_NewsAnalysis', 'BrowserAgent_LLMProvider', fallback='VertexAI')
        self.browser_agent_max_steps: int = self.config.getint('AI_NewsAnalysis', 'BrowserAgent_Max_Steps', fallback=20)

        self.enable_sentiment_analysis: bool = self.config.getboolean('AI_NewsAnalysis', 'EnableSentimentAnalysis', fallback=True)
        self.notification_listener_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NotificationListener_Enable', fallback=False)

        self.news_listener_check_interval: int = self.config.getint('AI_NewsAnalysis', 'NewsListener_Check_Interval_Seconds', fallback=300)

        self.sentiment_filter_threshold_long: float = self.config.getfloat('AI_NewsAnalysis', 'SentimentFilter_Threshold_Long', fallback=0.1)
        self.sentiment_filter_threshold_short: float = self.config.getfloat('AI_NewsAnalysis', 'SentimentFilter_Threshold_Short', fallback=-0.1)
        
        self.simulated_sentiment_data_path: Optional[str] = self.config.get('AI_NewsAnalysis', 'SimulatedSentimentDataPath', fallback=None)
        if self.simulated_sentiment_data_path and self.simulated_sentiment_data_path.lower() in ['none', '']:
                self.simulated_sentiment_data_path = None
        if self.simulated_sentiment_data_path and not os.path.isabs(self.simulated_sentiment_data_path):
            path_parts = self.simulated_sentiment_data_path.split(os.sep)
            if path_parts[0] == 'cipher_edge':
                correct_relative_path = os.path.join(*path_parts[1:])
            else:
                correct_relative_path = self.simulated_sentiment_data_path
            self.simulated_sentiment_data_path = os.path.join(PROJECT_ROOT, correct_relative_path)


        self.news_scraper_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NewsScraper_Enable', fallback=True)
        self.notification_listener_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'NotificationListener_Enable', fallback=False)
        self.browser_agent_enable: bool = self.config.getboolean('AI_NewsAnalysis', 'BrowserAgent_Enable', fallback=False)
        
        # VertexAI Settings
        self.vertex_ai_project_id: Optional[str] = self.config.get('VertexAI', 'ProjectID', fallback=None)
        self.vertex_ai_location: Optional[str] = self.config.get('VertexAI', 'Location', fallback=None)
        self.vertex_ai_sentiment_model_name: str = self.config.get('VertexAI', 'SentimentModelName', fallback='gemini-1.5-flash-preview-0514')
        self.vertex_ai_browser_agent_model_name: str = self.config.get('VertexAI', 'BrowserAgentModelName', fallback='gemini-1.5-pro-preview-0514')

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

        try:
            self.phase3_params = self.get_strategy_params('Phase3')
            
            self.PHASE3_SYMBOLS = json.loads(self.phase3_params.get('phase3_symbols', '[]'))
            self.PHASE3_STRATEGIES = json.loads(self.phase3_params.get('phase3_strategies', '[]'))
            self.PHASE3_RISK_MODULES = json.loads(self.phase3_params.get('phase3_risk_modules', '[]'))
            self.PHASE3_STOP_MANAGERS = json.loads(self.phase3_params.get('phase3_stop_managers', '[]'))
            self.PHASE3_TOP_COMBOS_COUNT = int(self.phase3_params.get('phase3_top_combos_count', 5))
            self.PHASE3_COMPUTE_WEIGHTS_METHOD = self.phase3_params.get('phase3_compute_weights_method', 'risk_parity')
            self.PHASE3_COMPOSITE_METHOD = self.phase3_params.get('phase3_composite_method', 'weighted_vote')
            
            self.PHASE3_GRID_SEARCH = {}
            if self.config.has_section('Phase3_GridSearch'):
                for key, value in self.config.items('Phase3_GridSearch'):
                    try:
                        self.PHASE3_GRID_SEARCH[key] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse grid search params for '{key}'. Invalid JSON: {value}")
            
        except (json.JSONDecodeError, configparser.NoSectionError) as e:
            logger.warning(f"Could not load Phase3 settings from config.ini: {e}. Using empty defaults.")
            self.phase3_params = {}
            self.PHASE3_SYMBOLS = []
            self.PHASE3_STRATEGIES = []
            self.PHASE3_RISK_MODULES = []
            self.PHASE3_STOP_MANAGERS = []
            self.PHASE3_TOP_COMBOS_COUNT = 5
            self.PHASE3_COMPUTE_WEIGHTS_METHOD = "risk_parity"
            self.PHASE3_COMPOSITE_METHOD = "weighted_vote"
            self.PHASE3_GRID_SEARCH = {}

    def get_strategy_params(self, strategy_or_component_name: str) -> dict:
        """
        Retrieves parameters for a given strategy or component section name.
        Example section names: EWMAC_Strategy, LightGBM_Forecaster, MLForecaster_Strategy
        FIX: Made matching case-insensitive and underscore-insensitive.
        """
        params = {}
        found_section = None
        cleaned_name = strategy_or_component_name.lower().replace('_', '')
        
        for section in self.config.sections():
            cleaned_section = section.lower().replace('_', '')
            if cleaned_section == cleaned_name:
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