from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class RiskManager:
    """
    Manages overall portfolio risk, including drawdown control.
    """
    def __init__(self, settings: Any):
        self.settings = settings
        risk_params = self.settings.get_strategy_params('RiskManagement')
        self.max_portfolio_drawdown_pct = float(risk_params.get('maxportfoliodrawdownpct', 0.20))

        self._peak_equity: float = 0.0
        self._current_drawdown_pct: float = 0.0
        self._trading_halt_active: bool = False

        logger.info(f"RiskManager initialized. Max Portfolio Drawdown: {self.max_portfolio_drawdown_pct * 100:.2f}%")

    def update_portfolio_metrics(self, equity_curve_df: pd.DataFrame, current_timestamp: datetime):
        if equity_curve_df.empty:
            return

        equity_values = equity_curve_df['total_value_usd']
        
        if self._peak_equity == 0.0 and not equity_values.empty:
            self._peak_equity = equity_values.iloc[0]

        latest_total_value = equity_values.iloc[-1]
        self._peak_equity = max(self._peak_equity, latest_total_value)

        if self._peak_equity > 0:
            self._current_drawdown_pct = (self._peak_equity - latest_total_value) / self._peak_equity
        else:
            self._current_drawdown_pct = 0.0

        logger.debug(
            f"RiskManager Metrics Updated: Equity=${latest_total_value:,.2f}, "
            f"Peak=${self._peak_equity:,.2f}, Drawdown={self._current_drawdown_pct:.2%}"
        )

    def check_portfolio_drawdown(self) -> bool:
        """
        Checks if the current portfolio drawdown exceeds the maximum allowed threshold.
        If breached, sets a trading halt flag. Returns True if breached.
        """
        if self._current_drawdown_pct > self.max_portfolio_drawdown_pct:
            if not self._trading_halt_active:
                self._trading_halt_active = True
                logger.critical(
                    f"PORTFOLIO DRAWDOWN LIMIT BREACHED! "
                    f"Current Drawdown: {self._current_drawdown_pct:.2%}, "
                    f"Max Allowed: {self.max_portfolio_drawdown_pct:.2%}. "
                    f"TRADING HALTED."
                )
            return True
        return False

    def is_trading_halted(self) -> bool:
        return self._trading_halt_active