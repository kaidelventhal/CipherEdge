# CipherEdge

## Overview/Mission Statement

CipherEdge is an AI-driven quantitative trading program designed to discover, backtest, and deploy a portfolio of diverse trading strategies. The system's primary mission is to achieve consistent profitability by dynamically adapting to changing market conditions through a robust, multi-strategy approach.

## Core Features & Architecture

* **AI-Driven Strategy Discovery**: Utilizes parallel processing and advanced validation techniques like Walk-Forward Optimization to discover optimal strategy configurations across multiple assets and timeframes.
* **Multi-Strategy Portfolio Management**: Implements a higher-level portfolio constructor that can manage capital allocation between several uncorrelated strategies, aggregating their signals into a unified set of trades.
* **Advanced Backtesting Engine**: Goes beyond simple backtesting to include performance analysis with metrics like the Deflated Sharpe Ratio and Monte Carlo simulations to assess the robustness of strategy equity curves.
* **Comprehensive Feature Engineering**: Generates a rich feature set for machine learning models, including advanced technical indicators, market structure features, and sentiment analysis data.
* **Live Trading Orchestration**: A fully asynchronous, event-driven engine for live paper or real trading, connecting to exchange data feeds via WebSockets.
* **Modular Design**: The system is broken down into distinct modules for data handling, strategy implementation, backtesting, risk management, and portfolio construction, allowing for easy extension and maintenance.

## Algorithms & Techniques

* **Backtesting**: The system employs a vectorized backtesting engine for speed during the initial discovery phase and a more detailed event-driven engine for portfolio-level simulations.
* **Strategies**: Includes a library of classic and modern strategies:
    * Trend Following (EWMAC, Ehlers' Instantaneous Trendline)
    * Mean Reversion (Bollinger Bands)
    * Breakout (Volatility Squeeze, Bollinger Band Breakout)
    * Machine Learning (LightGBM, XGBoost, LSTM Forecasters)
    * Ensemble & Composite Strategies
* **Risk Management**: Features multiple layers of risk control:
    * **Position Sizing**: Fixed Fractional, ATR-Based, Optimal F, and ML Confidence-based sizing.
    * **Stop Management**: Percentage-based, ATR, Parabolic SAR, and Triple-Barrier methods.
    * **Portfolio-Level**: Max drawdown limits and volatility targeting.