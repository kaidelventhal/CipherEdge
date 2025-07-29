## Overview

CipherEdge is an AI quantitative trading program I built for crypto markets. It's designed to automate trading strategies for crypto futures and options on Linux, using a multi-strategy and AI-based approach to adapt to changing market conditions.

I wanted to make something that could mix traditional trading with real-time news. The crypto market moves a lot based on news and how people feel, so regular strategies can't always keep up. This program uses a local LLM to read RSS news feeds and react quickly for trades.


## Trading Strategies

* EWMAC Strategy
* Bollinger Band Mean Reversion
* Bollinger Band Breakout
* Volatility Squeeze Breakout
* Ehlers Instantaneous Trendline
* Funding Rate Strategy
* ML Forecaster Strategy
* Ensemble & Composite Strategies


## Risk Management

I've put a lot of focus on risk management. Here are some of the techniques used:

* **Position Sizing Models:**
    * Fixed Fractional
    * ATR-Based Sizing
    * Optimal F (Kelly Criterion)
    * ML Confidence Sizing
* **Stop-Loss and Take-Profit Models:**
    * ATR Stop-Loss
    * Parabolic SAR Stop
    * Triple-Barrier Method
    * Percentage-Based


## Backtesting

All strategies are thoroughly tested using a custom-built backtesting engine. It supports Walk-Forward Optimization to make sure strategies are robust and calculates a full suite of performance metrics.


## Tech Stack

* **AI & ML**: PyTorch, TensorFlow, LightGBM, LangChain, Ollama
* **Data**: Pandas, NumPy
* **Exchange Connection**: CCXT


## Future Steps

* Set up live trading with exchange APIs (paper and real money).
* Store AI news analysis data in a database for future use in backtests.
* Develop more advanced AI agents.
* Add options strategies for backtesting and trading.