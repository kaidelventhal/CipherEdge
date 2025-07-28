## Overview

CipherEdge is an AI quantitative trading program I built for the cryptocurrency markets. It runs on Linux and is designed to automate trading strategies for crypto futures and options. The main goal is to use a multi-strategy and AI-based approach to adapt to changing market conditions.

## Core Components

### Trading Strategies
The program can run and test several different trading strategies:
- **EWMAC Strategy**: A classic trend-following strategy using Exponential Moving Average crossovers.
- **Bollinger Band Mean Reversion**: A strategy that works best in ranging markets, selling at the top band and buying at the lower band.
- **Bollinger Band Breakout**: A trend-following strategy that enters when price breaks out of the bands with high volume.
- **Volatility Squeeze Breakout**: Enters trades when volatility expands after a period of low volatility (a "squeeze").
- **Ehlers Instantaneous Trendline**: A low-lag trend-following strategy.
- **Funding Rate Strategy**: A contrarian strategy that trades based on perpetual futures funding rates.
- **ML Forecaster Strategy**: Uses the output of a machine learning model to decide when to buy or sell.
- **Ensemble & Composite Strategies**: Meta-strategies that combine the signals from multiple other strategies to make a final decision.

### Risk Management
Risk management is a top priority. I've implemented several layers of risk control.

**Position Sizing Models:**
- **Fixed Fractional**: Risks a fixed percentage of the portfolio on each trade.
- **ATR-Based Sizing**: Adjusts trade size based on market volatility (ATR), so each trade has a similar dollar risk.
- **Optimal F (Kelly Criterion)**: Uses a strategy's past win rate and payoff ratio to calculate the optimal fraction of capital to allocate.
- **ML Confidence Sizing**: Increases or decreases trade size based on the confidence score from an AI model's prediction.

**Stop-Loss and Take-Profit Models:**
- **ATR Stop-Loss**: A trailing stop that moves based on a multiple of the Average True Range (ATR).
- **Parabolic SAR Stop**: A trailing stop that follows the Parabolic SAR indicator.
- **Triple-Barrier Method**: Closes a trade based on one of three barriers: a profit target, a stop-loss, or a time limit.
- **Percentage-Based**: A simple stop-loss or take-profit set at a fixed percentage from the entry price.

### Portfolio Construction
The system can build a portfolio from the best-performing strategies found during optimization.
- **Strategy Selection**: It uses metrics like the Deflated Sharpe Ratio to rank strategies and filters them based on their equity curve correlation to ensure diversification.
- **Weighting Methods**: Once strategies are selected, it can assign capital using Equal Weighting or a Risk Parity approach (giving more capital to less volatile strategies).

## AI and Machine Learning
The program uses AI for market analysis and trade signal generation.

- **Price Forecasting**: I have integrated models like LightGBM, XGBoost, and LSTMs (using PyTorch/TensorFlow) to predict future price movements.
- **Market Regime Detection**: A K-Means clustering model is used to identify the current market environment (e.g., trending, ranging, or high volatility).
- **AI News Analysis**: An asynchronous system (`NewsListener` and `NewsProcessor`) monitors RSS feeds, scrapes the full article text, and uses a local LLM (Ollama) with structured output to analyze the news for sentiment, key themes, and related crypto tickers.
- **AI Research Agent**: A `BrowserAgent` uses LangChain tools to perform active research on a given crypto ticker, Browse websites to find the latest news that could impact its price.

## Backtesting and Optimization
All strategies are tested before use. The backtesting engine is custom-built and supports advanced features like Walk-Forward Optimization to avoid overfitting and ensure strategies are robust. It calculates a full suite of performance metrics, including the Sharpe Ratio and results from Monte Carlo simulations.

## Technology Stack
- **AI & ML**: PyTorch, TensorFlow, LightGBM, LangChain, Ollama
- **Data**: Pandas, NumPy
- **Exchange Connection**: CCXT

## Future Steps
- **Live Trading Deployment**: Connect the engine to an exchange API for live paper and real-money trading.
- **Data Persistence**: Store the structured output from the AI news analysis in a database. This will allow new sentiment and theme data to be used as features in future backtests.
- **Expand AI Agents**: Build more advanced AI agents that can perform more complex research tasks or even interact with smart contracts.
- **Add Option Strategies**: Implement functionality for backtesting and trading options strategies.