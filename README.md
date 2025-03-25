# Forecasting Bitcoin Price Movements Using a CNN-LSTM Architecture with Reinforcement Learning Extensions

## Project Overview

Day trading and swing trading have traditionally been dominated by traders relying on technical analysis to identify patterns and trends in financial charts. While the efficient market hypothesis suggests many observed patterns might be noise, the day trading economy indicates that human trading behavior introduces inefficiencies that can be systematically exploited.

This project implements a hybrid CNN-LSTM model for Bitcoin price forecasting, motivated by the hypothesis that collective trading strategies embedded in chart patterns and technical signals can be more effectively captured by machine learning than human intuition alone. The model combines:
- Convolutional Neural Networks (CNN) for spatial feature extraction
- Long Short-Term Memory (LSTM) networks for temporal pattern learning
- Potential reinforcement learning extensions for automated trading

### Input Features
- Initial input: 5×1 vector (OHLCV - Open, High, Low, Volume, Close)
- Potential extension: 10×1 vector incorporating risk arbitrage signals from multiple data sources
- Design mirrors decision-making cues used by human traders

## Project Structure

### 1. Data Sourcing and Preprocessing
- **Historical Data Acquisition**
  - Collection of Bitcoin OHLCV data
  - 5×1 input vector encoding for each time interval
- **Risk Arbitrage Extension**
  - Data collection from multiple exchanges (e.g., Coinbase vs Kraken)
  - Integration of cross-exchange pricing signals
- **Data Cleaning**
  - Structured dataframe creation
  - Consistency and reliability verification

### 2. Model Architecture and Methodology
- **CNN-LSTM Framework**
  - CNN first layer for local spatial feature extraction
  - LSTM layer for temporal pattern learning
  - Specialized approach for infinitely divisible time series
- **Forecasting Objective**
  - Binary prediction of up/down price movements
  - Foundation for reinforcement learning extensions
- **Dimensionality Reduction**
  - PCA implementation for feature optimization
  - Variance contribution analysis

### 3. Hyperparameter Tuning and Regularization
- **Bayesian Optimization** for:
  - Batch size
  - Number of epochs
  - Learning rates
- **Regularization Techniques**
  - L2 regularization
  - Dropout implementation
- **Validation Strategy**
  - 80-20 training-validation split
  - Optional k-fold cross-validation

## Project Scope

### Primary Objectives
1. Analysis of Bitcoin's directional price movements using OHLCV data
2. Achievement of >50% prediction accuracy (edge over random walk)
3. Development of profitable trading strategies

### Future Extensions
1. **Reinforcement Learning Integration**
   - Dynamic trading decisions (buy/sell/hold)
   - Reward function optimization
   - Account balance management
   - Transaction cost consideration

2. **Advanced Trading Features**
   - Options trading
   - Short selling
   - Leveraged positions
   - Mean-variance optimization

## Requirements

Dependencies and setup instructions will be specified in `requirements.txt`

## Getting Started

Detailed setup and usage instructions will be added as implementation progresses.

## License

TBD

## Author

Jeddy Xie 