# Stock Price Prediction Using Random Forest

## Overview
This project predicts stock prices using a Random Forest Regressor model trained on historical stock data with technical indicators.

## Features
- Fetches historical stock data using `yfinance`
- Computes key technical indicators (Moving Averages, RSI, Bollinger Bands, MACD, ATR)
- Trains a `RandomForestRegressor` for stock price forecasting
- Evaluates model performance using MAE and RMSE
- Plots actual vs predicted prices
- Saves the trained model and scaler for future use

## Requirements
Install dependencies using:
```bash
pip install numpy pandas yfinance scikit-learn matplotlib joblib
```

## Usage
Run the script:
```bash
python stock_price_prediction.py
```
Enter the stock symbol when prompted (e.g., `AAPL`).

## Output
- Predicted stock price for the next trading day
- Visualization of actual vs predicted prices
- Model saved as `stock_model.pkl`
- Scaler saved as `scaler.pkl`

## License
This project is open-source and free to use.

