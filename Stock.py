import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_stock_data(symbol, start_date, end_date, max_retries=3):
    """Fetch historical stock data using yfinance with retries"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(symbol).history(period="2y")
            if stock.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            return stock
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    logging.error("Error fetching stock data after multiple attempts.")
    return None

def compute_technical_indicators(df):
    """Compute technical indicators"""
    df['Returns'] = df['Close'].pct_change()
    df['Lagged_Returns'] = df['Returns'].shift(1)
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Upper'] = df['MA21'] + (2 * df['Close'].rolling(window=21).std())
    df['BB_Lower'] = df['MA21'] - (2 * df['Close'].rolling(window=21).std())
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    
    # ATR
    df['ATR'] = df['High'].subtract(df['Low']).rolling(window=14).mean()
    
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    """Prepare data for model training"""
    if df is None or df.empty:
        raise ValueError("No stock data available")
    
    df = compute_technical_indicators(df)
    
    features = ['Close', 'Lagged_Returns', 'MA7', 'MA21', 'RSI', 'BB_Upper', 'BB_Lower', 'MACD', 'ATR', 'Volume']
    target = 'Close'
    
    X = df[features].values
    y = df[target].shift(-1).dropna().values
    X = X[:-1]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y):
    """Train a Random Forest model"""
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return model, X_test, y_test, y_pred

def plot_predictions(actual, predicted, title):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip()
    if not symbol:
        logging.error("Invalid stock symbol!")
        return
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    df = fetch_stock_data(symbol, start_date, end_date)
    if df is None:
        return
    
    X, y, scaler = prepare_data(df)
    model, X_test, y_test, y_pred = train_model(X, y)
    plot_predictions(y_test, y_pred, f'{symbol} Stock Price Prediction')
    
    joblib.dump(model, 'stock_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    logging.info("Model and scaler saved successfully.")
    
    last_data = X[-1:].reshape(1, -1)
    next_day_pred = model.predict(last_data)[0]
    next_trading_date = df.index[-1] + timedelta(days=1)
    print(f"\nPredicted price for {next_trading_date.date()}: ${next_day_pred:.2f}")
    
if __name__ == "__main__":
    main()