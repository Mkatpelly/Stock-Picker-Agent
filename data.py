# data.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(ticker: str, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Close']].dropna()
    return df  # index is datetime, column 'Close'

def make_sequences(series: np.ndarray, seq_len: int = 60):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X)      # (N, seq_len, 1)
    y = np.array(y)      # (N, 1)
    return X, y

def prepare_data(df, seq_len=60, train_ratio=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['Close']].values)

    X, y = make_sequences(scaled, seq_len)
    split = int(len(X) * train_ratio)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    return X_train, y_train, X_test, y_test, scaler
