#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

CONFIG = {
    "symbols": [
        "SPY", "QQQ", "GLD", "XLE", "AAPL", "TSLA", "NVDA", "JNJ",
        "META", "MU", "CRWD", "GOOGL", "HOOD", "PLTR", "AMD", "RKLB"
    ],
    "sma_short": 20,
    "sma_long": 50,
    "atr_period": 14,
    "adx_period": 14,
    "lookforward": 5,  # days ahead for label
    "threshold": 0.01, # label up if future return >1%
}

def atr(df, period):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period):
    up = df['High'].diff()
    down = df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = atr(df, period)
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / tr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

def compute_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(CONFIG['sma_short']).mean()
    df['SMA50'] = df['Close'].rolling(CONFIG['sma_long']).mean()
    df['ATR'] = atr(df, CONFIG['atr_period'])
    df['ADX'] = adx(df, CONFIG['adx_period'])
    delta = df['Close'].diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100/(1+rs))
    df['Return_5d'] = df['Close'].pct_change(CONFIG['lookforward']).shift(-CONFIG['lookforward'])
    return df

def build_dataset():
    data = yf.download(CONFIG['symbols'], start='2020-01-01', end='2026-03-27')
    if isinstance(data, pd.Series):
        data = data.to_frame(name=CONFIG['symbols'][0])
    data = data.dropna()
    # separate
    close = data['Close']
    high = data['High']
    low = data['Low']
    feat_list = []
    label_list = []
    for sym in CONFIG['symbols']:
        df_sym = pd.DataFrame({'Open': close[sym], 'High': high[sym], 'Low': low[sym], 'Close': close[sym]})
        df_sym = compute_indicators(df_sym)
        df_sym = df_sym.dropna()
        # features
        df_sym['SMA_diff'] = (df_sym['SMA20'] - df_sym['SMA50']) / df_sym['Close']
        df_sym['ATR_pct'] = df_sym['ATR'] / df_sym['Close']
        df_sym['RSI_norm'] = (df_sym['RSI'] - 50) / 50
        features = df_sym[['SMA_diff', 'ADX', 'ATR_pct', 'RSI_norm']]
        # label: future return > threshold
        label = (df_sym['Return_5d'] > CONFIG['threshold']).astype(int)
        feat_list.append(features)
        label_list.append(label)
    X = pd.concat(feat_list, axis=0)
    y = pd.concat(label_list, axis=0)
    return X, y

def main():
    print("Building dataset...")
    X, y = build_dataset()
    print(f"Samples: {len(X)}")
    print(f"Positive label ratio: {y.mean():.3f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    print(classification_report(y_test, preds))
    # save model and scaler
    model_dir = '/data/.openclaw/workspace/ml_model'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'logistic_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print("Model saved to", model_dir)

if __name__ == '__main__':
    main()