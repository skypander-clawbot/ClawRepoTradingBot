import yfinance as yf
import pandas as pd
import numpy as np

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
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['ATR'] = atr(df, 14)
    df['ADX'] = adx(df, 14)
    return df

sym = 'SPY'
df = yf.download(sym, start='2024-01-01', end='2026-03-27')
df = compute_indicators(df)
print(df[['Close','SMA20','SMA50','ADX']].tail(20))
print('ADX >20?', (df['ADX']>20).any())
print('Golden crosses?')
df['prev_SMA20'] = df['SMA20'].shift()
df['prev_SMA50'] = df['SMA50'].shift()
gc = (df['prev_SMA20'] <= df['prev_SMA50']) & (df['SMA20'] > df['SMA50'])
print(gc.sum())
print(df[gc][['Close','SMA20','SMA50','ADX']].head())