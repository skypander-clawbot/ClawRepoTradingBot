#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np

CONFIG = {
    "symbols": ["SPY", "QQQ", "GLD", "XLE", "AAPL", "TSLA", "NVDA", "JNJ"],
    "sma_short": 20,
    "sma_long": 50,
    "atr_period": 14,
    "risk_per_trade": 0.01,
    "max_equity_at_risk": 0.80,
    "partial_profit_pct": 0.20,
    "initial_sl_atr_mult": 2.0,
    "trailing_atr_mult": 2.5,
    "adx_period": 14,
    "adx_threshold": 20.0,  # changed to 20
    "initial_capital": 8000.0,
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
    return df

def backtest(symbols, start, end):
    data = yf.download(symbols, start=start, end=end)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    data = data.dropna()
    close = data['Close']
    high = data['High']
    low = data['Low']
    price_df = pd.DataFrame({s: close[s] for s in symbols if s in close.columns})
    high_df = pd.DataFrame({s: high[s] for s in symbols if s in high.columns})
    low_df = pd.DataFrame({s: low[s] for s in symbols if s in low.columns})
    price_df = price_df.dropna()
    high_df = high_df.loc[price_df.index]
    low_df = low_df.loc[price_df.index]
    indicators = {}
    for s in symbols:
        df_sym = pd.DataFrame({'Open': price_df[s], 'High': high_df[s], 'Low': low_df[s], 'Close': price_df[s]})
        df_sym = compute_indicators(df_sym)
        indicators[s] = df_sym
    cash = CONFIG['initial_capital']
    positions = {s:{'shares':0,'entry':0.0,'highest':0.0,'stop':0.0,'trail':0.0,'atr_entry':0.0,'partial':False} for s in symbols}
    trades = []
    equity_curve = []
    for date in price_df.index:
        prices = price_df.loc[date].to_dict()
        # update positions
        for s in symbols:
            pos = positions[s]
            if pos['shares']>0:
                price = prices[s]
                if price > pos['highest']:
                    pos['highest'] = price
                    new_trail = price - CONFIG['trailing_atr_mult']*pos['atr_entry']
                    if new_trail > pos['trail']:
                        pos['trail'] = new_trail
                exited=False
                if not pos['partial'] and price >= pos['entry']*(1+CONFIG['partial_profit_pct']):
                    sell = pos['shares']//2
                    if sell>0:
                        proceeds=sell*price
                        cash+=proceeds
                        pnl=(price-pos['entry'])*sell
                        trades.append({'date':date,'symbol':s,'action':'SELL_PARTIAL','shares':sell,'price':price,'pnl':pnl,'reason':'Partial Profit'})
                        pos['shares']-=sell
                        pos['partial']=True
                        exited=True
                if not exited and price <= pos['trail']:
                    proceeds=pos['shares']*price
                    cash+=proceeds
                    pnl=(price-pos['entry'])*pos['shares']
                    trades.append({'date':date,'symbol':s,'action':'SELL','shares':pos['shares'],'price':price,'pnl':pnl,'reason':'Trailing Stop'})
                    pos['shares']=0
                    exited=True
                if not exited and price <= pos['stop']:
                    proceeds=pos['shares']*price
                    cash+=proceeds
                    pnl=(price-pos['entry'])*pos['shares']
                    trades.append({'date':date,'symbol':s,'action':'SELL','shares':pos['shares'],'price':price,'pnl':pnl,'reason':'Stop-Loss'})
                    pos['shares']=0
                    exited=True
        # generate signals
        for s in symbols:
            if positions[s]['shares']>0:
                continue
            df = indicators[s].loc[:date]
            if len(df) < CONFIG['sma_long']+5:
                continue
            row = df.iloc[-1]
            prev = df.iloc[-2]
            signal='HOLD'; reason=''
            if prev is not None:
                if row['SMA20']>row['SMA50'] and row['ADX']>CONFIG['adx_threshold']:
                    if prev['SMA20']<=prev['SMA50'] and row['SMA20']>row['SMA50']:
                        signal='BUY'; reason='Golden Cross + ADX>'
                    elif prev['SMA20']>=prev['SMA50'] and row['SMA20']<row['SMA50']:
                        signal='SELL'; reason='Death Cross + ADX>'
            if signal=='BUY':
                atr_val = row['ATR']
                risk_per_share = CONFIG['initial_sl_atr_mult']*atr_val
                equity = cash + sum(positions[ss]['shares']*prices[ss] for ss in symbols)
                max_risk = equity*CONFIG['risk_per_trade']
                shares_by_risk = int(max_risk/risk_per_share) if risk_per_share>0 else 0
                shares_by_cash = int((cash*0.4)/row['Close'])
                shares = min(shares_by_risk, shares_by_cash)
                if shares<=0: continue
                cost=shares*row['Close']
                if cost>cash: continue
                cash-=cost
                positions[s]={'shares':shares,'entry':row['Close'],'highest':row['Close'],'stop':row['Close']-CONFIG['initial_sl_atr_mult']*atr_val,
                              'trail':row['Close']-CONFIG['trailing_atr_mult']*atr_val,'atr_entry':atr_val,'partial':False}
                trades.append({'date':date,'symbol':s,'action':'BUY','shares':shares,'price':row['Close'],'pnl':0,'reason':reason})
        # equity
        positions_val = sum(positions[s]['shares']*prices[s] for s in symbols)
        equity_curve.append({'date':date,'equity':cash+positions_val})
    last = price_df.iloc[-1]
    for s in symbols:
        pos=positions[s]
        if pos['shares']>0:
            price=last[s]
            proceeds=pos['shares']*price
            cash+=proceeds
            pnl=(price-pos['entry'])*pos['shares']
            trades.append({'date':price_df.index[-1],'symbol':s,'action':'SELL_END','shares':pos['shares'],'price':price,'pnl':pnl,'reason':'End of backtest'})
            pos['shares']=0
    final_eq = cash
    init=CONFIG['initial_capital']
    weight=1/len(symbols)
    alloc=init*weight
    bh_end=0
    for s in symbols:
        start_p = price_df[s].iloc[0]
        end_p = price_df[s].iloc[-1]
        shares = alloc/start_p
        bh_end+=shares*end_p
    return {
        'trades':trades,
        'equity_curve':equity_curve,
        'final_eq':final_eq,
        'bh_eq':bh_end
    }

if __name__=='__main__':
    start='2024-01-01'
    end='2026-03-27'
    res=backtest(CONFIG['symbols'],start,end)
    print(f"Backtest {start} to {end}")
    print(f"Final strategy equity: {res['final_eq']:.2f}")
    print(f"Buy&Hold equity: {res['bh_eq']:.2f}")
    print(f"Return strategy: {(res['final_eq']/CONFIG['initial_capital']-1)*100:.2f}%")
    print(f"Return B&H: {(res['bh_eq']/CONFIG['initial_capital']-1)*100:.2f}%")
    print(f"Number of trades: {len(res['trades'])}")
    for t in res['trades'][-20:]:
        print(f"{t['date'].date()} {t['symbol']} {t['action']} {t['shares']} @ {t['price']:.2f} PnL {t['pnl']:.2f} ({t['reason']})")