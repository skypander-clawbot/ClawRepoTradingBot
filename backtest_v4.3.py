#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Config from v4.3
CONFIG = {
    "symbols": ["SPY", "QQQ", "GLD", "XLE", "AAPL", "TSLA", "NVDA", "JNJ"],
    "pair": ("SPY", "QQQ"),
    "sma_short": 20,
    "sma_long": 50,
    "stop_loss_pct": 0.05,
    "trailing_stop_pct": 0.10,
    "partial_profit_pct": 0.20,
    "trend_filter": True,
    "max_position_pct": 0.40,
    "pair_z_threshold": 2.0,
    "pair_lookback": 20,
    "initial_capital": 8000.0,
}

def compute_indicators(series):
    df = pd.DataFrame(index=series.index)
    df["Close"] = series
    df["SMA20"] = df["Close"].rolling(window=CONFIG["sma_short"]).mean()
    df["SMA50"] = df["Close"].rolling(window=CONFIG["sma_long"]).mean()
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def generate_signal(row, prev):
    signal = "HOLD"
    reason = ""
    if prev is None:
        return signal, reason
    # Golden Cross
    if prev["SMA20"] <= prev["SMA50"] and row["SMA20"] > row["SMA50"]:
        if not CONFIG["trend_filter"] or row["SMA20"] > row["SMA50"]:
            signal = "BUY"
            reason = "Golden Cross"
    # Death Cross
    elif prev["SMA20"] >= prev["SMA50"] and row["SMA20"] < row["SMA50"]:
        signal = "SELL"
        reason = "Death Cross"
    return signal, reason

def backtest(symbols, start, end):
    # Download data as DataFrame with columns symbols
    data = yf.download(symbols, start=start, end=end)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    data = data.dropna()
    cash = CONFIG["initial_capital"]
    positions = {s: {"shares":0, "entry":0.0, "highest":0.0, "stop":0.0, "trailing":0.0, "partial_sold":False} for s in symbols}
    trades = []
    portfolio_values = []
    for date in data.index:
        # Update prices
        prices = data.loc[date].to_dict()
        # Update trailing stops etc
        for sym in symbols:
            pos = positions[sym]
            price = prices.get(sym)
            if price is None or np.isnan(price):
                continue
            if pos["shares"] > 0:
                # update highest
                if price > pos["highest"]:
                    pos["highest"] = price
                    new_trail = price * (1 - CONFIG["trailing_stop_pct"])
                    if new_trail > pos["trailing"]:
                        pos["trailing"] = new_trail
                # check exits
                exit_flag = False
                exit_reason = ""
                # partial profit
                if not pos["partial_sold"]:
                    target = pos["entry"] * (1 + CONFIG["partial_profit_pct"])
                    if price >= target:
                        sell_shares = pos["shares"] // 2
                        if sell_shares > 0:
                            proceeds = sell_shares * price
                            cash += proceeds
                            pnl = (price - pos["entry"]) * sell_shares
                            trades.append({"date":date, "symbol":sym, "action":"SELL_PARTIAL", "shares":sell_shares, "price":price, "pnl":pnl, "reason":"Partial Profit"})
                            pos["shares"] -= sell_shares
                            pos["partial_sold"] = True
                            exit_flag = True
                # trailing stop
                if not exit_flag and pos["trailing"] > 0 and price <= pos["trailing"]:
                    sell_shares = pos["shares"]
                    proceeds = sell_shares * price
                    cash += proceeds
                    pnl = (price - pos["entry"]) * sell_shares
                    trades.append({"date":date, "symbol":sym, "action":"SELL", "shares":sell_shares, "price":price, "pnl":pnl, "reason":"Trailing Stop"})
                    pos["shares"] = 0
                    exit_flag = True
                # stop loss
                if not exit_flag and pos["stop"] > 0 and price <= pos["stop"]:
                    sell_shares = pos["shares"]
                    proceeds = sell_shares * price
                    cash += proceeds
                    pnl = (price - pos["entry"]) * sell_shares
                    trades.append({"date":date, "symbol":sym, "action":"SELL", "shares":sell_shares, "price":price, "pnl":pnl, "reason":"Stop-Loss"})
                    pos["shares"] = 0
                    exit_flag = True
        # generate signals for buying
        for sym in symbols:
            # need historical data up to date
            hist = data[sym].loc[:date]
            if len(hist) < CONFIG["sma_long"]+1:
                continue
            df = compute_indicators(hist)
            row = df.iloc[-1]
            prev = df.iloc[-2] if len(df)>=2 else None
            signal, reason = generate_signal(row, prev)
            if signal == "BUY" and positions[sym]["shares"] == 0:
                price = prices.get(sym)
                if price is None or np.isnan(price):
                    continue
                max_invest = cash * CONFIG["max_position_pct"]
                if max_invest < price:
                    continue
                shares = int(max_invest // price)
                if shares == 0:
                    continue
                cost = shares * price
                cash -= cost
                positions[sym] = {"shares":shares, "entry":price, "highest":price, "stop":price*(1-CONFIG["stop_loss_pct"]), "trailing":price*(1-CONFIG["trailing_stop_pct"]), "partial_sold":False}
                trades.append({"date":date, "symbol":sym, "action":"BUY", "shares":shares, "price":price, "pnl":0, "reason":reason})
        # compute portfolio value
        positions_value = sum(pos["shares"]*prices.get(sym,0) for sym,pos in positions.items() if not np.isnan(prices.get(sym,np.nan)))
        portfolio_values.append({"date":date, "cash":cash, "positions_value":positions_value, "total":cash+positions_value})
    # finalize any open positions at last price
    last_prices = data.iloc[-1].to_dict()
    for sym in symbols:
        pos = positions[sym]
        if pos["shares"]>0:
            price = last_prices.get(sym,0)
            proceeds = pos["shares"]*price
            cash += proceeds
            pnl = (price - pos["entry"])*pos["shares"]
            trades.append({"date":data.index[-1], "symbol":sym, "action":"SELL_END", "shares":pos["shares"], "price":price, "pnl":pnl, "reason":"End of backtest"})
            pos["shares"]=0
    # compute final value
    final_value = cash
    # buy&hold
    bh_cash = CONFIG["initial_capital"]
    weight = 1/len(symbols)
    alloc = bh_cash * weight
    bh_shares = {}
    bh_end = {}
    for sym in symbols:
        start_price = data[sym].iloc[0]
        end_price = data[sym].iloc[-1]
        shares = alloc / start_price
        bh_shares[sym]=shares
        bh_end[sym]=shares*end_price
    bh_total = sum(bh_end.values())
    return {
        "trades":trades,
        "portfolio_values":portfolio_values,
        "final_value":final_value,
        "buy_hold":bh_total,
        "bh_shares":bh_shares,
        "bh_end":bh_end,
        "data":data
    }

if __name__=="__main__":
    start="2024-01-01"
    end="2026-03-27"
    res = backtest(CONFIG["symbols"], start, end)
    print(f"Backtest {start} to {end}")
    print(f"Final strategy value: {res['final_value']:.2f}")
    print(f"Buy&Hold value: {res['buy_hold']:.2f}")
    print(f"Return strategy: {(res['final_value']/CONFIG['initial_capital']-1)*100:.2f}%")
    print(f"Return B&H: {(res['buy_hold']/CONFIG['initial_capital']-1)*100:.2f}%")
    # print trades
    print(f"\nNumber of trades: {len(res['trades'])}")
    for t in res['trades'][-30:]:
        print(f"{t['date'].date()} {t['symbol']} {t['action']} {t['shares']} @ {t['price']:.2f} PnL {t['pnl']:.2f} ({t['reason']})")