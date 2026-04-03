#!/usr/bin/env python3
"""
Botti Trader – v6.0
Vollständig überarbeiteter Trend-Following + Mean-Reversion Bot.

Neue Features gegenüber v5:
  - MACD & Volume-Bestätigung für Golden-Cross-Signale
  - Bollinger-Band Mean-Reversion als zweite Strategie
  - RSI-Filter für Entry & MR
  - Sektorbasierter Klumpenrisiko-Guard (max N Positionen pro Sektor)
  - VIX-Volatilitätsregime-Filter (Positionsgröße halbieren wenn VIX > 30)
  - Transaktionskosten & Slippage-Simulation
  - Max-Volumen-Guard (max 1% des Tagesvolumens)
  - Drawdown-Circuit-Breaker (keine neuen Trades bei >15% Drawdown)
  - Re-Entry nach Partial-Profit-Verkauf
  - Pair-Trading Execution (Long + Short Positionen)
  - Telegram-Trade-Alerts (optional)
  - Benchmark (SPY Buy-and-Hold) Tracking im Report
  - Alpha-Berechnung vs. Benchmark
  - Backtest-Modul (CAGR, Sharpe, Max-Drawdown, Win-Rate)
  - ML-Training mit LightGBM (Fallback: LogisticRegression)
    erweiterte Features: MACD_hist zusätzlich zu sma_diff/adx/atr/rsi
  - Alpaca Markets Edition: Daten + Orderausführung via alpaca-py

Universe (16 Assets):
    SPY, QQQ, GLD, XLE, AAPL, TSLA, NVDA, JNJ,
    META, MU, CRWD, GOOGL, HOOD, PLTR, AMD, RKLB

Installation:
    pip install alpaca-py pytz pandas numpy

Umgebungsvariablen (OpenClaw setzt diese automatisch):
    APCA_API_KEY_ID      – Alpaca API Key
    APCA_API_SECRET_KEY  – Alpaca Secret Key
    APCA_PAPER           – "true" für Paper Trading (Standard), "false" für Live
    APCA_DATA_FEED       – "iex" (kostenlos, Standard) oder "sip" (Echtzeit, kostenpflichtig)

Aufruf:
    python trader_v6.py --mode scan       → täglichen Run starten (Standard)
    python trader_v6.py --mode status     → Portfolio-Status (JSON-Ausgabe)
    python trader_v6.py --mode eod        → Alle Positionen schließen
    python trader_v6.py --mode backtest   → Historischen Backtest laufen lassen
    python trader_v6.py --mode train      → ML-Modell trainieren

    # Ältere Aufrufformen bleiben kompatibel:
    python trader_v6.py backtest [400d]
    python trader_v6.py train
"""

import json
import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# Optional heavy dependencies
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings("ignore")

# ── Alpaca-py (pip install alpaca-py) ──────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        GetOrdersRequest,
        StopLossRequest,
        TakeProfitRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARN] alpaca-py fehlt → pip install alpaca-py", file=sys.stderr)


# ============================= Configuration =============================
CONFIG = {
    # --- Universe -------------------------------------------------------
    "symbols": [
        "SPY", "QQQ", "GLD", "XLE", "AAPL", "TSLA", "NVDA", "JNJ",
        "META", "MU", "CRWD", "GOOGL", "HOOD", "PLTR", "AMD", "RKLB",
    ],
    # --- Sector groups (Klumpenrisiko-Guard) ----------------------------
    "sector_groups": {
        "tech_semi":   ["NVDA", "AMD", "MU"],
        "tech_large":  ["AAPL", "META", "GOOGL"],
        "tech_growth": ["CRWD", "PLTR", "HOOD", "RKLB"],
        "etf_broad":   ["SPY", "QQQ"],
        "defensive":   ["JNJ", "GLD"],
        "energy":      ["XLE"],
        "ev_space":    ["TSLA"],
    },
    "max_per_sector": 2,
    # --- Core trend (SMA-Crossover) -------------------------------------
    "sma_short": 20,
    "sma_long":  50,
    # --- Bestätigungsfilter --------------------------------------------
    "use_rsi_filter":    True,
    "rsi_buy_min":       40,   # Mindestmomentum vorhanden
    "rsi_buy_max":       70,   # Nicht überkauft
    "use_volume_filter": True,
    "volume_sma_period": 20,
    "use_macd_filter":   True,
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal_period": 9,
    # --- Mean Reversion (Bollinger) ------------------------------------
    "use_mean_reversion": True,
    "bb_period":          20,
    "bb_std":             2.0,
    "mr_rsi_max":         35,   # MR nur wenn RSI überverkauft
    "mr_profit_target_pct": 0.05,  # MR-Ziel: +5% vom Einstieg
    # --- Risk / Sizing --------------------------------------------------
    "atr_period":        14,
    "risk_per_trade":    0.02,
    "max_equity_at_risk": 0.80,
    # --- Trade Management -----------------------------------------------
    "partial_profit_pct":   0.25,
    "initial_sl_atr_mult":  2.5,
    "trailing_atr_mult":    3.0,
    # --- Re-Entry nach Partial-Sell ------------------------------------
    "allow_reentry":     True,
    "reentry_atr_mult":  1.5,  # Re-Entry wenn Preis < highest - 1.5*ATR
    # --- Trend Filter ---------------------------------------------------
    "adx_period":    14,
    "adx_threshold": 20,
    # --- VIX Regime Filter ---------------------------------------------
    "use_vix_filter":     True,
    "vix_high_threshold": 30,
    "vix_size_reduction": 0.5,
    # --- Drawdown Circuit Breaker --------------------------------------
    "max_drawdown_pct": 0.15,
    # --- Pair Trading ---------------------------------------------------
    "pair":              ("SPY", "QQQ"),
    "pair_lookback_base": 20,
    "pair_z_entry":      2.0,
    "pair_z_exit":       0.5,
    "pair_position_pct": 0.05,  # 5% of equity pro Seite
    # --- Transaktionskosten & Slippage ---------------------------------
    "commission_pct": 0.001,   # 0,1% pro Trade
    "slippage_pct":   0.001,   # 0,1% Slippage
    "max_volume_pct": 0.01,    # max 1% des Tagesvolumens
    # --- ML Overlay ----------------------------------------------------
    "use_ml":             False,
    "ml_prob_threshold":  0.6,
    # --- Telegram Alerts -----------------------------------------------
    "telegram_token":   "",    # Bot-Token setzen um Alerts zu aktivieren
    "telegram_chat_id": "",
    # --- Benchmark -----------------------------------------------------
    "benchmark": "SPY",
    # --- Alpaca-Verbindung ─────────────────────────────────────────────
    # Werte aus Umgebungsvariablen; hier als Fallback-Defaults
    "alpaca_paper":     True,    # überschrieben durch APCA_PAPER env-var
    "alpaca_data_feed": "iex",   # "iex" = kostenlos | "sip" = Echtzeit (kostenpflichtig)
    # --- Misc -----------------------------------------------------------
    "currency":        "EUR",
    "initial_capital": 10_000.0,
    "data_dir":        Path(__file__).parent / "trading_data",
    "portfolio_file":  Path(__file__).parent / "trading_data" / "portfolio.json",
}


# ============================= AlpacaClient =================================

class AlpacaClient:
    """
    Zentrale Klasse für alle Alpaca-Interaktionen.
    Trennt sauber zwischen Datenabruf (StockHistoricalDataClient)
    und Orderausführung (TradingClient).
    Nutzt Daily-Bars für den Botti Trader (statt 5-Min-Bars im ORB-Bot).
    """

    def __init__(self, api_key: str, secret_key: str,
                 paper: bool = True, data_feed: str = "iex"):
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py fehlt – pip install alpaca-py")
        self.paper     = paper
        self.data_feed = data_feed
        self.trading   = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.data      = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        mode = "PAPER" if paper else "LIVE ⚠️"
        print(f"[Alpaca] Verbunden  Modus={mode}  Feed={data_feed}")

    # ── Marktdaten ──────────────────────────────────────────────────────────

    def fetch_bars(self, symbol: str, days: int = 150) -> pd.DataFrame:
        """Daily-Bars für ein Symbol – ersetzt yfinance._fetch()."""
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now(pytz.UTC) - timedelta(days=days),
                end=datetime.now(pytz.UTC),
                adjustment="raw",
                feed=self.data_feed,
            )
            bars = self.data.get_stock_bars(req)
            if bars.df.empty:
                return pd.DataFrame()
            df = bars.df.loc[symbol].copy() if isinstance(bars.df.index, pd.MultiIndex) \
                 else bars.df.copy()
            return self._rename(df)
        except Exception as e:
            print(f"[Alpaca] Datenfehler {symbol}: {e}")
            return pd.DataFrame()

    def fetch_bars_bulk(self, symbols: List[str],
                        start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        Historische Daily-Bars für mehrere Symbole auf einmal.
        Ersetzt yfinance-Downloads im Backtester.
        Alpaca hat kein 60-Tage-Limit wie yfinance → kein Chunking nötig.
        """
        result: Dict[str, pd.DataFrame] = {}
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC),
                end=datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=pytz.UTC),
                adjustment="raw",
                feed=self.data_feed,
            )
            bars = self.data.get_stock_bars(req)
            if bars.df.empty:
                return result
            for sym in symbols:
                try:
                    result[sym] = self._rename(bars.df.loc[sym].copy())
                    print(f"  ✓ {sym}: {len(result[sym])} Bars")
                except KeyError:
                    print(f"  ✗ {sym}: keine Daten")
        except Exception as e:
            print(f"[Alpaca] Bulk-Fehler: {e}")
        return result

    @staticmethod
    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        """Alpaca-Spaltennamen → OHLCV-Standard."""
        return df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]

    # ── Account & Positionen ────────────────────────────────────────────────

    def get_equity(self) -> float:
        try:
            return float(self.trading.get_account().equity)
        except Exception as e:
            print(f"[Alpaca] Equity-Fehler: {e}")
            return 0.0

    def get_cash(self) -> float:
        try:
            return float(self.trading.get_account().cash)
        except Exception:
            return 0.0

    def get_buying_power(self) -> float:
        try:
            return float(self.trading.get_account().buying_power)
        except Exception:
            return 0.0

    def sync_positions(self) -> Dict[str, dict]:
        """
        Aktuelle Positionen direkt von Alpaca holen.
        Gibt Wahrheit über offene Positionen – verhindert Doppel-Entries.
        """
        try:
            positions = self.trading.get_all_positions()
            return {
                p.symbol: {
                    "symbol":          p.symbol,
                    "qty":             float(p.qty),
                    "side":            p.side.value,        # "long" | "short"
                    "entry":           float(p.avg_entry_price),
                    "current_price":   float(p.current_price),
                    "unrealized_pnl":  float(p.unrealized_pl),
                    "market_value":    float(p.market_value),
                }
                for p in positions
            }
        except Exception as e:
            print(f"[Alpaca] Positions-Fehler: {e}")
            return {}

    def is_shortable(self, symbol: str) -> bool:
        """Prüft ob Alpaca das Symbol für Shorts freigibt."""
        try:
            asset = self.trading.get_asset(symbol)
            return bool(asset.shortable) and bool(asset.easy_to_borrow)
        except Exception:
            return False

    def get_open_orders(self) -> List[dict]:
        try:
            req    = GetOrdersRequest(status="open", limit=50)
            orders = self.trading.get_orders(req)
            return [{"id": str(o.id), "symbol": o.symbol,
                     "side": o.side.value, "qty": float(o.qty),
                     "status": o.status.value} for o in orders]
        except Exception as e:
            print(f"[Alpaca] Orders-Fehler: {e}")
            return []

    # ── Orderausführung ─────────────────────────────────────────────────────

    def place_long_bracket(self, symbol: str, qty: int,
                            stop_loss: float, take_profit: float) -> Optional[dict]:
        """
        Long-Entry als Bracket-Order.
        Alpaca verwaltet Stop-Loss und Take-Profit serverseitig.
        """
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_loss,    2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            )
            r = self.trading.submit_order(order)
            print(f"[Alpaca] LONG  {symbol} {qty} Aktien | SL {stop_loss:.2f} | TP {take_profit:.2f} → {r.status.value}")
            return {"id": str(r.id), "symbol": symbol, "qty": qty,
                    "side": "long", "stop_loss": stop_loss,
                    "take_profit": take_profit, "status": r.status.value}
        except Exception as e:
            print(f"[Alpaca] Long-Order {symbol} fehlgeschlagen: {e}")
            return None

    def place_short_bracket(self, symbol: str, qty: int,
                             stop_loss: float, take_profit: float) -> Optional[dict]:
        """
        Short-Entry als Bracket-Order.
        stop_loss liegt ÜBER dem Entry, take_profit DARUNTER.
        Erfordert Margin-Konto + Shortability-Check.
        """
        if not self.is_shortable(symbol):
            print(f"[Alpaca] {symbol} nicht shortbar – Order abgebrochen")
            return None
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,          # Sell-to-Open = Short
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_loss,    2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            )
            r = self.trading.submit_order(order)
            print(f"[Alpaca] SHORT {symbol} {qty} Aktien | SL {stop_loss:.2f} | TP {take_profit:.2f} → {r.status.value}")
            return {"id": str(r.id), "symbol": symbol, "qty": qty,
                    "side": "short", "stop_loss": stop_loss,
                    "take_profit": take_profit, "status": r.status.value}
        except Exception as e:
            print(f"[Alpaca] Short-Order {symbol} fehlgeschlagen: {e}")
            return None

    def close_position(self, symbol: str) -> None:
        """Schließt eine offene Position vollständig (alle Legs + Orders storniert)."""
        try:
            self.trading.close_position(symbol)
            print(f"[Alpaca] Position {symbol} geschlossen")
        except Exception as e:
            print(f"[Alpaca] Close {symbol} Fehler: {e}")

    def reduce_position(self, symbol: str, qty: int) -> None:
        """Verkauft `qty` Anteile einer bestehenden Long-Position (Partial-Close)."""
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            self.trading.submit_order(order)
            print(f"[Alpaca] SELL {qty} {symbol} (Partial-Close)")
        except Exception as e:
            print(f"[Alpaca] Reduce {symbol} Fehler: {e}")

    def cancel_all_orders(self) -> None:
        try:
            self.trading.cancel_orders()
            print("[Alpaca] Alle offenen Orders storniert")
        except Exception as e:
            print(f"[Alpaca] Stornierungsfehler: {e}")

    def close_all_positions(self) -> None:
        """EOD: alle Positionen schließen + offene Orders stornieren."""
        try:
            self.trading.close_all_positions(cancel_orders=True)
            print("[Alpaca] Alle Positionen geschlossen (EOD)")
        except Exception as e:
            print(f"[Alpaca] EOD-Close Fehler: {e}")


# ============================= Indikatoren =============================
def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl   = df["High"] - df["Low"]
    hc   = np.abs(df["High"] - df["Close"].shift())
    lc   = np.abs(df["Low"]  - df["Close"].shift())
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    up   = df["High"].diff()
    down = df["Low"].diff()
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), -down, 0.0), index=df.index)
    tr   = _atr(df, period)
    pdi  = 100 * pdm.rolling(period, min_periods=1).mean() / tr.replace(0, np.nan)
    mdi  = 100 * mdm.rolling(period, min_periods=1).mean() / tr.replace(0, np.nan)
    dx   = (np.abs(pdi - mdi) / (pdi + mdi + 1e-9)) * 100
    return dx.rolling(period, min_periods=1).mean()


def compute_indicators(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    if cfg is None:
        cfg = CONFIG
    df = df.copy()
    # Trend
    df["SMA20"] = df["Close"].rolling(cfg["sma_short"]).mean()
    df["SMA50"] = df["Close"].rolling(cfg["sma_long"]).mean()
    df["ATR"]   = _atr(df, cfg["atr_period"])
    df["ADX"]   = _adx(df, cfg["adx_period"])
    # RSI (14)
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    # MACD
    ema_f = df["Close"].ewm(span=cfg["macd_fast"], adjust=False).mean()
    ema_s = df["Close"].ewm(span=cfg["macd_slow"], adjust=False).mean()
    df["MACD"]       = ema_f - ema_s
    df["MACD_signal"] = df["MACD"].ewm(span=cfg["macd_signal_period"], adjust=False).mean()
    df["MACD_hist"]  = df["MACD"] - df["MACD_signal"]
    # Bollinger Bands
    bb_sma = df["Close"].rolling(cfg["bb_period"]).mean()
    bb_std = df["Close"].rolling(cfg["bb_period"]).std()
    df["BB_upper"] = bb_sma + cfg["bb_std"] * bb_std
    df["BB_lower"] = bb_sma - cfg["bb_std"] * bb_std
    df["BB_mid"]   = bb_sma
    # Volume SMA
    df["Vol_SMA"] = df["Volume"].rolling(cfg["volume_sma_period"]).mean()
    return df


def kalman_estimate(series: pd.Series) -> Tuple[float, float]:
    """Einfacher Kalman-Filter: gibt (Mittelwert, Varianz) zurück."""
    x, p = float(series.iloc[0]), 1.0
    Q, R  = 1e-5, 0.01
    for z in series.iloc[1:]:
        p_pred = p + Q
        k      = p_pred / (p_pred + R)
        x      = x + k * (float(z) - x)
        p      = (1 - k) * p_pred
    return x, p


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Sendet eine Telegram-Nachricht. Gibt True bei Erfolg zurück."""
    if not token or not chat_id:
        return False
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": message},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ============================= Portfolio =============================
class Portfolio:
    def __init__(self, config: dict):
        self.cfg = config
        self.cfg["data_dir"].mkdir(exist_ok=True)
        self.data = self._load()

    # ----------------------------------------------------------------- I/O
    def _load(self) -> dict:
        if self.cfg["portfolio_file"].exists():
            with open(self.cfg["portfolio_file"], "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for _sym, pos in data.get("positions", {}).items():
                pos.setdefault("highest",      pos["entry"])
                pos.setdefault("atr_at_entry", 0.0)
                pos.setdefault("stop",         pos["entry"] - self.cfg["initial_sl_atr_mult"] * pos["atr_at_entry"])
                pos.setdefault("trailing",     pos["highest"] - self.cfg["trailing_atr_mult"] * pos["atr_at_entry"])
                pos.setdefault("partial_sold", False)
                pos.setdefault("strategy",     "trend")
                pos.setdefault("mr_target",    None)
            data.setdefault("short_positions", {})
            data.setdefault("pair_trade",      None)
            data.setdefault("peak_equity",     data.get("cash", self.cfg["initial_capital"]))
            return data

        init = self.cfg["initial_capital"]
        return {
            "cash":             init,
            "initial_capital":  init,
            "peak_equity":      init,
            "positions":        {},
            "short_positions":  {},
            "pair_trade":       None,
            "trades":           [],
            "equity_curve":     [],
        }

    def save(self, price_dict: dict = None, update_equity: bool = True):
        if update_equity:
            lv = (
                sum(p["shares"] * price_dict.get(sym, p.get("price", p["entry"]))
                    for sym, p in self.data["positions"].items())
                if price_dict else 0.0
            )
            sv = (
                sum(p["shares"] * (p["entry"] - price_dict.get(sym, p["entry"]))
                    for sym, p in self.data["short_positions"].items())
                if price_dict else 0.0
            )
            self.data["equity_curve"].append({
                "date":            datetime.now().isoformat(),
                "cash":            self.data["cash"],
                "positions_value": lv + sv,
            })
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.cfg["portfolio_file"], "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, default=str)

    # ----------------------------------------------------------------- helpers
    def has_pos(self, sym: str) -> bool:
        return sym in self.data["positions"]

    def get_pos(self, sym: str) -> Optional[dict]:
        return self.data["positions"].get(sym)

    def equity(self, price_dict: dict) -> float:
        longs  = sum(p["shares"] * price_dict.get(sym, 0)
                     for sym, p in self.data["positions"].items())
        shorts = sum(p["shares"] * (p["entry"] - price_dict.get(sym, p["entry"]))
                     for sym, p in self.data["short_positions"].items())
        return self.data["cash"] + longs + shorts

    def _exec_price(self, price: float, is_buy: bool) -> float:
        slip = self.cfg.get("slippage_pct", 0.0)
        return price * (1 + slip) if is_buy else price * (1 - slip)

    def _commission(self, price: float, shares: int) -> float:
        return price * shares * self.cfg.get("commission_pct", 0.0)

    # ----------------------------------------------------------------- long
    def buy(self, sym: str, price: float, shares: int, atr_val: float,
            reason: str, strategy: str = "trend",
            mr_target: float = None) -> dict:
        ep   = self._exec_price(price, is_buy=True)
        cost = ep * shares + self._commission(ep, shares)
        if cost > self.data["cash"]:
            return {"ok": False, "msg": "not enough cash"}
        self.data["cash"] -= cost
        self.data["positions"][sym] = {
            "symbol":      sym,
            "shares":      shares,
            "entry":       ep,
            "atr_at_entry": atr_val,
            "highest":     ep,
            "stop":        ep - self.cfg["initial_sl_atr_mult"] * atr_val,
            "trailing":    ep - self.cfg["trailing_atr_mult"]   * atr_val,
            "partial_sold": False,
            "strategy":    strategy,
            "mr_target":   mr_target,
            "reason":      reason,
            "price":       ep,
        }
        self._log("BUY", sym, shares, ep, 0.0, reason)
        self.save(update_equity=False)
        return {"ok": True, "pos": self.data["positions"][sym]}

    def sell(self, sym: str, price: float, shares: int, reason: str) -> dict:
        pos = self.data["positions"].get(sym)
        if not pos or pos["shares"] < shares:
            return {"ok": False, "msg": "no position"}
        ep       = self._exec_price(price, is_buy=False)
        proceeds = ep * shares - self._commission(ep, shares)
        pnl      = proceeds - pos["entry"] * shares
        self.data["cash"] += proceeds
        pos["shares"] -= shares
        if pos["shares"] <= 0:
            del self.data["positions"][sym]
        self._log("SELL", sym, shares, ep, pnl, reason)
        self.save(update_equity=False)
        return {"ok": True, "pnl": pnl, "remaining": pos.get("shares", 0)}

    # ----------------------------------------------------------------- short (Pair Trading)
    def short(self, sym: str, price: float, shares: int, reason: str) -> dict:
        ep     = self._exec_price(price, is_buy=False)
        margin = ep * shares + self._commission(ep, shares)
        if margin > self.data["cash"]:
            return {"ok": False, "msg": "not enough cash for margin"}
        self.data["cash"] -= margin
        self.data["short_positions"][sym] = {
            "symbol": sym,
            "shares": shares,
            "entry":  ep,
            "price":  ep,
            "reason": reason,
        }
        self._log("SHORT", sym, shares, ep, 0.0, reason)
        self.save(update_equity=False)
        return {"ok": True}

    def cover(self, sym: str, price: float, reason: str) -> dict:
        pos = self.data["short_positions"].get(sym)
        if not pos:
            return {"ok": False, "msg": "no short position"}
        ep     = self._exec_price(price, is_buy=True)
        shares = pos["shares"]
        pnl    = (pos["entry"] - ep) * shares - self._commission(ep, shares)
        # Margin zurück + PnL
        self.data["cash"] += pos["entry"] * shares + pnl
        del self.data["short_positions"][sym]
        self._log("COVER", sym, shares, ep, pnl, reason)
        self.save(update_equity=False)
        return {"ok": True, "pnl": pnl}

    # ----------------------------------------------------------------- Alpaca live log
    def log_order(self, sym: str, action: str, shares: int,
                  price: float, stop: float, target: float,
                  alpaca_order_id: str = "", reason: str = ""):
        """
        Speichert eine von Alpaca ausgeführte Order lokal.
        Kein virtuelles Cash-Update – Alpaca führt die Bücher.
        """
        record = {
            "time":     datetime.now().isoformat(),
            "symbol":   sym,
            "action":   action,
            "shares":   shares,
            "price":    price,
            "stop":     stop,
            "target":   target,
            "order_id": alpaca_order_id,
            "reason":   reason,
            "strategy": "trader_v6",
            "pnl":      0.0,
        }
        self.data["trades"].append(record)
        self.save(update_equity=False)

    # ----------------------------------------------------------------- trade log
    def _log(self, action: str, sym: str, shares: int,
             price: float, pnl: float, reason: str):
        self.data["trades"].append({
            "time":   datetime.now().isoformat(),
            "symbol": sym,
            "action": action,
            "shares": shares,
            "price":  price,
            "pnl":    pnl,
            "reason": reason,
        })


# ============================= Strategie =============================
class Strategy:
    def __init__(self, config: dict):
        self.cfg = config

    def signal_row(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Gibt (Signal, Begründung) für die letzte Kerze zurück."""
        if len(df) < 2:
            return "HOLD", "not enough data"
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ---- Death Cross: immer SELL, keine weiteren Filter ----
        if prev["SMA20"] >= prev["SMA50"] and last["SMA20"] < last["SMA50"]:
            return "SELL", "Death Cross"

        # ---- Mean Reversion (Bollinger untere Band) ----
        if self.cfg.get("use_mean_reversion", False):
            bb_ok     = pd.notna(last["BB_lower"]) and last["Close"] <= last["BB_lower"]
            rsi_ok    = pd.notna(last["RSI"])       and last["RSI"]   < self.cfg["mr_rsi_max"]
            vol_ok    = pd.notna(last["Vol_SMA"])   and last["Volume"] > last["Vol_SMA"]
            if bb_ok and rsi_ok and vol_ok:
                return "BUY_MR", f"BB Lower Touch | RSI={last['RSI']:.0f}"

        # ---- Uptrend-Filter (nur für Trend-Signale) ----
        if not (last["SMA20"] > last["SMA50"]):
            return "HOLD", "no uptrend"
        if pd.notna(last["ADX"]) and last["ADX"] < self.cfg["adx_threshold"]:
            return "HOLD", f"ADX {last['ADX']:.1f} < {self.cfg['adx_threshold']}"

        # ---- Golden Cross ----
        golden = prev["SMA20"] <= prev["SMA50"] and last["SMA20"] > last["SMA50"]
        if not golden:
            return "HOLD", ""

        # ---- Bestätigungsfilter (nur bei Golden Cross) ----
        if self.cfg.get("use_rsi_filter", True):
            rsi = last["RSI"]
            if pd.isna(rsi) or not (self.cfg["rsi_buy_min"] <= rsi <= self.cfg["rsi_buy_max"]):
                return "HOLD", f"RSI {rsi:.0f} außerhalb [{self.cfg['rsi_buy_min']}, {self.cfg['rsi_buy_max']}]"

        if self.cfg.get("use_macd_filter", True):
            if pd.isna(last["MACD_hist"]) or last["MACD_hist"] <= 0:
                return "HOLD", f"MACD-Histogramm negativ"

        if self.cfg.get("use_volume_filter", True):
            if pd.isna(last["Vol_SMA"]) or last["Volume"] < last["Vol_SMA"]:
                return "HOLD", "Volume unter Durchschnitt"

        return "BUY", "Golden Cross + MACD + Volume bestätigt"

    def pair_signal(self, df: pd.DataFrame) -> Tuple[str, float, str]:
        """Gibt (Signal, Z-Score, Begründung) für SPY/QQQ zurück."""
        sym1, sym2 = self.cfg["pair"]
        if sym1 not in df.columns or sym2 not in df.columns:
            return "HOLD", 0.0, "missing pair data"
        spread = df[sym2] - df[sym1]
        vol    = spread.rolling(10).std().iloc[-1]
        base   = self.cfg["pair_lookback_base"]
        lk     = max(10, min(50, int(base * (1 + vol / (abs(spread.mean()) + 1e-9)))))
        sd     = spread.rolling(lk).std().replace(0, np.nan)
        mu, _  = kalman_estimate(spread.tail(50))
        z      = (spread.iloc[-1] - mu) / (sd.iloc[-1] + 1e-9)
        if z >  self.cfg["pair_z_entry"]:
            return "SHORT_QQQ_LONG_SPY",  float(z), f"Z={z:.2f}"
        if z < -self.cfg["pair_z_entry"]:
            return "LONG_QQQ_SHORT_SPY",  float(z), f"Z={z:.2f}"
        if abs(z) < self.cfg["pair_z_exit"]:
            return "FLAT",  float(z), f"Z={z:.2f} innerhalb Exit-Band"
        return "HOLD", float(z), ""

    def ml_prob(self, features: dict,
                ml_model=None, ml_scaler=None) -> float:
        """ML-Wahrscheinlichkeit. Modell + Scaler werden übergeben."""
        if ml_model is None or ml_scaler is None:
            return 0.5
        X = np.array([[
            features["sma_diff"],
            features["adx"],
            features["atr_pct"],
            features["rsi"],
            features["macd_hist"],
        ]])
        return float(ml_model.predict_proba(ml_scaler.transform(X))[0, 1])


# ============================= Backtest =============================
class Backtester:
    """
    Einfacher tages-bar-basierter Backtest ohne Look-Ahead-Bias.
    Führt dieselbe Strategie wie Trader.run_daily() aus.
    Nutzt Alpaca für historische Daten wenn alpaca-Client vorhanden.
    """

    def __init__(self, config: dict, alpaca: "AlpacaClient" = None):
        self.cfg   = config
        self.strat = Strategy(config)
        self.alpaca = alpaca

    def run(self, symbols: List[str], period: str = "800d") -> dict:
        print(f"\n=== Backtest ({period}) – {len(symbols)} Symbole ===")
        raw: Dict[str, pd.DataFrame] = {}

        # ── Daten holen: Alpaca bevorzugt, yfinance als Fallback ─────────────
        if self.alpaca:
            days = int(period.replace("d", ""))
            end_date   = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            print("Lade Daten via Alpaca...")
            raw_bulk = self.alpaca.fetch_bars_bulk(symbols, start_date, end_date)
            for sym, df in raw_bulk.items():
                if len(df) >= 60:
                    raw[sym] = compute_indicators(df, self.cfg)
        else:
            for sym in symbols:
                df = yf.Ticker(sym).history(period=period)
                if df.empty or len(df) < 60:
                    continue
                raw[sym] = compute_indicators(df[["Open", "High", "Low", "Close", "Volume"]], self.cfg)

        cash      = self.cfg["initial_capital"]
        positions: Dict[str, dict] = {}
        trades:    List[dict]      = []
        eq_curve:  List[dict]      = []
        peak_eq   = cash
        min_bars  = self.cfg["sma_long"] + 10

        # Vereinigung aller Handelsdaten
        all_dates = sorted(set().union(*[set(df.index) for df in raw.values()]))

        for date in all_dates:
            price_dict = {
                sym: float(raw[sym].loc[date, "Close"])
                for sym in raw if date in raw[sym].index
            }
            # --- bestehende Positionen verwalten ---
            for sym in list(positions.keys()):
                if sym not in price_dict:
                    continue
                pos   = positions[sym]
                price = price_dict[sym]
                # Highest + Trailing aktualisieren
                if price > pos["highest"]:
                    pos["highest"]  = price
                    new_trail = price - self.cfg["trailing_atr_mult"] * pos["atr"]
                    pos["trailing"] = max(pos["trailing"], new_trail)
                # MR-Gewinnziel
                if pos.get("strategy") == "mr" and pos.get("mr_target") and price >= pos["mr_target"]:
                    pnl = (price - pos["entry"]) * pos["shares"]
                    cash += price * pos["shares"]
                    trades.append(self._trade(date, sym, "SELL", price, pnl, "MR Target"))
                    del positions[sym]
                    continue
                # Partial Profit
                if not pos["partial"] and price >= pos["entry"] * (1 + self.cfg["partial_profit_pct"]):
                    half = pos["shares"] // 2
                    if half > 0:
                        cash += price * half
                        pnl   = (price - pos["entry"]) * half
                        pos["shares"] -= half
                        pos["partial"] = True
                        trades.append(self._trade(date, sym, "PARTIAL_SELL", price, pnl, "Partial Profit"))
                        continue
                # Signal-Exit / Stops
                df_sym = raw[sym]
                i_loc  = df_sym.index.get_loc(date) if date in df_sym.index else -1
                if i_loc >= 2:
                    sig, reason = self.strat.signal_row(df_sym.iloc[:i_loc + 1])
                    if sig == "SELL" or price <= pos["trailing"] or price <= pos["stop"]:
                        ex = "Death Cross" if sig == "SELL" else ("Trailing" if price <= pos["trailing"] else "Stop-Loss")
                        pnl = (price - pos["entry"]) * pos["shares"]
                        cash += price * pos["shares"]
                        trades.append(self._trade(date, sym, "SELL", price, pnl, ex))
                        del positions[sym]

            # --- neue Signale ---
            for sym in symbols:
                if sym in positions:
                    continue
                df_sym = raw.get(sym)
                if df_sym is None:
                    continue
                i_loc = df_sym.index.get_loc(date) if date in df_sym.index else -1
                if i_loc < min_bars:
                    continue
                sub          = df_sym.iloc[:i_loc + 1]
                sig, reason  = self.strat.signal_row(sub)
                if sig not in ("BUY", "BUY_MR"):
                    continue
                price   = price_dict.get(sym, 0)
                atr_val = float(sub["ATR"].iloc[-1])
                if not price or np.isnan(atr_val) or atr_val == 0:
                    continue
                eq          = cash + sum(positions[s]["shares"] * price_dict.get(s, 0) for s in positions)
                risk_ps     = self.cfg["initial_sl_atr_mult"] * atr_val
                sh_risk     = int((eq * self.cfg["risk_per_trade"]) / risk_ps)
                sh_cash     = int(cash * 0.4 / price)
                shares      = min(sh_risk, sh_cash)
                if shares <= 0 or shares * price > cash:
                    continue
                cash -= shares * price
                mr_target = float(sub["BB_mid"].iloc[-1]) if sig == "BUY_MR" else None
                positions[sym] = {
                    "shares":   shares,
                    "entry":    price,
                    "atr":      atr_val,
                    "highest":  price,
                    "stop":     price - self.cfg["initial_sl_atr_mult"] * atr_val,
                    "trailing": price - self.cfg["trailing_atr_mult"]   * atr_val,
                    "partial":  False,
                    "strategy": "mr" if sig == "BUY_MR" else "trend",
                    "mr_target": mr_target,
                }
                trades.append(self._trade(date, sym, "BUY", price, 0, reason))

            # Equity-Snapshot
            eq = cash + sum(positions[s]["shares"] * price_dict.get(s, 0) for s in positions)
            eq_curve.append({"date": str(date), "equity": eq})
            peak_eq = max(peak_eq, eq)

        return self._summary(trades, eq_curve)

    @staticmethod
    def _trade(date, sym, action, price, pnl, reason) -> dict:
        return {"date": str(date), "sym": sym, "action": action,
                "price": price, "pnl": pnl, "reason": reason}

    def _summary(self, trades: list, eq_curve: list) -> dict:
        if not eq_curve:
            return {}
        init     = self.cfg["initial_capital"]
        final_eq = eq_curve[-1]["equity"]
        n_days   = len(eq_curve)
        cagr     = (final_eq / init) ** (252 / max(n_days, 1)) - 1
        eqs      = pd.Series([e["equity"] for e in eq_curve])
        rets     = eqs.pct_change().dropna()
        sharpe   = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        max_dd   = float(((eqs - eqs.cummax()) / eqs.cummax()).min())
        wins     = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / max(len(trades), 1)
        r = {
            "initial_capital":  init,
            "final_equity":     final_eq,
            "total_return_pct": (final_eq / init - 1) * 100,
            "cagr_pct":         cagr * 100,
            "sharpe":           sharpe,
            "max_drawdown_pct": max_dd * 100,
            "total_trades":     len(trades),
            "win_rate_pct":     win_rate * 100,
        }
        print(f"  Trades total  : {r['total_trades']}")
        print(f"  Final equity  : {r['final_equity']:>12,.2f}")
        print(f"  Total return  : {r['total_return_pct']:>+11.2f}%")
        print(f"  CAGR          : {r['cagr_pct']:>+11.2f}%")
        print(f"  Sharpe Ratio  : {r['sharpe']:>11.2f}")
        print(f"  Max Drawdown  : {r['max_drawdown_pct']:>11.2f}%")
        print(f"  Win Rate      : {r['win_rate_pct']:>11.1f}%")
        print("=== Backtest abgeschlossen ===\n")
        return r


# ============================= ML Training =============================
def train_ml_model(symbols: List[str], cfg: dict, lookforward: int = 20) -> bool:
    """
    Erstellt Trainingsdaten: Features bei jedem BUY/BUY_MR-Signal,
    Label = 1 wenn Preis nach lookforward Tagen höher als Entry.
    Speichert Modell + Scaler unter ml_model/.
    """
    if not HAS_SKLEARN:
        print("sklearn nicht installiert – ML-Training abgebrochen.")
        return False

    print("Erstelle ML-Trainingsdaten ...")
    strat = Strategy(cfg)
    all_X, all_y = [], []

    for sym in symbols:
        df = yf.Ticker(sym).history(period="1500d")
        if df.empty or len(df) < 100:
            continue
        df = compute_indicators(df[["Open", "High", "Low", "Close", "Volume"]], cfg)
        for i in range(70, len(df) - lookforward):
            sub = df.iloc[:i + 1]
            sig, _ = strat.signal_row(sub)
            if sig not in ("BUY", "BUY_MR"):
                continue
            last    = sub.iloc[-1]
            fut     = df.iloc[i + lookforward]["Close"]
            label   = 1 if fut > last["Close"] else 0
            atr_pct = (last["ATR"] / last["Close"]) if last["Close"] > 0 else 0
            all_X.append([
                (last["SMA20"] - last["SMA50"]) / (last["Close"] + 1e-9),
                last["ADX"]       if pd.notna(last["ADX"])       else 0.0,
                atr_pct,
                last["RSI"]       if pd.notna(last["RSI"])       else 50.0,
                last["MACD_hist"] if pd.notna(last["MACD_hist"]) else 0.0,
            ])
            all_y.append(label)

    if len(all_X) < 50:
        print(f"Nur {len(all_X)} Samples – Training abgebrochen.")
        return False

    X = np.array(all_X)
    y = np.array(all_y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr)
    X_te_s    = scaler.transform(X_te)

    if HAS_LGB:
        model      = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                        num_leaves=31, random_state=42, verbose=-1)
        model_name = "lgbm_model.pkl"
    else:
        model      = LogisticRegression(max_iter=1000, random_state=42)
        model_name = "logistic_model.pkl"

    model.fit(X_tr_s, y_tr)
    print(classification_report(y_te, model.predict(X_te_s)))

    ml_dir = Path(__file__).parent / "ml_model"
    ml_dir.mkdir(exist_ok=True)
    joblib.dump(model,  ml_dir / model_name)
    joblib.dump(scaler, ml_dir / "scaler.pkl")
    print(f"Modell gespeichert: {ml_dir / model_name}")
    return True


# ============================= Trader (Hauptmaschine) =============================
class Trader:
    def __init__(self, config: Optional[dict] = None,
                 alpaca: Optional[AlpacaClient] = None):
        self.cfg         = config or CONFIG
        self.alpaca      = alpaca
        self.portfolio   = Portfolio(self.cfg)
        self._seed_nvda_if_fresh()
        self.strategy    = Strategy(self.cfg)
        self.reports_dir = self.cfg["data_dir"] / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.ml_model  = None
        self.ml_scaler = None
        if self.cfg.get("use_ml", False):
            self._load_ml()

        mode = "PAPER" if (alpaca and alpaca.paper) else "LIVE ⚠️" if alpaca else "SIMULATION (yfinance)"
        print(f"Botti Trader v6.0  Modus={mode}  Symbole={len(self.cfg['symbols'])}")

    # ----------------------------------------------------------------- Init-Helfer
    def _seed_nvda_if_fresh(self):
        """Setzt eine initiale NVDA-Position wenn Portfolio leer ist."""
        if self.portfolio.data["positions"] or \
           self.portfolio.data["cash"] != self.cfg["initial_capital"]:
            return
        try:
            df = yf.Ticker("NVDA").history(start="2025-12-01", end="2026-03-24")
            if df.empty:
                return
            df      = df[["Open", "High", "Low", "Close", "Volume"]]
            price   = float(df["Close"].iloc[-1])
            shares  = 6
            if price * shares > self.portfolio.data["cash"]:
                return
            df_ind  = compute_indicators(df, self.cfg)
            atr_val = float(df_ind["ATR"].iloc[-1])
            if np.isnan(atr_val):
                atr_val = price * 0.02
            self.portfolio.data["cash"] -= price * shares
            self.portfolio.data["positions"]["NVDA"] = {
                "symbol":      "NVDA",
                "shares":      shares,
                "entry":       price,
                "atr_at_entry": atr_val,
                "highest":     price,
                "stop":        price - self.cfg["initial_sl_atr_mult"] * atr_val,
                "trailing":    price - self.cfg["trailing_atr_mult"]   * atr_val,
                "partial_sold": False,
                "strategy":    "seed",
                "mr_target":   None,
                "reason":      "Seed position",
                "price":       price,
            }
            self.portfolio._log("BUY", "NVDA", shares, price, 0.0, "Seed position")
            self.portfolio.save()
        except Exception:
            pass

    def _load_ml(self):
        ml_dir = Path(__file__).parent / "ml_model"
        for name in ("lgbm_model.pkl", "logistic_model.pkl"):
            mp, sp = ml_dir / name, ml_dir / "scaler.pkl"
            if mp.exists() and sp.exists():
                try:
                    self.ml_model  = joblib.load(mp)
                    self.ml_scaler = joblib.load(sp)
                    print(f"ML-Modell geladen: {name}")
                    return
                except Exception as e:
                    print(f"ML-Laden fehlgeschlagen: {e}")
        print("Keine ML-Modelldateien gefunden – ML deaktiviert.")

    # ----------------------------------------------------------------- Daten
    def _fetch(self, symbol: str, period: str = "150d") -> pd.DataFrame:
        """Holt Daily-Bars: Alpaca bevorzugt, yfinance als Fallback."""
        if self.alpaca:
            days = int(period.replace("d", ""))
            df   = self.alpaca.fetch_bars(symbol, days=days)
            if not df.empty:
                return df
        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            return pd.DataFrame()
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _fetch_pair(self) -> pd.DataFrame:
        s1, s2 = self.cfg["pair"]
        if self.alpaca:
            end_date   = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
            bulk = self.alpaca.fetch_bars_bulk([s1, s2], start_date, end_date)
            if s1 in bulk and s2 in bulk:
                df = pd.concat(
                    {s1: bulk[s1]["Close"], s2: bulk[s2]["Close"]}, axis=1
                ).dropna()
                return df
        df = yf.download([s1, s2], period="400d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        return df

    # ----------------------------------------------------------------- Hilfsfunktionen
    def _get_vix(self) -> float:
        try:
            df = yf.Ticker("^VIX").history(period="5d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return 20.0

    def _get_benchmark_return(self) -> Optional[float]:
        bm = self.cfg.get("benchmark", "SPY")
        try:
            df = yf.Ticker(bm).history(period="400d")
            if df.empty:
                return None
            ec = self.portfolio.data.get("equity_curve", [])
            if not ec:
                return None
            start = ec[0]["date"][:10]
            df_s  = df[df.index >= start]
            if df_s.empty:
                return None
            return (float(df["Close"].iloc[-1]) / float(df_s["Close"].iloc[0]) - 1) * 100
        except Exception:
            return None

    def _drawdown_active(self, equity: float) -> bool:
        """True wenn Circuit Breaker aktiv ist."""
        peak = self.portfolio.data.get("peak_equity", equity)
        if equity > peak:
            self.portfolio.data["peak_equity"] = equity
        return (peak - equity) / peak >= self.cfg.get("max_drawdown_pct", 0.15)

    def _sector_count(self, sym: str) -> int:
        for members in self.cfg.get("sector_groups", {}).values():
            if sym in members:
                return sum(1 for s in members if self.portfolio.has_pos(s))
        return 0

    def _volume_ok(self, df: pd.DataFrame, shares: int) -> bool:
        avg = df["Volume"].tail(20).mean()
        return avg <= 0 or (shares / avg) <= self.cfg.get("max_volume_pct", 0.01)

    def _alert(self, msg: str):
        send_telegram(
            self.cfg.get("telegram_token", ""),
            self.cfg.get("telegram_chat_id", ""),
            msg,
        )

    # ----------------------------------------------------------------- Hauptloop
    def run_daily(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"\n=== Botti Trader v6.0 – {today} ===")

        # VIX-Regime
        vix        = self._get_vix() if self.cfg.get("use_vix_filter", True) else 20.0
        vix_reduce = vix >= self.cfg.get("vix_high_threshold", 30)
        size_mult  = self.cfg.get("vix_size_reduction", 0.5) if vix_reduce else 1.0
        if vix_reduce:
            print(f"  [VIX={vix:.1f}] Hochvolatiles Regime – Positionsgrößen halbiert")

        # Alpaca-Positionen holen (verhindert Doppel-Entries im Live-Betrieb)
        open_positions = self.alpaca.sync_positions() if self.alpaca else {}

        # 1) Alle Daten holen & Exits verwalten
        price_dict:  Dict[str, float]          = {}
        sym_to_df:   Dict[str, pd.DataFrame]   = {}
        current_risk = 0.0

        for sym in self.cfg["symbols"]:
            df = self._fetch(sym, period="150d")
            if df.empty:
                continue
            df = compute_indicators(df, self.cfg)
            sym_to_df[sym] = df
            price = float(df["Close"].iloc[-1])
            price_dict[sym] = price

            # Im Live-Betrieb: bereits offene Alpaca-Positionen anzeigen
            if self.alpaca and sym in open_positions:
                pos = open_positions[sym]
                print(f"  {sym}: offen {pos['side'].upper()} {pos['qty']} "
                      f"@ {pos['entry']:.2f}  uPnL {pos['unrealized_pnl']:+.2f}")
                continue

            if not self.portfolio.has_pos(sym):
                continue

            pos = self.portfolio.get_pos(sym)
            pos["price"] = price
            risk_this = pos["shares"] * max(pos["entry"] - pos["stop"], 0.0)
            current_risk += risk_this

            # Highest & Trailing aktualisieren
            if price > pos["highest"]:
                pos["highest"] = price
                new_trail = price - self.cfg["trailing_atr_mult"] * pos["atr_at_entry"]
                if new_trail > pos["trailing"]:
                    pos["trailing"] = new_trail

            # Pair-Positionen: nur durch Pair-Exit schließen
            if pos.get("strategy") == "pair":
                continue

            exited = False

            # MR-Gewinnziel
            if not exited and pos.get("strategy") == "mr" and pos.get("mr_target"):
                if price >= pos["mr_target"]:
                    res = self.portfolio.sell(sym, price, pos["shares"], "MR Target")
                    if res["ok"]:
                        if self.alpaca:
                            self.alpaca.close_position(sym)
                        print(f"  {sym}: MR TARGET SELL @ {price:.2f} (PNL {res['pnl']:.2f})")
                        self._alert(f"[Botti v6] {sym} MR TARGET @ {price:.2f}  PNL {res['pnl']:.2f}")
                        exited = True

            # Partial Profit (+25%)
            if not exited and not pos["partial_sold"] and \
               price >= pos["entry"] * (1 + self.cfg["partial_profit_pct"]):
                half = pos["shares"] // 2
                if half > 0:
                    res = self.portfolio.sell(sym, price, half, "Partial Profit")
                    if res["ok"]:
                        if self.alpaca:
                            self.alpaca.reduce_position(sym, half)
                        print(f"  {sym}: PARTIAL SELL {half} @ {price:.2f} (PNL {res['pnl']:.2f})")
                        self._alert(f"[Botti v6] {sym} PARTIAL SELL {half} @ {price:.2f}")
                        pos["partial_sold"] = True
                        exited = True

            # Trailing Stop
            if not exited and price <= pos["trailing"]:
                res = self.portfolio.sell(sym, price, pos["shares"], "Trailing Stop")
                if res["ok"]:
                    if self.alpaca:
                        self.alpaca.close_position(sym)
                    print(f"  {sym}: TRAILING STOP @ {price:.2f} (PNL {res['pnl']:.2f})")
                    self._alert(f"[Botti v6] {sym} TRAILING STOP @ {price:.2f}  PNL {res['pnl']:.2f}")
                    exited = True

            # Stop-Loss
            if not exited and price <= pos["stop"]:
                res = self.portfolio.sell(sym, price, pos["shares"], "Stop-Loss")
                if res["ok"]:
                    if self.alpaca:
                        self.alpaca.close_position(sym)
                    print(f"  {sym}: STOP-LOSS @ {price:.2f} (PNL {res['pnl']:.2f})")
                    self._alert(f"[Botti v6] {sym} STOP-LOSS @ {price:.2f}  PNL {res['pnl']:.2f}")
                    exited = True  # noqa: F841

        # Equity: im Live-Betrieb von Alpaca holen
        equity = self.alpaca.get_equity() if self.alpaca else self.portfolio.equity(price_dict)

        # 2) Circuit Breaker prüfen
        if self._drawdown_active(equity):
            peak    = self.portfolio.data["peak_equity"]
            dd_pct  = (peak - equity) / peak * 100
            print(f"  [CIRCUIT BREAKER] Drawdown {dd_pct:.1f}% >= "
                  f"{self.cfg['max_drawdown_pct']*100:.0f}% – keine neuen Trades")
            self._alert(f"[Botti v6] CIRCUIT BREAKER – Drawdown {dd_pct:.1f}%")
        else:
            # 3) Neue Signale + Death-Cross-Exits
            for sym in self.cfg["symbols"]:
                df = sym_to_df.get(sym)
                if df is None or df.empty or len(df) < self.cfg["sma_long"] + 5:
                    continue
                signal, reason = self.strategy.signal_row(df)
                price   = price_dict.get(sym, 0.0)
                atr_val = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else 0.0

                # Im Live-Betrieb: bereits bei Alpaca offene Positionen überspringen
                if self.alpaca and sym in open_positions:
                    continue

                # Death Cross Exit
                if signal == "SELL" and self.portfolio.has_pos(sym):
                    pos = self.portfolio.get_pos(sym)
                    res = self.portfolio.sell(sym, price, pos["shares"], reason)
                    if res["ok"]:
                        if self.alpaca:
                            self.alpaca.close_position(sym)
                        print(f"  {sym}: DEATH CROSS SELL @ {price:.2f} (PNL {res['pnl']:.2f})")
                        self._alert(f"[Botti v6] {sym} DEATH CROSS SELL @ {price:.2f}")
                    continue

                # Re-Entry nach Partial Profit
                if self.portfolio.has_pos(sym):
                    if self.cfg.get("allow_reentry", False):
                        pos  = self.portfolio.get_pos(sym)
                        last = df.iloc[-1]
                        reentry_threshold = pos["highest"] - self.cfg["reentry_atr_mult"] * pos["atr_at_entry"]
                        if (pos["partial_sold"]
                                and last["SMA20"] > last["SMA50"]
                                and price <= reentry_threshold
                                and atr_val > 0):
                            re_shares = max(pos["shares"], 1)
                            if re_shares * price <= self.portfolio.data["cash"] * 0.20:
                                res = self.portfolio.buy(sym, price, re_shares, atr_val,
                                                         "Re-Entry nach Partial Profit", "trend")
                                if res["ok"]:
                                    if self.alpaca:
                                        stop   = price - self.cfg["initial_sl_atr_mult"] * atr_val
                                        target = price + self.cfg["trailing_atr_mult"] * atr_val * 2
                                        order  = self.alpaca.place_long_bracket(sym, re_shares, stop, target)
                                        if order:
                                            self.portfolio.log_order(sym, "BUY", re_shares, price,
                                                                      stop, target, order["id"],
                                                                      "Re-Entry nach Partial Profit")
                                    print(f"  {sym}: RE-ENTRY BUY {re_shares} @ {price:.2f}")
                    continue

                # Kein Signal → skip
                if signal not in ("BUY", "BUY_MR"):
                    continue

                # Portfolio-Heat-Check
                equity = self.alpaca.get_equity() if self.alpaca else self.portfolio.equity(price_dict)
                if current_risk > equity * self.cfg["max_equity_at_risk"]:
                    print(f"  {sym}: SKIP – Portfolio Heat {current_risk / equity * 100:.1f}%")
                    continue

                # Sektor-Guard
                if self._sector_count(sym) >= self.cfg.get("max_per_sector", 2):
                    print(f"  {sym}: SKIP – Sektor-Limit erreicht")
                    continue

                if atr_val == 0 or price == 0:
                    continue

                # Positionsgrößen
                rps        = self.cfg["initial_sl_atr_mult"] * atr_val
                sh_risk    = int((equity * self.cfg["risk_per_trade"]) / rps)
                sh_cash    = int((self.alpaca.get_cash() if self.alpaca
                                  else self.portfolio.data["cash"]) * 0.4 / price)
                shares     = int(min(sh_risk, sh_cash) * size_mult)
                if shares <= 0:
                    continue

                # Volumen-Guard
                if not self._volume_ok(df, shares):
                    print(f"  {sym}: SKIP – Order zu groß vs. Tagesvolumen")
                    continue

                # ML-Filter
                if self.cfg.get("use_ml", False):
                    last = df.iloc[-1]
                    feats = {
                        "sma_diff":  (last["SMA20"] - last["SMA50"]) / (last["Close"] + 1e-9),
                        "rsi":       last["RSI"]       if pd.notna(last["RSI"])       else 50.0,
                        "adx":       last["ADX"]       if pd.notna(last["ADX"])       else 0.0,
                        "atr_pct":   atr_val / last["Close"],
                        "macd_hist": last["MACD_hist"] if pd.notna(last["MACD_hist"]) else 0.0,
                    }
                    prob = self.strategy.ml_prob(feats, self.ml_model, self.ml_scaler)
                    if prob < self.cfg["ml_prob_threshold"]:
                        print(f"  {sym}: ML-Filter reject (prob={prob:.2f})")
                        continue

                mr_target  = float(df.iloc[-1]["BB_mid"]) if signal == "BUY_MR" else None
                strategy_t = "mr" if signal == "BUY_MR" else "trend"
                stop_price = price - self.cfg["initial_sl_atr_mult"] * atr_val
                tp_price   = (mr_target if mr_target
                              else price + self.cfg["trailing_atr_mult"] * atr_val * 2)

                if self.alpaca:
                    # Live: Alpaca-Bracket-Order, lokales Log-Eintrag
                    order = self.alpaca.place_long_bracket(sym, shares,
                                                           round(stop_price, 2),
                                                           round(tp_price, 2))
                    if order is None:
                        continue
                    self.portfolio.log_order(sym, "BUY", shares, price,
                                             stop_price, tp_price,
                                             alpaca_order_id=order["id"],
                                             reason=reason)
                    print(f"  {sym}: BUY {shares} @ {price:.2f} [{strategy_t.upper()}] – {reason}")
                    self._alert(f"[Botti v6] {sym} BUY {shares} @ {price:.2f} ({reason})")
                else:
                    # Simulation: lokale virtuelle Execution
                    res = self.portfolio.buy(sym, price, shares, atr_val, reason, strategy_t, mr_target)
                    if res["ok"]:
                        print(f"  {sym}: BUY {shares} @ {price:.2f} [{strategy_t.upper()}] – {reason}")
                        self._alert(f"[Botti v6] {sym} BUY {shares} @ {price:.2f} ({reason})")
                        current_risk += shares * self.cfg["initial_sl_atr_mult"] * atr_val

            # 4) Pair-Trading Execution
            pair_df = self._fetch_pair()
            if not pair_df.empty:
                equity = self.alpaca.get_equity() if self.alpaca else self.portfolio.equity(price_dict)
                self._execute_pair_trade(pair_df, price_dict, equity)

        # 5) Tagesabschluss
        equity  = self.alpaca.get_equity() if self.alpaca else self.portfolio.equity(price_dict)
        self.portfolio.save(price_dict, update_equity=True)
        init    = self.cfg["initial_capital"]
        ret     = (equity / init - 1) * 100
        bm_ret  = self._get_benchmark_return()
        bm_str  = (f" | Benchmark {self.cfg['benchmark']}: {bm_ret:+.2f}%"
                   if bm_ret is not None else "")
        print(f"  Equity: {equity:,.2f} {self.cfg['currency']} (Return {ret:+.2f}%{bm_str})")
        self._generate_report(today, price_dict, equity, vix, bm_ret)
        return {
            "date":         today,
            "equity":       equity,
            "return_pct":   ret,
            "vix":          vix,
            "positions":    list(self.portfolio.data["positions"].keys()),
            "trades_today": len([t for t in self.portfolio.data["trades"]
                                 if t["time"].startswith(today)]),
        }

    # ----------------------------------------------------------------- Status (Alpaca)
    def get_status(self) -> dict:
        """Portfolio-Status – nutzt Alpaca-Daten im Live-Betrieb."""
        positions = self.alpaca.sync_positions() if self.alpaca else self.portfolio.data["positions"]
        equity    = self.alpaca.get_equity()     if self.alpaca else 0.0
        orders    = self.alpaca.get_open_orders()if self.alpaca else []
        return {
            "mode":           "PAPER" if (self.alpaca and self.alpaca.paper) else
                              "LIVE"  if self.alpaca else "SIMULATION",
            "equity":         equity,
            "cash":           self.alpaca.get_cash()         if self.alpaca else self.portfolio.data["cash"],
            "buying_power":   self.alpaca.get_buying_power() if self.alpaca else 0.0,
            "open_positions": positions,
            "open_orders":    orders,
            "trades_today":   len([t for t in self.portfolio.data["trades"]
                                   if t["time"].startswith(datetime.now().strftime("%Y-%m-%d"))]),
        }

    # ----------------------------------------------------------------- Pair-Trade Execution
    def _execute_pair_trade(self, pair_df: pd.DataFrame,
                            price_dict: dict, equity: float):
        s1, s2         = self.cfg["pair"]
        signal, z, msg = self.strategy.pair_signal(pair_df)
        active         = self.portfolio.data.get("pair_trade")
        alloc          = equity * self.cfg.get("pair_position_pct", 0.05)

        # Offenen Pair-Trade schließen
        if signal == "FLAT" and active:
            long_sym  = active.get("long")
            short_sym = active.get("short")
            if long_sym and self.portfolio.has_pos(long_sym):
                pos = self.portfolio.get_pos(long_sym)
                self.portfolio.sell(long_sym, price_dict.get(long_sym, pos["entry"]),
                                    pos["shares"], "Pair Exit")
                if self.alpaca:
                    self.alpaca.close_position(long_sym)
            if short_sym and short_sym in self.portfolio.data["short_positions"]:
                self.portfolio.cover(short_sym,
                                     price_dict.get(short_sym, 0), "Pair Exit")
                if self.alpaca:
                    self.alpaca.close_position(short_sym)
            self.portfolio.data["pair_trade"] = None
            print(f"  PAIR: Geschlossen – {msg}")
            self._alert(f"[Botti v6] PAIR EXIT ({msg})")
            return

        # Neuen Pair-Trade eröffnen (nur wenn kein aktiver vorhanden)
        if signal in ("SHORT_QQQ_LONG_SPY", "LONG_QQQ_SHORT_SPY") and not active:
            long_sym  = s1 if signal == "SHORT_QQQ_LONG_SPY" else s2
            short_sym = s2 if signal == "SHORT_QQQ_LONG_SPY" else s1
            lp        = price_dict.get(long_sym, 0)
            sp        = price_dict.get(short_sym, 0)
            if lp <= 0 or sp <= 0:
                return
            sh_long  = int(alloc / lp)
            sh_short = int(alloc / sp)
            if sh_long <= 0 or sh_short <= 0:
                return
            # ATR-Schätzung für Long-Seite (1.5% des Preises als Näherung)
            atr_est = lp * 0.015

            if self.alpaca:
                # Live: Bracket-Orders via Alpaca
                stop_l  = round(lp - 2 * atr_est, 2)
                tp_l    = round(lp + 3 * atr_est, 2)
                stop_s  = round(sp + 2 * sp * 0.015, 2)
                tp_s    = round(sp - 3 * sp * 0.015, 2)
                r_long  = self.alpaca.place_long_bracket(long_sym,  sh_long,  stop_l, tp_l)
                r_short = self.alpaca.place_short_bracket(short_sym, sh_short, stop_s, tp_s)
                ok_long  = r_long  is not None
                ok_short = r_short is not None
                if ok_long:
                    self.portfolio.log_order(long_sym,  "BUY",   sh_long,  lp, stop_l, tp_l,
                                             r_long["id"],  f"Pair Z={z:.2f}")
                if ok_short:
                    self.portfolio.log_order(short_sym, "SHORT", sh_short, sp, stop_s, tp_s,
                                             r_short["id"], f"Pair Z={z:.2f}")
            else:
                # Simulation
                r_long  = self.portfolio.buy(
                    long_sym, lp, sh_long, atr_est, f"Pair Z={z:.2f}", "pair"
                )
                r_short = self.portfolio.short(
                    short_sym, sp, sh_short, f"Pair Z={z:.2f}"
                )
                ok_long  = r_long["ok"]
                ok_short = r_short["ok"]

            if ok_long and ok_short:
                self.portfolio.data["pair_trade"] = {
                    "long":    long_sym,
                    "short":   short_sym,
                    "z_entry": z,
                }
                print(f"  PAIR: LONG {long_sym} + SHORT {short_sym} (Z={z:.2f})")
                self._alert(f"[Botti v6] PAIR: LONG {long_sym} SHORT {short_sym} Z={z:.2f}")
            else:
                # Rollback wenn nur eine Seite geklappt hat (Simulation)
                if ok_long and not ok_short and not self.alpaca:
                    pos = self.portfolio.get_pos(long_sym)
                    if pos:
                        self.portfolio.sell(long_sym, lp, pos["shares"], "Pair Rollback")
        elif signal not in ("HOLD", "FLAT"):
            print(f"  PAIR {s1}/{s2}: {signal} (Z={z:.2f}) – {msg}")

    # ----------------------------------------------------------------- Report
    def _generate_report(self, date_str: str, price_dict: dict,
                         equity: float, vix: float = 0.0,
                         bm_ret: Optional[float] = None):
        report_file = self.reports_dir / f"report_{date_str}.txt"
        init        = self.cfg["initial_capital"]
        mode        = "PAPER" if (self.alpaca and self.alpaca.paper) else \
                      "LIVE"  if self.alpaca else "SIMULATION"
        lines       = [
            "=" * 68,
            f"BOTTI TRADER v6.0  –  TAGESBERICHT  –  {date_str}",
            f"Modus: {mode}",
            "=" * 68,
            "",
            f"  Startkapital     : {init:>14,.2f} {self.cfg['currency']}",
            f"  Aktuelles Equity : {equity:>14,.2f} {self.cfg['currency']}",
            f"  Return (Bot)     : {(equity / init - 1) * 100:>+13.2f}%",
        ]
        if bm_ret is not None:
            alpha = (equity / init - 1) * 100 - bm_ret
            lines += [
                f"  Return ({self.cfg['benchmark']} BnH)  : {bm_ret:>+13.2f}%",
                f"  Alpha vs {self.cfg['benchmark']}      : {alpha:>+13.2f}%",
            ]
        lines += [
            f"  VIX              : {vix:>14.1f}",
            "",
            "Offene Long Positionen:",
        ]
        if self.portfolio.data["positions"]:
            for sym, pos in self.portfolio.data["positions"].items():
                cur     = price_dict.get(sym, pos["entry"])
                pnl     = (cur - pos["entry"]) * pos["shares"]
                pnl_pct = (cur / pos["entry"] - 1) * 100 if pos["entry"] else 0
                lines.append(
                    f"  {sym:<6} {pos['shares']:>4} Stk @ {pos['entry']:>8.2f}"
                    f" → {cur:>8.2f} | PNL {pnl:>+9.2f} ({pnl_pct:>+5.1f}%)"
                    f" | SL {pos['stop']:>8.2f} | Trail {pos['trailing']:>8.2f}"
                    f" | [{pos.get('strategy','?').upper()}]"
                    f" {'Partial' if pos['partial_sold'] else 'Full'}"
                )
        else:
            lines.append("  (keine)")

        if self.portfolio.data.get("short_positions"):
            lines += ["", "Short Positionen (Pair-Trading):"]
            for sym, pos in self.portfolio.data["short_positions"].items():
                cur = price_dict.get(sym, pos["entry"])
                pnl = (pos["entry"] - cur) * pos["shares"]
                lines.append(
                    f"  SHORT {sym:<6} {pos['shares']:>4} Stk @ {pos['entry']:>8.2f}"
                    f" → {cur:>8.2f} | PNL {pnl:>+9.2f}"
                )

        lines += ["", "Letzte 10 Trades:"]
        for t in self.portfolio.data["trades"][-10:]:
            lines.append(
                f"  {t['time'][:16]}  {t['symbol']:<6} {t['action']:<12}"
                f" {t['shares']:>4} Stk @ {t['price']:>8.2f}"
                f"  PNL {t['pnl']:>+9.2f}  ({t['reason']})"
            )

        lines += ["", "=" * 68]
        with open(report_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"  Report → {report_file}")


# ============================= CLI / OpenClaw-Einstieg ======================

def _build_alpaca_client(cfg: dict) -> Optional[AlpacaClient]:
    """
    Liest Alpaca-Keys aus Umgebungsvariablen.
    OpenClaw setzt APCA_API_KEY_ID und APCA_API_SECRET_KEY automatisch,
    wenn der Nutzer den Alpaca-Skill installiert hat.
    """
    if not ALPACA_AVAILABLE:
        print("[ERROR] alpaca-py fehlt – pip install alpaca-py", file=sys.stderr)
        return None

    key    = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")

    if not key or not secret:
        print("[ERROR] APCA_API_KEY_ID / APCA_API_SECRET_KEY nicht gesetzt.\n"
              "  In OpenClaw: clawhub install alpaca-trading → Keys hinterlegen\n"
              "  Lokal:       export APCA_API_KEY_ID=pk_...\n"
              "               export APCA_API_SECRET_KEY=sk_...",
              file=sys.stderr)
        return None

    # APCA_PAPER=false → Live; alles andere → Paper
    paper_env = os.getenv("APCA_PAPER", "true").lower()
    paper     = paper_env != "false"

    # cfg-Wert als Fallback, aber Env-Var hat Vorrang
    paper     = paper and cfg.get("alpaca_paper", True)

    feed = os.getenv("APCA_DATA_FEED", cfg.get("alpaca_data_feed", "iex"))

    return AlpacaClient(api_key=key, secret_key=secret, paper=paper, data_feed=feed)


# ============================= Einstiegspunkte =============================

def main():
    parser = argparse.ArgumentParser(
        description="Botti Trader v6.0 – Trend Following + Mean Reversion (Alpaca Edition)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["scan", "status", "eod", "backtest", "train"],
        default="scan",
        help=(
            "scan      – Täglichen Run starten  (Standard)\n"
            "status    – Portfolio-Status ausgeben (JSON)\n"
            "eod       – Alle Positionen sofort schließen\n"
            "backtest  – Historischen Backtest starten\n"
            "train     – ML-Modell trainieren"
        ),
    )
    parser.add_argument("--start", default="2024-01-01",
                        help="Backtest-Start (YYYY-MM-DD)")
    parser.add_argument("--end",   default=None,
                        help="Backtest-Ende  (YYYY-MM-DD, Standard: heute)")
    parser.add_argument("--period", default="800d",
                        help="Backtest-Période als '800d' (Standard: 800d)")
    parser.add_argument("--live", action="store_true",
                        help="Live-Modus – überschreibt APCA_PAPER=true")

    # Ältere Positions-Argumente (rückwärtskompatibel)
    # python trader_v6.py backtest [400d]
    # python trader_v6.py train
    parser.add_argument("legacy_cmd", nargs="?", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("legacy_period", nargs="?", default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Ältere Aufrufform: python trader_v6.py backtest [400d]
    if args.legacy_cmd in ("backtest", "train"):
        args.mode   = args.legacy_cmd
        if args.legacy_period:
            args.period = args.legacy_period

    cfg = dict(CONFIG)
    if args.live:
        cfg["alpaca_paper"] = False
        os.environ["APCA_PAPER"] = "false"

    alpaca = _build_alpaca_client(cfg)

    # ── Modus-Ausführung ─────────────────────────────────────────────────────

    if args.mode == "scan":
        trader = Trader(config=cfg, alpaca=alpaca)
        result = trader.run_daily()
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "status":
        trader = Trader(config=cfg, alpaca=alpaca)
        status = trader.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.mode == "eod":
        if alpaca:
            alpaca.close_all_positions()
            send_telegram(cfg.get("telegram_token", ""),
                          cfg.get("telegram_chat_id", ""),
                          "Botti Trader v6: manueller EOD-Close ausgeführt")
        else:
            print("[ERROR] Kein Alpaca-Client – EOD nicht möglich", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "backtest":
        cfg["initial_capital"] = 10_000.0
        tester = Backtester(config=cfg, alpaca=alpaca)
        tester.run(cfg["symbols"], period=args.period)

    elif args.mode == "train":
        train_ml_model(cfg["symbols"], cfg)


if __name__ == "__main__":
    # Ältere Aufrufform direkt abfangen bevor argparse aktiv ist
    if len(sys.argv) > 1 and sys.argv[1] in ("backtest", "train") and \
       not sys.argv[1].startswith("--"):
        # Umschreiben zu --mode backtest / --mode train
        mode_arg = sys.argv.pop(1)
        sys.argv.insert(1, mode_arg)
        sys.argv.insert(1, "--mode")

    main()
