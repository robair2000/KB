"""Entry filters for the Kronos bots: RSI + open interest. Used by bot.py (spot) and futures_bot.py.

Kronos only sees OHLCV. These rules can VETO a new entry; they never close a position, and an exit
on the forecast still happens. Every blocked entry is logged with its reason.

  RSI            no new long above RSI_LONG_MAX, no new short below RSI_SHORT_MIN
  OI positioning no long while price fell and OI rose >= OI_MIN_CHANGE over OI_LOOKBACK_BARS
                 candles (shorts building); no short while price rose and OI rose (longs building)
  OI spike       a one-candle OI change OI_SPIKE_Z std devs away from that coin's own last
                 OI_SPIKE_WINDOW candles (and >= OI_SPIKE_MIN_PCT). OI_SPIKE_MODE:
                 avoid  = no entries on that coin for OI_SPIKE_BARS candles
                 follow = only entries in the price direction of the spike candle
                 off    = log the spike only

Open interest is read from OI_EXCHANGE (binance or bybit, public data, no keys) because KuCoin
publishes no OI history. Trading still happens only on KuCoin. Only percentage changes are used,
so contract sizes (KuCoin PEPE vs Binance 1000PEPE) don't matter. OI readings are taken strictly
before each candle close, so nothing from the future leaks in.

All settings are environment variables; the defaults below are used when one isn't set.
"""
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


def _b(name, default):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


RSI_FILTER = _b("RSI_FILTER", "true")
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_LONG_MAX = float(os.getenv("RSI_LONG_MAX", "75"))
RSI_SHORT_MIN = float(os.getenv("RSI_SHORT_MIN", "25"))

OI_FILTER = _b("OI_FILTER", "true")
OI_EXCHANGE = os.getenv("OI_EXCHANGE", "binance").strip().lower()      # binance | bybit
OI_PERIOD = os.getenv("OI_PERIOD", "5m")                                # 5m | 15m | 30m | 1h
OI_LOOKBACK_BARS = int(os.getenv("OI_LOOKBACK_BARS", "4"))
OI_MIN_CHANGE = float(os.getenv("OI_MIN_CHANGE", "0.01"))
OI_SPIKE_MODE = os.getenv("OI_SPIKE_MODE", "avoid").strip().lower()    # avoid | follow | off
OI_SPIKE_Z = float(os.getenv("OI_SPIKE_Z", "3.0"))
OI_SPIKE_MIN_PCT = float(os.getenv("OI_SPIKE_MIN_PCT", "0.005"))
OI_SPIKE_WINDOW = int(os.getenv("OI_SPIKE_WINDOW", "100"))
OI_SPIKE_BARS = int(os.getenv("OI_SPIKE_BARS", "2"))
OI_REQUIRED = _b("OI_REQUIRED", "false")

PERIOD_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}
BYBIT_INTERVAL = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def summary():
    on = []
    if RSI_FILTER:
        on.append(f"RSI{RSI_PERIOD} (long<={RSI_LONG_MAX:g}, short>={RSI_SHORT_MIN:g})")
    if OI_FILTER:
        on.append(f"OI via {OI_EXCHANGE} (build {OI_MIN_CHANGE:.0%}/{OI_LOOKBACK_BARS}c, spikes {OI_SPIKE_MODE})")
    return ", ".join(on) or "none"


# ------------------------------------------------------------------ RSI
def rsi(close, n=14):
    """Wilder RSI (TradingView's ta.rsi)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn)
    return out.where(dn > 0, 100.0).where(d.notna().cumsum() >= n, np.nan)


# ------------------------------------------------------------------ open interest
def _base(sym):
    """'LINK-USDT' (spot) or 'LINKUSDTM' / 'XBTUSDTM' (futures) -> 'LINK' / 'BTC'."""
    b = sym.split("-")[0]
    if b.endswith("USDTM"):
        b = b[:-5]
    return "BTC" if b == "XBT" else b


class OIFeed:
    """Public OI history from Binance USDⓈ-M or Bybit linear. Never raises: returns None on trouble."""
    RETRY_S = 900

    def __init__(self, exchange=OI_EXCHANGE, period=OI_PERIOD, session=requests):
        if exchange not in ("binance", "bybit"):
            raise SystemExit(f"OI_EXCHANGE must be binance or bybit, not {exchange!r}")
        if period not in PERIOD_MS:
            raise SystemExit(f"OI_PERIOD must be one of {list(PERIOD_MS)}, not {period!r}")
        self.ex, self.period, self.step, self.http = exchange, period, PERIOD_MS[period], session
        self._symbols, self._failed_at, self._map = None, 0.0, {}

    def _get(self, url, params):
        r = self.http.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def symbols(self):
        if self._symbols is not None:
            return self._symbols
        if time.time() - self._failed_at < self.RETRY_S:
            return None
        try:
            if self.ex == "binance":
                info = self._get("https://fapi.binance.com/fapi/v1/exchangeInfo", {})
                self._symbols = {s["symbol"] for s in info["symbols"]
                                 if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                                 and s.get("status") == "TRADING"}
            else:
                syms, cursor = set(), ""
                for _ in range(10):
                    res = self._get("https://api.bybit.com/v5/market/instruments-info",
                                    {"category": "linear", "limit": 1000, "cursor": cursor})["result"]
                    syms |= {s["symbol"] for s in res["list"] if s.get("status") == "Trading"
                             and s.get("quoteCoin") == "USDT" and s.get("contractType") == "LinearPerpetual"}
                    cursor = res.get("nextPageCursor") or ""
                    if not cursor:
                        break
                self._symbols = syms
            log(f"OI source {self.ex} ready ({len(self._symbols)} perpetuals, {self.period} samples)")
        except Exception as e:
            self._failed_at = time.time()
            log(f"OI source {self.ex} unavailable, OI rules skipped for now: {e}")
        return self._symbols

    def resolve(self, sym):
        if sym in self._map:
            return self._map[sym]
        syms = self.symbols()
        if syms is None:
            return None
        b = _base(sym)
        cands = [b, "1000" + b, "1000000" + b]
        if b.startswith("1000000"):
            cands.append(b[7:])
        elif b.startswith("1000"):
            cands.append(b[4:])
        hit = next((c + "USDT" for c in cands if c + "USDT" in syms), None)
        if hit is None:
            log(f"{sym}: no OI market on {self.ex}, OI rules skipped for it")
        self._map[sym] = hit
        return hit

    def history(self, sym, since_ms, until_ms=None):
        """pd.Series of open interest indexed by ms timestamp (ascending), or None."""
        osym = self.resolve(sym)
        if osym is None:
            return None
        now = int(time.time() * 1000)
        until_ms = min(until_ms or now, now)
        page = 500 if self.ex == "binance" else 200
        if self.ex == "binance":                     # Binance keeps 30 days of OI history
            since_ms = max(since_ms, now - 29 * 86_400_000)
        pts, since = {}, since_ms
        try:
            for _ in range(400):
                if since >= until_ms:
                    break
                end = min(since + page * self.step, until_ms)
                if self.ex == "binance":
                    rows = self._get("https://fapi.binance.com/futures/data/openInterestHist",
                                     {"symbol": osym, "period": self.period, "limit": page,
                                      "startTime": since, "endTime": end})
                    got = [(int(r["timestamp"]), float(r["sumOpenInterest"])) for r in rows]
                else:
                    res = self._get("https://api.bybit.com/v5/market/open-interest",
                                    {"category": "linear", "symbol": osym, "intervalTime": BYBIT_INTERVAL[self.period],
                                     "limit": page, "startTime": since, "endTime": end})["result"]
                    got = [(int(r["timestamp"]), float(r["openInterest"])) for r in res["list"]]
                pts.update((t, v) for t, v in got if v > 0)
                since = end + 1
        except Exception as e:
            log(f"{sym}: OI fetch from {self.ex} failed: {e}")
            if not pts:
                return None
        return pd.Series(pts, dtype="float64").sort_index() if pts else None


def bars_needed():
    return max(OI_LOOKBACK_BARS, OI_SPIKE_WINDOW + OI_SPIKE_BARS) + 2


def _oi_at(oi, t_ms, max_age_ms):
    """OI strictly before each time in t_ms (no look-ahead); NaN where missing or stale."""
    t = np.asarray(t_ms, dtype="int64")
    k = oi.index.values.searchsorted(t, side="left") - 1
    ok = k >= 0
    kk = np.where(ok, k, 0)
    age = t - oi.index.values[kk]
    return np.where(ok & (age <= max_age_ms), oi.values[kk], np.nan)


def find_spike(oi_chg, px_chg):
    """Most recent spike among the last OI_SPIKE_BARS one-candle OI changes, or None."""
    nb = OI_SPIKE_BARS
    hist = oi_chg[:-nb][-OI_SPIKE_WINDOW:]
    hist = hist[~np.isnan(hist)]
    if len(hist) < 20:
        return None
    mu, sd = float(hist.mean()), float(hist.std())
    for ago in range(nb):
        x = oi_chg[-1 - ago]
        if np.isnan(x) or abs(x) < OI_SPIKE_MIN_PCT:
            continue
        z = (x - mu) / sd if sd > 0 else math.copysign(math.inf, x)
        if abs(z) >= OI_SPIKE_Z:
            return dict(ago=ago, chg=float(x), z=float(z), px=float(px_chg[-1 - ago]))
    return None


# ------------------------------------------------------------------ per-coin reading
def context(candles, oi, step_seconds):
    """Readings at the last closed candle. candles: KB frame (timestamps = candle open time)."""
    close = candles["close"].astype("float64").reset_index(drop=True)
    n = OI_LOOKBACK_BARS
    ctx = dict(rsi=float(rsi(close, RSI_PERIOD).iloc[-1]),
               px_chg=float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else None,
               oi_chg=None, oi_spike=None, oi_ok=False)
    if oi is not None and len(oi):
        tf = int(step_seconds) * 1000
        m = min(len(candles), bars_needed() + 1)
        ts = pd.to_datetime(candles["timestamps"]).values.astype("datetime64[ms]").astype("int64")[-m:]
        vals = _oi_at(oi, ts + tf, tf)                         # OI at each candle close
        chg = vals[1:] / vals[:-1] - 1
        pxc = close.values[-m:]
        pxc = pxc[1:] / pxc[:-1] - 1
        if len(vals) > n and not np.isnan(vals[-1]) and not np.isnan(vals[-1 - n]):
            ctx["oi_chg"] = float(vals[-1] / vals[-1 - n] - 1)
        ctx["oi_ok"] = bool(len(vals) and not np.isnan(vals[-1]))
        if len(chg) > OI_SPIKE_BARS:
            ctx["oi_spike"] = find_spike(chg, pxc)
    return ctx


_feed = None


def evaluate(sym, candles, step_seconds):
    """Fetch what's needed and return the readings for one coin (never raises)."""
    global _feed
    oi = None
    if OI_FILTER:
        if _feed is None:
            _feed = OIFeed()
        oi = _feed.history(sym, int(time.time() * 1000) - (bars_needed() + 3) * int(step_seconds) * 1000)
    try:
        ctx = context(candles, oi, step_seconds)
    except Exception as e:
        log(f"{sym}: filter readings failed ({e}), filters skipped")
        return None
    sp = ctx.get("oi_spike")
    if sp and sp["ago"] == 0:
        log(f"{sym}: ⚡ OI spike {spike_text(sp)}: {spike_kind(sp)}")
    return ctx


def entry_block(direction, ctx):
    """Reason string if a new entry (1 long, -1 short) should be skipped, else None."""
    if not ctx:
        return None
    r = ctx.get("rsi")
    if RSI_FILTER and r is not None and not math.isnan(r):
        if direction == 1 and r > RSI_LONG_MAX:
            return f"RSI {r:.0f} > {RSI_LONG_MAX:g}"
        if direction == -1 and r < RSI_SHORT_MIN:
            return f"RSI {r:.0f} < {RSI_SHORT_MIN:g}"
    if OI_FILTER:
        if not ctx.get("oi_ok"):
            return "no OI data" if OI_REQUIRED else None
        sp = ctx.get("oi_spike")
        if sp and OI_SPIKE_MODE in ("avoid", "follow"):
            kind = "up" if sp["chg"] > 0 else "down"
            if OI_SPIKE_MODE == "avoid":
                return f"OI spike {kind} {spike_text(sp)}"
            if direction * sp["px"] <= 0:
                return f"against OI spike {kind} {spike_text(sp)}"
        oc, pc = ctx.get("oi_chg"), ctx.get("px_chg")
        if oc is not None and pc is not None and oc >= OI_MIN_CHANGE:
            if direction == 1 and pc < 0:
                return f"shorts building (price {pc:+.1%}, OI {oc:+.1%} over {OI_LOOKBACK_BARS} candles)"
            if direction == -1 and pc > 0:
                return f"longs building (price {pc:+.1%}, OI {oc:+.1%} over {OI_LOOKBACK_BARS} candles)"
    return None


def spike_text(sp):
    when = "this candle" if sp["ago"] == 0 else f"{sp['ago']} candle(s) ago"
    return f"{sp['chg']:+.1%} (z {sp['z']:+.1f}, price {sp['px']:+.1%}) {when}"


def spike_kind(sp):
    if abs(sp["px"]) < 0.001:
        return "positions opening, price flat" if sp["chg"] > 0 else "positions closing, price flat"
    if sp["chg"] > 0:
        return "longs piling in" if sp["px"] > 0 else "shorts piling in"
    return "shorts squeezed / closing" if sp["px"] > 0 else "longs liquidated / closing"


def describe(ctx):
    if not ctx:
        return ""
    parts = []
    if ctx.get("rsi") is not None and not math.isnan(ctx["rsi"]):
        parts.append(f"RSI {ctx['rsi']:.0f}")
    if ctx.get("oi_chg") is not None:
        parts.append(f"OI {OI_LOOKBACK_BARS}c {ctx['oi_chg']:+.1%}")
    return " | ".join(parts)
