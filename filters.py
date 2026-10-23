"""Entry filters: RSI / EMA / Bollinger on the bot's own candles, plus open interest.

Kronos only sees OHLCV. These rules sit on top of its signal and can veto a NEW entry;
they never force an exit. Shared by bot.py and backtest.py so both trade the same way.

OI spikes: each candle's OI change is compared with that coin's own recent normal (a z-score
over OI_SPIKE_WINDOW candles), so a 2% hourly drop on BTC counts as a spike even though it would
be routine on a small altcoin.

Open interest comes from OI_EXCHANGE (Binance USDⓈ-M by default) because KuCoin Futures
publishes no OI history. Only percentage changes are used, so contract size differences
(e.g. KuCoin PEPE vs Binance 1000PEPE) don't matter.
"""
import logging
import math
import time

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger("filters")


# --------------------------------------------------------------- indicators --
def ema(close, n):
    return close.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    """Wilder RSI (TradingView's ta.rsi)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn)
    return out.where(dn > 0, 100.0).where(d.notna().cumsum() >= n, np.nan)


def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)       # population stdev, as TradingView
    return mid + k * sd, mid - k * sd


# ------------------------------------------------------------ open interest --
class OIFeed:
    """Public OI history from another exchange via ccxt. Never raises: on trouble it returns None."""
    RETRY_S = 900

    def __init__(self, exchange_id=None, period=None):
        self.id = exchange_id or C.OI_EXCHANGE
        self.period = period or C.OI_PERIOD
        self.ex, self._failed_at, self._map = None, 0.0, {}

    def _exchange(self):
        if self.ex is not None:
            return self.ex
        if time.time() - self._failed_at < self.RETRY_S:
            return None
        try:
            import ccxt
            ex = getattr(ccxt, self.id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
            ex.load_markets()
            self.ex = ex
            log.info("OI source %s ready (%s samples)", self.id, self.period)
        except Exception as e:
            self._failed_at = time.time()
            log.warning("OI source %s unavailable, OI rules skipped for now: %s", self.id, e)
        return self.ex

    def resolve(self, sym):
        """KuCoin symbol -> symbol on the OI exchange (tries 1000x / 1000000x listings), or None."""
        if sym in self._map:
            return self._map[sym]
        ex = self._exchange()
        if ex is None:
            return None
        base, rest = sym.split("/", 1)
        bases = [base, "1000" + base, "1000000" + base]
        if base.startswith("1000000"):
            bases.append(base[7:])
        elif base.startswith("1000"):
            bases.append(base[4:])
        hit = None
        for b in bases:
            m = ex.markets.get(f"{b}/{rest}")
            if m and m.get("swap") and m.get("linear") and m.get("active", True):
                hit = m["symbol"]
                break
        if hit is None:
            log.info("no OI market for %s on %s", sym, self.id)
        self._map[sym] = hit
        return hit

    def history(self, sym, since_ms, until_ms=None):
        """pd.Series of open interest (in coins) indexed by ms timestamp, ascending; None if unavailable."""
        osym = self.resolve(sym)
        if osym is None:
            return None
        ex = self.ex
        step = ex.parse_timeframe(self.period) * 1000
        until_ms = until_ms or ex.milliseconds()
        if self.id.startswith("binance"):                       # Binance keeps 30 days of OI history
            since_ms = max(since_ms, ex.milliseconds() - 29 * 86_400_000)
        page = 500 if self.id.startswith("binance") else 200
        rows, since = [], since_ms
        try:
            for _ in range(400):
                if since >= until_ms:
                    break
                chunk = ex.fetch_open_interest_history(osym, self.period, since=since, limit=page)
                if not chunk:
                    since += page * step
                    continue
                rows += chunk
                nxt = max(r["timestamp"] for r in chunk) + step
                since = nxt if nxt > since else since + page * step
        except Exception as e:
            log.warning("OI %s (%s): %s", sym, osym, e)
            if not rows:
                return None
        pts = {}
        for r in rows:
            v = r.get("openInterestAmount")
            if v is None:
                v = r.get("openInterestValue")
            if r.get("timestamp") is not None and v:
                pts[int(r["timestamp"])] = float(v)
        if not pts:
            return None
        return pd.Series(pts).sort_index()

    def recent(self, sym, bars=None):
        """Enough history for the OI rules (incl. the spike z-score window) on the latest candle."""
        bars = bars or oi_bars_needed()
        tf = C.TF_MINUTES * 60_000
        now = int(time.time() * 1000)
        return self.history(sym, now - (bars + 3) * tf)


def oi_bars_needed():
    return max(C.OI_LOOKBACK_BARS, C.OI_SPIKE_WINDOW + C.OI_SPIKE_BARS) + 2


def _oi_at(oi, t_ms, max_age_ms):
    """Latest OI strictly before each time in t_ms (no look-ahead); NaN where missing or stale."""
    t = np.asarray(t_ms, dtype="int64")
    k = oi.index.values.searchsorted(t, side="left") - 1
    ok = k >= 0
    kk = np.where(ok, k, 0)
    age = t - oi.index.values[kk]
    return np.where(ok & (age <= max_age_ms), oi.values[kk], np.nan)


def find_spike(oi_chg, px_chg):
    """Most recent spike among the last OI_SPIKE_BARS one-candle OI changes, or None.
    oi_chg / px_chg: arrays of one-candle % changes, oldest first, aligned, last = latest candle."""
    nb = C.OI_SPIKE_BARS
    hist = oi_chg[:-nb][-C.OI_SPIKE_WINDOW:]
    hist = hist[~np.isnan(hist)]
    if len(hist) < 20:
        return None
    mu, sd = float(hist.mean()), float(hist.std())
    for ago in range(nb):
        x = oi_chg[-1 - ago]
        if np.isnan(x) or abs(x) < C.OI_SPIKE_MIN_PCT:
            continue
        z = (x - mu) / sd if sd > 0 else math.copysign(math.inf, x)
        if abs(z) >= C.OI_SPIKE_Z:
            return dict(ago=ago, chg=float(x), z=float(z), px=float(px_chg[-1 - ago]))
    return None


# ------------------------------------------------------------------ context --
def context(candles, oi=None):
    """Indicator readings at the last closed candle. candles: the frame fed to Kronos."""
    close = candles["close"].astype("float64").reset_index(drop=True)
    last = float(close.iloc[-1])
    up, lo = bollinger(close, C.BB_PERIOD, C.BB_STD)
    n = C.OI_LOOKBACK_BARS
    ctx = dict(close=last,
               rsi=float(rsi(close, C.RSI_PERIOD).iloc[-1]),
               ema=float(ema(close, C.EMA_PERIOD).iloc[-1]),
               bb_upper=float(up.iloc[-1]), bb_lower=float(lo.iloc[-1]),
               px_chg=last / float(close.iloc[-1 - n]) - 1 if len(close) > n else None,
               oi_chg=None, oi_last_chg=None, oi_spike=None)
    if oi is not None and len(oi):
        tf = C.TF_MINUTES * 60_000
        m = min(len(candles), oi_bars_needed() + 1)
        ts = candles["ts"].astype("int64").values[-m:]
        vals = _oi_at(oi, ts + tf, tf)                             # OI at each candle close
        chg = vals[1:] / vals[:-1] - 1
        pxc = close.values[-m:]
        pxc = pxc[1:] / pxc[:-1] - 1
        if len(chg) and not np.isnan(chg[-1]):
            ctx["oi_last_chg"] = float(chg[-1])
        if len(vals) > n and not np.isnan(vals[-1]) and not np.isnan(vals[-1 - n]):
            ctx["oi_chg"] = float(vals[-1] / vals[-1 - n] - 1)
        ctx["oi_spike"] = find_spike(chg, pxc) if len(chg) > C.OI_SPIKE_BARS else None
    return ctx


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def entry_block(side, ctx):
    """Reason string if a new entry on `side` (1 long, -1 short) should be skipped, else None."""
    if ctx is None:
        return None
    long = side == 1
    c = ctx["close"]
    if C.RSI_FILTER and _ok(ctx["rsi"]):
        if long and ctx["rsi"] > C.RSI_LONG_MAX:
            return f"RSI {ctx['rsi']:.0f} > {C.RSI_LONG_MAX:.0f}"
        if not long and ctx["rsi"] < C.RSI_SHORT_MIN:
            return f"RSI {ctx['rsi']:.0f} < {C.RSI_SHORT_MIN:.0f}"
    if C.EMA_FILTER and _ok(ctx["ema"]):
        if long and c <= ctx["ema"]:
            return f"below EMA{C.EMA_PERIOD}"
        if not long and c >= ctx["ema"]:
            return f"above EMA{C.EMA_PERIOD}"
    if C.BB_FILTER and _ok(ctx["bb_upper"]):
        if long and c > ctx["bb_upper"]:
            return "above upper BB"
        if not long and c < ctx["bb_lower"]:
            return "below lower BB"
    if C.OI_FILTER:
        oc, ol, pc = ctx.get("oi_chg"), ctx.get("oi_last_chg"), ctx.get("px_chg")
        if oc is None and ol is None:
            return "no OI data" if C.OI_REQUIRED else None
        sp, mode = ctx.get("oi_spike"), C.OI_SPIKE_MODE.lower()
        if sp and mode in ("avoid", "follow"):
            kind = "up" if sp["chg"] > 0 else "down"
            if mode == "avoid":
                return f"OI spike {kind} {spike_text(sp)}"
            if side * sp["px"] <= 0:
                return f"against OI spike {kind} {spike_text(sp)}"
        if _ok(oc) and _ok(pc) and oc >= C.OI_MIN_CHANGE:
            if long and pc < 0:
                return f"shorts building (price {pc:+.1%}, OI {oc:+.1%})"
            if not long and pc > 0:
                return f"longs building (price {pc:+.1%}, OI {oc:+.1%})"
    return None


def spike_text(sp):
    when = "this candle" if sp["ago"] == 0 else f"{sp['ago']} candle(s) ago"
    return f"{sp['chg']:+.1%} (z {sp['z']:+.1f}, price {sp['px']:+.1%}) {when}"


def spike_kind(sp):
    """Plain-language reading of a spike from OI and price direction."""
    if abs(sp["px"]) < 0.001:
        return "positions opening, price flat" if sp["chg"] > 0 else "positions closing, price flat"
    if sp["chg"] > 0:
        return "longs piling in" if sp["px"] > 0 else "shorts piling in"
    return "shorts squeezed / closing" if sp["px"] > 0 else "longs liquidated / closing"


def describe(ctx):
    if not ctx:
        return ""
    parts = []
    if _ok(ctx.get("rsi")):
        parts.append(f"RSI {ctx['rsi']:.0f}")
    if _ok(ctx.get("ema")):
        parts.append(f"EMA{C.EMA_PERIOD} {ctx['close'] / ctx['ema'] - 1:+.1%}")
    if _ok(ctx.get("oi_chg")):
        parts.append(f"OI {C.OI_LOOKBACK_BARS}c {ctx['oi_chg']:+.1%}")
    if ctx.get("oi_spike"):
        parts.append(f"OI spike {spike_text(ctx['oi_spike'])}")
    return " | ".join(parts)
