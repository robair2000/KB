"""Minimal KuCoin spot client: public candles + signed live orders."""
import base64
import hashlib
import hmac
import json
import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.kucoin.com"
INTERVAL_SECONDS = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800,
                    "1hour": 3600, "4hour": 14400, "1day": 86400}


def fetch_candles(symbol, interval="1hour", n=400, end_ts=None, session=requests):
    """Return the last `n` CLOSED candles as a DataFrame sorted oldest -> newest.

    Columns: timestamps, open, high, low, close, volume, amount.
    KuCoin returns max 1500 candles per call, newest first, as
    [time, open, close, high, low, volume, turnover].
    """
    step = INTERVAL_SECONDS[interval]
    end_ts = int(end_ts or time.time())
    rows, cursor = [], end_ts
    while len(rows) < n + 1:
        start = cursor - 1500 * step
        r = session.get(BASE_URL + "/api/v1/market/candles",
                        params={"symbol": symbol, "type": interval,
                                "startAt": start, "endAt": cursor}, timeout=20)
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            break
        rows.extend(data)
        cursor = int(data[-1][0]) - 1
    if not rows:
        raise RuntimeError(f"No candles returned for {symbol}")
    df = pd.DataFrame(rows, columns=["t", "open", "close", "high", "low", "volume", "amount"])
    df = df.drop_duplicates("t")
    for c in ["t", "open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c])
    df = df.sort_values("t").reset_index(drop=True)
    # drop the still-forming candle: only trust candles that have closed
    df = df[df["t"] + step <= end_ts]
    df["timestamps"] = pd.to_datetime(df["t"], unit="s")
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    return df.tail(n).reset_index(drop=True)


class KuCoinLive:
    """Signed spot trading. Only used when MODE=live. Not exercised by the paper bot."""

    def __init__(self):
        self.key = os.environ["KUCOIN_API_KEY"]
        self.secret = os.environ["KUCOIN_API_SECRET"]
        self.passphrase = os.environ["KUCOIN_PASSPHRASE"]
        self._rules = {}

    def _headers(self, method, endpoint, body=""):
        ts = str(int(time.time() * 1000))
        sig = base64.b64encode(hmac.new(self.secret.encode(), (ts + method + endpoint + body).encode(),
                                        hashlib.sha256).digest()).decode()
        pp = base64.b64encode(hmac.new(self.secret.encode(), self.passphrase.encode(),
                                       hashlib.sha256).digest()).decode()
        return {"KC-API-KEY": self.key, "KC-API-SIGN": sig, "KC-API-TIMESTAMP": ts,
                "KC-API-PASSPHRASE": pp, "KC-API-KEY-VERSION": "2",
                "Content-Type": "application/json"}

    def _call(self, method, endpoint, payload=None):
        body = json.dumps(payload) if payload else ""
        r = requests.request(method, BASE_URL + endpoint, headers=self._headers(method, endpoint, body),
                             data=body or None, timeout=20)
        r.raise_for_status()
        out = r.json()
        if out.get("code") != "200000":
            raise RuntimeError(f"KuCoin error: {out}")
        return out["data"]

    def usdt_balance(self):
        data = self._call("GET", "/api/v1/accounts?type=trade&currency=USDT")
        return sum(float(a["available"]) for a in data if a["currency"] == "USDT")

    def base_balance(self, currency):
        data = self._call("GET", f"/api/v1/accounts?type=trade&currency={currency}")
        return sum(float(a["available"]) for a in data if a["currency"] == currency)

    def rules(self, symbol):
        if symbol not in self._rules:
            data = requests.get(BASE_URL + "/api/v2/symbols/" + symbol, timeout=20).json()["data"]
            self._rules[symbol] = {"inc": float(data["baseIncrement"]), "min": float(data["baseMinSize"])}
        return self._rules[symbol]

    def round_size(self, symbol, qty):
        inc = self.rules(symbol)["inc"]
        return round((qty // inc) * inc, 10)

    def market_buy(self, symbol, usdt):
        return self._call("POST", "/api/v1/orders", {
            "clientOid": str(int(time.time() * 1e6)), "side": "buy", "symbol": symbol,
            "type": "market", "funds": f"{usdt:.2f}"})

    def market_sell(self, symbol, qty):
        qty = self.round_size(symbol, qty)
        if qty < self.rules(symbol)["min"]:
            raise RuntimeError(f"{symbol}: size {qty} below exchange minimum")
        return self._call("POST", "/api/v1/orders", {
            "clientOid": str(int(time.time() * 1e6)), "side": "sell", "symbol": symbol,
            "type": "market", "size": str(qty)})
