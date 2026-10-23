"""KuCoin FUTURES (classic account): public candles + signed trading client.

Endpoints below follow KuCoin's classic futures REST docs. Run `python futures_check.py` (read-only)
before trusting them: it verifies candles, contract info and your account overview.
"""
import base64
import hashlib
import hmac
import json
import math
import os
import time

import pandas as pd
import requests

FUT_BASE = "https://api-futures.kucoin.com"
GRANULARITY_MIN = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1hour": 60, "4hour": 240, "1day": 1440}
STEP_SECONDS = {k: v * 60 for k, v in GRANULARITY_MIN.items()}


def to_contract(pair):
    """'BTC-USDT' -> 'XBTUSDTM' (KuCoin calls BTC 'XBT' on futures)."""
    base = pair.split("-")[0]
    return ("XBT" if base == "BTC" else base) + "USDTM"


def fetch_futures_candles(contract, interval="1hour", n=450, end_ts=None, session=requests):
    """Last `n` CLOSED candles, oldest -> newest. Same columns as spot fetch_candles.
    `amount` is approximated as volume * close (futures klines carry no turnover)."""
    step = STEP_SECONDS[interval]
    end_ts = int(end_ts or time.time())
    rows, cursor = [], end_ts * 1000
    for _ in range(12):
        if len(rows) >= n + 2:
            break
        r = session.get(FUT_BASE + "/api/v1/kline/query", timeout=20,
                        params={"symbol": contract, "granularity": GRANULARITY_MIN[interval],
                                "from": cursor - 500 * step * 1000, "to": cursor})
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            break
        rows.extend(data)
        cursor = min(int(x[0]) for x in data) - 1
    if not rows:
        raise RuntimeError(f"No futures candles for {contract}")
    df = pd.DataFrame([x[:6] for x in rows], columns=["t", "open", "high", "low", "close", "volume"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    df["t"] = (df["t"] // 1000).astype("int64")
    df = df.drop_duplicates("t").sort_values("t")
    df = df[df["t"] + step <= end_ts]
    df["timestamps"] = pd.to_datetime(df["t"], unit="s")
    df["amount"] = df["volume"] * df["close"]
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].tail(n).reset_index(drop=True)


def contract_info(contract, session=requests):
    d = session.get(f"{FUT_BASE}/api/v1/contracts/{contract}", timeout=20).json()["data"]
    return {"multiplier": float(d["multiplier"]), "lot": float(d["lotSize"]),
            "tick": float(d["tickSize"]), "max_leverage": float(d.get("maxLeverage", 0))}


class FuturesLive:
    def __init__(self):
        self.key = os.environ["KUCOIN_API_KEY"]
        self.secret = os.environ["KUCOIN_API_SECRET"]
        self.passphrase = os.environ["KUCOIN_PASSPHRASE"]
        self._info = {}

    # ---- signing / transport
    def _headers(self, method, endpoint, body=""):
        ts = str(int(time.time() * 1000))
        sig = base64.b64encode(hmac.new(self.secret.encode(), (ts + method + endpoint + body).encode(),
                                        hashlib.sha256).digest()).decode()
        pp = base64.b64encode(hmac.new(self.secret.encode(), self.passphrase.encode(),
                                       hashlib.sha256).digest()).decode()
        return {"KC-API-KEY": self.key, "KC-API-SIGN": sig, "KC-API-TIMESTAMP": ts,
                "KC-API-PASSPHRASE": pp, "KC-API-KEY-VERSION": "2", "Content-Type": "application/json"}

    def _call(self, method, endpoint, payload=None):
        body = json.dumps(payload) if payload else ""
        r = requests.request(method, FUT_BASE + endpoint, headers=self._headers(method, endpoint, body),
                             data=body or None, timeout=20)
        out = r.json()
        if out.get("code") != "200000":
            raise RuntimeError(f"KuCoin futures error on {endpoint}: {out}")
        return out["data"]

    # ---- account
    def overview(self):
        d = self._call("GET", "/api/v1/account-overview?currency=USDT")
        return float(d["accountEquity"]), float(d["availableBalance"])

    def positions(self):
        d = self._call("GET", "/api/v1/positions")
        d = d if isinstance(d, list) else d.get("items", [])
        return {p["symbol"]: p for p in d if float(p.get("currentQty", 0)) != 0}

    def info(self, contract):
        if contract not in self._info:
            self._info[contract] = contract_info(contract)
        return self._info[contract]

    # ---- sizing helpers
    def lots_for(self, contract, notional, price):
        i = self.info(contract)
        lots = notional / (price * i["multiplier"])
        return math.floor(lots / i["lot"]) * i["lot"]

    def round_price(self, contract, price):
        t = self.info(contract)["tick"]
        return round(round(price / t) * t, 10)

    # ---- trading
    def prepare(self, contract, leverage):
        try:
            self._call("POST", "/api/v2/position/changeMarginMode", {"symbol": contract, "marginMode": "CROSS"})
        except RuntimeError as e:
            print(f"[warn] margin mode switch: {e}", flush=True)   # e.g. already CROSS
        self._call("POST", "/api/v2/changeCrossUserLeverage", {"symbol": contract, "leverage": str(leverage)})

    def market(self, contract, side, lots, leverage, reduce_only=False):
        return self._call("POST", "/api/v1/orders", {
            "clientOid": str(int(time.time() * 1e6)), "symbol": contract, "side": side, "type": "market",
            "size": int(lots), "leverage": int(leverage), "marginMode": "CROSS",
            "positionSide": "BOTH", "reduceOnly": reduce_only})

    def exchange_stop(self, contract, direction, lots, leverage, stop_price):
        """Protective reduce-only market stop that lives on the exchange."""
        return self._call("POST", "/api/v1/orders", {
            "clientOid": str(int(time.time() * 1e6)), "symbol": contract,
            "side": "sell" if direction == 1 else "buy", "type": "market",
            "stop": "down" if direction == 1 else "up", "stopPriceType": "TP",
            "stopPrice": str(self.round_price(contract, stop_price)),
            "size": int(lots), "leverage": int(leverage), "marginMode": "CROSS",
            "positionSide": "BOTH", "reduceOnly": True})

    def exchange_take(self, contract, direction, lots, leverage, take_price):
        """Protective reduce-only market take-profit that lives on the exchange."""
        return self._call("POST", "/api/v1/orders", {
            "clientOid": str(int(time.time() * 1e6)), "symbol": contract,
            "side": "sell" if direction == 1 else "buy", "type": "market",
            "stop": "up" if direction == 1 else "down", "stopPriceType": "TP",
            "stopPrice": str(self.round_price(contract, take_price)),
            "size": int(lots), "leverage": int(leverage), "marginMode": "CROSS",
            "positionSide": "BOTH", "reduceOnly": True})

    def cancel(self, order_id):
        return self._call("DELETE", f"/api/v1/orders/{order_id}")

    def cancel_stop_orders(self, contract):
        """Cancel every untriggered stop / take-profit order on a contract (leftovers after a close)."""
        return self._call("DELETE", f"/api/v1/stopOrders?symbol={contract}")
