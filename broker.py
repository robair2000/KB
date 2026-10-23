"""Order execution on USDT-margined perpetuals (KuCoin Futures via ccxt), many symbols at once.

  limit()        post-only limit (maker): entries, take-profits, exits at the touch
  market()       market order (taker): market / chased entries, stops, chased exits
  stop_market()  reduce-only stop-MARKET resting on the exchange (crash safety net, always fills)
  stop_limit()   reduce-only stop-LIMIT resting on the exchange (STOP_ORDER_TYPE=limit)
Sizes are whole contracts. 1 contract = market contractSize of the coin
(e.g. 0.001 BTC, or 1,000,000 PEPE), so tiny-priced coins work the same as BTC.
"""
import logging
import math
import time
import uuid

import pandas as pd

import config as C
from fmt import fp

log = logging.getLogger("broker")


class OrderRejected(Exception):
    pass


def make_exchange(exchange_id, key="", secret="", password="", uta="auto"):
    import ccxt
    opts = {"defaultType": "swap"}
    if uta in ("true", "false"):
        opts["uta"] = uta == "true"
    ex = getattr(ccxt, exchange_id)({"apiKey": key, "secret": secret, "password": password or None,
                                     "enableRateLimit": True, "options": opts})
    ex.load_markets()
    return ex


class BaseBroker:
    def __init__(self, exchange):
        self.ex = exchange
        self._meta = {}

    # ------------------------------------------------------------- markets --
    def meta(self, sym):
        if sym not in self._meta:
            m = self.ex.market(sym)
            if not m.get("contract"):
                raise ValueError(f"{sym} is not a futures/swap market")
            p = m["precision"]["price"]
            tick_mode = getattr(self.ex, "precisionMode", 4) == 4            # ccxt TICK_SIZE
            self._meta[sym] = {
                "settle": m.get("settle") or m["quote"],
                "cs": float(m.get("contractSize") or 1),
                "tick": float(p) if tick_mode else 10 ** -int(p),
                "min": max(1, int((m.get("limits", {}).get("amount") or {}).get("min") or 1)),
                "base": m.get("base", sym.split("/")[0]),
            }
        return self._meta[sym]

    def settle(self):
        return "USDT"

    def tick(self, sym):
        return self.meta(sym)["tick"]

    def px(self, sym, p):
        return float(self.ex.price_to_precision(sym, p))

    def contracts_for(self, sym, notional, price):
        return int(math.floor(notional / (price * self.meta(sym)["cs"])))

    def notional(self, sym, contracts, price):
        return contracts * self.meta(sym)["cs"] * price

    def universe(self):
        """[(symbol, 24h turnover USDT)] for active USDT-settled perpetuals, most liquid first."""
        tickers = self.ex.fetch_tickers(None, {"type": "swap"})
        rows = []
        for sym, t in tickers.items():
            m = self.ex.markets.get(sym) or {}
            if not (m.get("swap") and m.get("active", True) and (m.get("settle") == "USDT")):
                continue
            rows.append((sym, float(t.get("quoteVolume") or 0)))
        return sorted(rows, key=lambda r: -r[1])

    # --------------------------------------------------------- market data --
    def candles(self, sym, timeframe, limit, page=200):
        """Closed candles only, oldest first. Paginates (KuCoin returns <= 200 per call)."""
        tf_ms = self.ex.parse_timeframe(timeframe) * 1000
        now = self.ex.milliseconds()
        since = now - (limit + 2) * tf_ms
        rows = []
        while since < now:
            chunk = self.ex.fetch_ohlcv(sym, timeframe, since=since, limit=page)
            if not chunk:
                since += page * tf_ms
                continue
            rows += chunk
            nxt = chunk[-1][0] + tf_ms
            since = nxt if nxt > since else since + page * tf_ms
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts")
        df = df[df["ts"] + tf_ms <= now]                                  # drop the forming candle
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        return df.tail(limit).reset_index(drop=True)

    def book(self, sym):
        ob = self.ex.fetch_order_book(sym, limit=20)        # KuCoin Futures only accepts 20 or 100
        return float(ob["bids"][0][0]), float(ob["asks"][0][0])


class LiveBroker(BaseBroker):
    def setup(self, sym):
        """Margin mode / leverage for a symbol; only call while flat on it."""
        if sym in getattr(self, "_setup_done", set()):
            return
        self._setup_done = getattr(self, "_setup_done", set()) | {sym}
        if C.SET_MARGIN_MODE_ON_START:
            try:
                self.ex.set_margin_mode(C.MARGIN_MODE, sym)
            except Exception as e:
                log.info("set_margin_mode %s: %s", sym, e)
        if C.MARGIN_MODE == "cross":
            try:
                self.ex.set_leverage(C.LEVERAGE, sym, {"marginMode": "cross"})
            except Exception as e:
                log.info("set_leverage %s: %s", sym, e)

    def _params(self, **extra):
        p = {"leverage": C.LEVERAGE, "marginMode": C.MARGIN_MODE, "clientOrderId": f"kr{uuid.uuid4().hex[:20]}"}
        p.update(extra)
        return p

    def limit(self, sym, side, contracts, price, reduce_only=False):
        params = self._params(postOnly=True)
        if reduce_only:
            params["reduceOnly"] = True
        try:
            o = self.ex.create_order(sym, "limit", side, int(contracts), self.px(sym, price), params)
        except Exception as e:
            msg = str(e).lower()
            if "immediately" in msg or "post" in msg or "taker" in msg:
                raise OrderRejected(str(e))
            raise
        log.info("%s placed %s%s %d @ %s (%s)", sym, side, " RO" if reduce_only else "", contracts,
                 fp(self.px(sym, price)), o["id"])
        return o["id"]

    def market(self, sym, side, contracts, reduce_only=False):
        params = self._params()
        if reduce_only:
            params["reduceOnly"] = True
        o = self.ex.create_order(sym, "market", side, int(contracts), None, params)
        log.info("%s MARKET %s%s %d (%s)", sym, side, " RO" if reduce_only else "", contracts, o["id"])
        return o["id"]

    def stop_market(self, sym, side, contracts, trigger):
        o = self.ex.create_order(sym, "market", side, int(contracts), None,
                                 self._params(stopLossPrice=self.px(sym, trigger), reduceOnly=True,
                                              triggerPriceType=C.TRIGGER_PRICE_TYPE))
        log.info("%s exchange stop-market %s %d trig %s (%s)", sym, side, contracts, fp(trigger), o["id"])
        return o["id"]

    def stop_limit(self, sym, side, contracts, trigger, price):
        o = self.ex.create_order(sym, "limit", side, int(contracts), self.px(sym, price),
                                 self._params(stopLossPrice=self.px(sym, trigger), reduceOnly=True,
                                              triggerPriceType=C.TRIGGER_PRICE_TYPE))
        log.info("%s exchange stop-limit %s %d trig %s lim %s (%s)", sym, side, contracts,
                 fp(trigger), fp(price), o["id"])
        return o["id"]

    def order(self, sym, oid):
        """status: open | closed | canceled | missing (KuCoin forgets cancelled orders with no fill)."""
        try:
            o = self.ex.fetch_order(oid, sym)
        except Exception as e:
            if type(e).__name__ == "OrderNotFound" or "orderNotExist" in str(e):
                return {"status": "missing", "filled": 0.0}
            raise
        return {"status": o["status"], "filled": float(o.get("filled") or 0)}

    def cancel(self, sym, oid):
        try:
            self.ex.cancel_order(oid, sym)
        except Exception as e:
            # 100004 / orderNotExist = already filled or cancelled; that's normal, not an error
            quiet = "cannot be canceled" in str(e) or "orderNotExist" in str(e) or type(e).__name__ == "OrderNotFound"
            log.log(logging.DEBUG if quiet else logging.WARNING, "cancel %s %s: %s", sym, oid, e)

    def cancel_stops(self, sym):
        try:
            self.ex.cancel_all_orders(sym, {"trigger": True})
        except Exception as e:
            log.info("cancel stops %s: %s", sym, e)

    def cancel_own_open_orders(self, syms=None):
        for o in self.ex.fetch_open_orders(None) if syms is None else sum(
                (self.ex.fetch_open_orders(s) for s in syms), []):
            if (o.get("clientOrderId") or "").startswith("kr"):
                self.cancel(o["symbol"], o["id"])

    def positions(self):
        """{symbol: (signed contracts, entry price)} for every open position on the account."""
        out = {}
        for p in self.ex.fetch_positions():
            n = float(p.get("contracts") or 0)
            if n > 0:
                out[p["symbol"]] = ((1 if p.get("side") == "long" else -1) * int(round(n)),
                                    float(p.get("entryPrice") or 0))
        return out

    def realized_pnl(self, sym, since_ms):
        """Net realised P&L of positions on sym closed after since_ms, or None if unavailable."""
        for attempt in range(3):
            try:
                hist = self.ex.fetch_positions_history([sym], since_ms)
                rows = [h for h in hist if h.get("symbol") == sym]
                if rows:
                    return sum(float(h.get("realizedPnl") or 0) for h in rows)
            except Exception as e:
                log.info("positions history %s: %s", sym, e)
            time.sleep(1.5)
        return None

    def equity(self):
        b = self.ex.fetch_balance({"code": "USDT"})
        return float((b.get("USDT") or {}).get("total") or 0)

    def free(self):
        b = self.ex.fetch_balance({"code": "USDT"})
        return float((b.get("USDT") or {}).get("free") or 0)


class PaperBroker(BaseBroker):
    """Futures simulator on live public prices. Resting limits fill when the opposite touch
    trades through them; market orders fill at once at the touch (taker fee); stops trigger on mid
    price, then a stop-market fills at the touch and a stop-limit rests as a normal limit."""

    def __init__(self, exchange, start_equity, maker_fee):
        super().__init__(exchange)
        self.cash = start_equity
        self.pos, self.entry, self.realized = {}, {}, {}
        self.fee = maker_fee
        self.orders = {}

    def setup(self, sym):
        pass

    def _new(self, sym, **o):
        oid = f"paper-{uuid.uuid4().hex[:8]}"
        self.orders[oid] = dict(sym=sym, status="open", filled=0, **o)
        return oid

    def limit(self, sym, side, contracts, price, reduce_only=False):
        price = self.px(sym, price)
        bid, ask = self.book(sym)
        if (side == "buy" and price >= ask) or (side == "sell" and price <= bid):
            raise OrderRejected("post-only would cross")
        oid = self._new(sym, side=side, amount=int(contracts), price=price, ro=reduce_only, trig=None)
        log.info("[paper] %s %s%s %d @ %s", sym, side, " RO" if reduce_only else "", contracts, fp(price))
        return oid

    def market(self, sym, side, contracts, reduce_only=False):
        bid, ask = self.book(sym)
        oid = self._new(sym, side=side, amount=int(contracts), price=ask if side == "buy" else bid,
                        ro=reduce_only, trig=None)
        o = self.orders[oid]
        qty = o["amount"]
        if reduce_only:
            pos = self.pos.get(sym, 0)
            qty = min(qty, abs(pos) if pos and (pos > 0) != (side == "buy") else 0)
        if qty > 0:
            self._fill(o, qty, C.TAKER_FEE)
        o["status"] = "closed" if qty > 0 else "canceled"
        log.info("[paper] %s MARKET %s%s %d @ %s", sym, side, " RO" if reduce_only else "", qty, fp(o["price"]))
        return oid

    def stop_market(self, sym, side, contracts, trigger):
        return self._new(sym, side=side, amount=int(contracts), price=None, ro=True, trig=self.px(sym, trigger))

    def stop_limit(self, sym, side, contracts, trigger, price):
        return self._new(sym, side=side, amount=int(contracts), price=self.px(sym, price), ro=True,
                         trig=self.px(sym, trigger))

    def _fill(self, o, qty, fee):
        sym, buy, px = o["sym"], o["side"] == "buy", o["price"]
        cs, pos, ent = self.meta(sym)["cs"], self.pos.get(sym, 0), self.entry.get(sym, 0.0)
        sg = 1 if buy else -1
        f = qty * cs * px * fee
        pnl = -f
        if pos == 0 or (pos > 0) == buy:
            new = pos + sg * qty
            self.entry[sym] = (abs(pos) * ent + qty * px) / abs(new)
        else:
            pnl += qty * cs * (px - ent) * (1 if pos > 0 else -1)
            new = pos + sg * qty
            if new == 0:
                self.entry[sym] = 0.0
        self.pos[sym] = new
        self.cash += pnl
        self.realized[sym] = self.realized.get(sym, 0.0) + pnl
        o["filled"] += qty
        if o["filled"] >= o["amount"]:
            o["status"] = "closed"
        log.info("[paper] %s FILLED %s %d @ %s | pos %d", sym, o["side"], qty, fp(px), new)

    def _match(self, sym):
        live = [o for o in self.orders.values() if o["status"] == "open" and o["sym"] == sym]
        if not live:
            return
        bid, ask = self.book(sym)
        mid = (bid + ask) / 2
        for o in live:
            buy, fee = o["side"] == "buy", self.fee
            if o["trig"] is not None:
                if not o.get("triggered"):
                    if (buy and mid >= o["trig"]) or (not buy and mid <= o["trig"]):
                        o["triggered"] = True
                    else:
                        continue
                fee = C.TAKER_FEE
                if o["price"] is None:                      # stop-market: fill at the touch
                    o["price"] = ask if buy else bid
            if not ((buy and ask <= o["price"]) or (not buy and bid >= o["price"])):
                continue
            qty = o["amount"] - o["filled"]
            if o["ro"]:
                pos = self.pos.get(sym, 0)
                qty = min(qty, abs(pos) if pos and (pos > 0) != buy else 0)
                if qty <= 0:
                    o["status"] = "canceled"
                    continue
            self._fill(o, qty, fee)

    def order(self, sym, oid):
        self._match(sym)
        o = self.orders[oid]
        return {"status": o["status"], "filled": o["filled"]}

    def cancel(self, sym, oid):
        self._match(sym)
        if self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "canceled"

    def cancel_stops(self, sym):
        for o in self.orders.values():
            if o["status"] == "open" and o["sym"] == sym and o["trig"] is not None:
                o["status"] = "canceled"

    def cancel_own_open_orders(self, syms=None):
        for o in self.orders.values():
            if o["status"] == "open":
                o["status"] = "canceled"

    def positions(self):
        for sym in {o["sym"] for o in self.orders.values() if o["status"] == "open"}:
            self._match(sym)
        return {s: (n, self.entry[s]) for s, n in self.pos.items() if n}

    def realized_pnl(self, sym, since_ms):
        return None            # bot uses its own realised ledger snapshot in paper mode

    def realized_total(self, sym):
        return self.realized.get(sym, 0.0)

    def equity(self):
        eq = self.cash
        for s, n in self.pos.items():
            if n:
                bid, ask = self.book(s)
                eq += n * self.meta(s)["cs"] * ((bid + ask) / 2 - self.entry[s])
        return eq

    def free(self):
        used = sum(abs(n) * self.meta(s)["cs"] * self.entry[s] for s, n in self.pos.items() if n)
        used += sum((o["amount"] - o["filled"]) * self.meta(o["sym"])["cs"] * o["price"]
                    for o in self.orders.values() if o["status"] == "open" and not o["ro"])
        return self.equity() - used / C.LEVERAGE
