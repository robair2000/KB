"""Kronos FUTURES bot (KuCoin classic futures, CROSS margin, long + short, bot-chosen leverage).

Started through bot.py when MARKET=futures.  MODE=paper (default) simulates everything.
Kill switch: create a file named STOP.
"""
import json
import os
import time
from datetime import datetime, timezone

import filters
from fkucoin import STEP_SECONDS, FuturesLive, fetch_futures_candles, to_contract
from futures_rules import (FUNDING_8H, FUT_FEE, MAX_LEVERAGE, MAX_MARGIN_FRAC, MAX_TOTAL_LEVERAGE,
                           MMR, choose_leverage_and_notional, target_action)
from strategy import expected_return, load_predictor

PAIRS = [p.strip() for p in os.getenv("PAIRS", "BTC-USDT,ETH-USDT,SOL-USDT,LINK-USDT,DOGE-USDT").split(",")]
MODE = os.getenv("MODE", "paper").lower()
INTERVAL = os.getenv("INTERVAL", "1hour")
PRED_LEN = int(os.getenv("PRED_LEN", "6"))
SAMPLES = int(os.getenv("SAMPLES", "3"))
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.008"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0005"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.03"))
TAKE_PCT = float(os.getenv("TAKE_PCT", "0.06"))
RISK_FRAC = float(os.getenv("RISK_FRAC", "0.01"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))
START_CASH = float(os.getenv("START_CASH", "1000"))
MIN_NOTIONAL = 10.0
STATE_PATH = os.getenv("STATE_PATH_FUTURES", "state_futures.json")


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


# ------------------------------------------------------------------ state helpers
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"cash": START_CASH, "positions": {}, "trades": [], "day": None, "day_start_equity": None}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE_PATH)


def unrealized(pos, price):
    return pos["dir"] * pos["qty"] * (price - pos["entry"])


def paper_equity(state, prices):
    return state["cash"] + sum(unrealized(p, prices.get(s, p["entry"])) for s, p in state["positions"].items())


def exposure(state, prices):
    return sum(p["qty"] * prices.get(s, p["entry"]) for s, p in state["positions"].items())


def used_margin(state, prices):
    return sum(p["qty"] * prices.get(s, p["entry"]) / p["lev"] for s, p in state["positions"].items())


def record_trade(state, sym, pos, exit_price, reason):
    state["trades"].append({"symbol": sym, "dir": pos["dir"], "lev": pos["lev"], "entry": pos["entry"],
                            "exit": exit_price, "pnl_pct": pos["dir"] * (exit_price / pos["entry"] - 1) - 2 * FUT_FEE,
                            "reason": reason, "closed": datetime.now(timezone.utc).isoformat()})


# ------------------------------------------------------------------ brokers
class PaperFutures:
    live = False

    def open(self, state, sym, d, price, notional, lev, opened_candle):
        fill = price * (1 + d * SLIPPAGE)
        qty = notional / fill
        state["cash"] -= notional * FUT_FEE
        state["positions"][sym] = {"dir": d, "qty": qty, "lots": None, "entry": fill, "lev": lev,
                                   "stop": fill * (1 - d * STOP_PCT), "take": fill * (1 + d * TAKE_PCT),
                                   "opened_candle": opened_candle}
        return state["positions"][sym]

    def close(self, state, sym, price, reason):
        pos = state["positions"].pop(sym)
        fill = price * (1 - pos["dir"] * SLIPPAGE)
        state["cash"] += pos["dir"] * pos["qty"] * (fill - pos["entry"]) - pos["qty"] * fill * FUT_FEE
        record_trade(state, sym, pos, fill, reason)
        return fill, state["trades"][-1]["pnl_pct"]


class LiveFutures:
    live = True

    def __init__(self):
        self.api = FuturesLive()

    def open(self, state, sym, d, price, notional, lev, opened_candle):
        lots = self.api.lots_for(sym, notional, price)
        if lots < 1:
            raise RuntimeError(f"{sym}: {notional:.2f} USDT is below one contract")
        self.api.prepare(sym, lev)
        self.api.market(sym, "buy" if d == 1 else "sell", lots, lev)
        time.sleep(2)
        ex = self.api.positions().get(sym)
        if not ex:
            raise RuntimeError(f"{sym}: order sent but no position visible - check KuCoin manually!")
        entry, open_lots = float(ex["avgEntryPrice"]), abs(float(ex["currentQty"]))
        stop, take = entry * (1 - d * STOP_PCT), entry * (1 + d * TAKE_PCT)
        try:
            stop_id = self.api.exchange_stop(sym, d, open_lots, lev, stop)["orderId"]
        except Exception as e:
            # FAIL-SAFE: never keep a leveraged position without an exchange-side stop
            self.api.market(sym, "sell" if d == 1 else "buy", open_lots, lev, reduce_only=True)
            raise RuntimeError(f"{sym}: stop order failed, position was closed again: {e}")
        mult = self.api.info(sym)["multiplier"]
        state["positions"][sym] = {"dir": d, "qty": open_lots * mult, "lots": open_lots, "entry": entry,
                                   "lev": lev, "stop": stop, "take": take, "stop_order_id": stop_id,
                                   "opened_candle": opened_candle}
        return state["positions"][sym]

    def close(self, state, sym, price, reason):
        pos = state["positions"][sym]
        self.api.market(sym, "sell" if pos["dir"] == 1 else "buy", pos["lots"], pos["lev"], reduce_only=True)
        state["positions"].pop(sym)                      # closed -> forget, then tidy the stop order
        try:
            self.api.cancel(pos["stop_order_id"])
        except Exception as e:
            log(f"{sym}: could not cancel stop order (harmless, reduce-only): {e}")
        record_trade(state, sym, pos, price, reason)
        return price, state["trades"][-1]["pnl_pct"]


# ------------------------------------------------------------------ paper-only market checks
def paper_checks(state, candles, broker):
    """Funding, stops, take-profits and liquidation using the last closed candle."""
    step = STEP_SECONDS[INTERVAL]
    for sym in list(state["positions"]):
        c = candles.get(sym)
        if c is None:
            continue
        pos, row = state["positions"][sym], c.iloc[-1]
        if str(row["timestamps"]) <= pos["opened_candle"]:
            continue                                     # candle that existed when we entered
        d = pos["dir"]
        state["cash"] -= d * FUNDING_8H * pos["qty"] * float(row["close"]) * (step / 28800)
        o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
        if (d == 1 and l <= pos["stop"]) or (d == -1 and h >= pos["stop"]):
            gap = min(pos["stop"], o) if d == 1 else max(pos["stop"], o)
            fill, pnl = broker.close(state, sym, gap, "stop")
            log(f"{sym}: STOP hit, closed @ {fill:.4f} ({pnl:+.2%})")
        elif (d == 1 and h >= pos["take"]) or (d == -1 and l <= pos["take"]):
            fill, pnl = broker.close(state, sym, pos["take"], "take-profit")
            log(f"{sym}: TAKE-PROFIT @ {fill:.4f} ({pnl:+.2%})")
    if state["positions"]:                               # liquidation test at the worst prices of the candle
        worst, notional = state["cash"], 0.0
        for sym, pos in state["positions"].items():
            row = candles[sym].iloc[-1]
            adverse = float(row["low"]) if pos["dir"] == 1 else float(row["high"])
            worst += pos["dir"] * pos["qty"] * (adverse - pos["entry"])
            notional += pos["qty"] * adverse
        if worst <= MMR * notional:
            log("!!! PAPER LIQUIDATION: cross account wiped out !!!")
            for sym, pos in list(state["positions"].items()):
                record_trade(state, sym, pos, float(candles[sym]["close"].iloc[-1]), "LIQUIDATED")
            state["positions"].clear()
            state["cash"] = 0.0


def live_sync(state, candles, broker):
    """Exchange is the source of truth in live mode."""
    ex = broker.api.positions()
    for sym in list(state["positions"]):
        if sym not in ex:
            pos = state["positions"].pop(sym)
            last = float(candles[sym]["close"].iloc[-1]) if sym in candles else pos["entry"]
            record_trade(state, sym, pos, last, "closed on exchange (stop/liquidation/manual)")
            log(f"{sym}: position no longer on exchange -> recorded as closed")
    untracked = [s for s in ex if s not in state["positions"]]
    if untracked:
        log(f"WARNING: positions on the exchange that this bot did not open (left untouched): {untracked}")
    for sym in list(state["positions"]):                 # software take-profit
        c = candles.get(sym)
        if c is None:
            continue
        pos, row = state["positions"][sym], c.iloc[-1]
        if str(row["timestamps"]) <= pos["opened_candle"]:
            continue
        hit = (pos["dir"] == 1 and float(row["high"]) >= pos["take"]) or \
              (pos["dir"] == -1 and float(row["low"]) <= pos["take"])
        if hit:
            fill, pnl = broker.close(state, sym, pos["take"], "take-profit")
            log(f"{sym}: TAKE-PROFIT, closed ({pnl:+.2%})")
    return untracked


# ------------------------------------------------------------------ one cycle
def run_cycle(state, predictor, broker, fetch=fetch_futures_candles):
    today = datetime.now(timezone.utc).date().isoformat()
    candles, prices = {}, {}
    for pair in PAIRS:
        sym = to_contract(pair)
        try:
            candles[sym] = fetch(sym, INTERVAL, n=450)
            prices[sym] = float(candles[sym]["close"].iloc[-1])
        except Exception as e:
            log(f"{sym}: data error: {e}")

    untracked = []
    if broker.live:
        untracked = live_sync(state, candles, broker)
    else:
        paper_checks(state, candles, broker)

    if broker.live:
        equity, available = broker.api.overview()
    else:
        equity = paper_equity(state, prices)
        available = equity - used_margin(state, prices)
    if state["day"] != today:
        state["day"], state["day_start_equity"] = today, equity
    halted = equity < state["day_start_equity"] * (1 - DAILY_LOSS_LIMIT)
    log(f"equity {equity:.2f} | available {available:.2f} | exposure {exposure(state, prices):.2f} "
        f"| open {[(s, p['dir'], p['lev']) for s, p in state['positions'].items()]}"
        + ("  ** DAILY LOSS LIMIT: no new entries **" if halted else ""))

    for pair in PAIRS:
        sym = to_contract(pair)
        if sym not in candles:
            continue
        c, price = candles[sym], prices[sym]
        try:
            pos = state["positions"].get(sym)
            er = expected_return(predictor, c, PRED_LEN, SAMPLES)
            exit_now, enter_dir = target_action(er, pos["dir"] if pos else 0, ENTRY_THRESHOLD)
            ctx = None              # RSI / open-interest readings (filters.py); OI also logs spikes
            if filters.OI_FILTER or enter_dir:
                ctx = filters.evaluate(sym, c, STEP_SECONDS[INTERVAL])
            log(f"{sym}: price {price:.4f} | forecast {er:+.2%} | "
                f"{'EXIT ' if exit_now else ''}{('LONG' if enter_dir == 1 else 'SHORT') if enter_dir else ('HOLD' if not exit_now else '')}"
                + (f" | {filters.describe(ctx)}" if ctx else ""))
            why = filters.entry_block(enter_dir, ctx) if enter_dir else None
            if why:                 # exit above still happens; only the new entry is vetoed
                log(f"{sym}: {'LONG' if enter_dir == 1 else 'SHORT'} entry blocked: {why}")
                enter_dir = 0
            if pos and exit_now:
                fill, pnl = broker.close(state, sym, price, "signal")
                log(f"{sym}: closed on signal ({pnl:+.2%})")
                if broker.live:
                    equity, available = broker.api.overview()
                else:
                    equity = paper_equity(state, prices)
                    available = equity - used_margin(state, prices)
            if enter_dir and sym not in untracked:
                if halted or len(state["positions"]) >= MAX_POSITIONS:
                    log(f"{sym}: entry skipped (limit reached)")
                    continue
                plan = choose_leverage_and_notional(equity, exposure(state, prices), er,
                                                    ENTRY_THRESHOLD, STOP_PCT, RISK_FRAC)
                if not plan or plan["notional"] < MIN_NOTIONAL:
                    log(f"{sym}: entry skipped (no room under exposure cap)")
                    continue
                if plan["notional"] / plan["leverage"] > available * 0.95:
                    log(f"{sym}: entry skipped (not enough free margin)")
                    continue
                pos = broker.open(state, sym, enter_dir, price, plan["notional"], plan["leverage"],
                                  str(c["timestamps"].iloc[-1]))
                log(f"{sym}: OPENED {'LONG' if enter_dir == 1 else 'SHORT'} {plan['notional']:.2f} USDT "
                    f"@ {pos['entry']:.4f} | leverage {plan['leverage']}x (strength {plan['strength']:.2f}) "
                    f"| stop {pos['stop']:.4f} take {pos['take']:.4f}")
        except Exception as e:
            log(f"{sym}: cycle error: {e}")
        save_state(state)
    save_state(state)


def main():
    if MODE == "live":
        if os.getenv("CONFIRM_LIVE") != "YES_I_ACCEPT_THE_RISK":
            raise SystemExit("MODE=live needs CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK. Run paper mode first.")
        log("*** LIVE FUTURES MODE: real leveraged orders will be placed (CROSS margin) ***")
        broker = LiveFutures()
    else:
        log("PAPER FUTURES MODE: simulated fills only")
        broker = PaperFutures()
    log(f"entry filters: {filters.summary()}")
    log(f"limits: max leverage {MAX_LEVERAGE}x | max total exposure {MAX_TOTAL_LEVERAGE}x equity | "
        f"margin/position <= {MAX_MARGIN_FRAC:.0%} | stop {STOP_PCT:.0%} | daily loss limit {DAILY_LOSS_LIMIT:.0%}")
    predictor = load_predictor()
    state = load_state()
    step = STEP_SECONDS[INTERVAL]
    while True:
        if os.path.exists("STOP"):
            log("STOP file found, exiting (open positions stay open, with their exchange stops, in live mode).")
            break
        run_cycle(state, predictor, broker)
        wait = step - (time.time() % step) + 20
        log(f"sleeping {wait / 60:.1f} min")
        time.sleep(wait)


if __name__ == "__main__":
    main()
