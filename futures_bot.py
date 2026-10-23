"""Kronos FUTURES bot (KuCoin classic futures, CROSS margin, long + short, bot-chosen leverage).

Started through bot.py when MARKET=futures.  MODE=paper (default) simulates everything.
Kill switch: create a file named STOP.
"""
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

import filters
from fkucoin import STEP_SECONDS, FuturesLive, fetch_futures_candles, to_contract
from futures_rules import (FUNDING_8H, FUT_FEE, MAX_LEVERAGE, MAX_MARGIN_FRAC, MAX_TOTAL_LEVERAGE,
                           MMR, choose_leverage_and_notional, target_action)
from learner import Learner, make_features
from strategy import expected_return, expected_returns, load_predictor

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
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))          # after a stop / losing close: no re-entry on that coin
COOLDOWN_ANY_SIDE = os.getenv("COOLDOWN_ANY_SIDE", "false").lower() in ("1", "true", "yes")  # false = same side only
MAX_SAME_SIDE = int(os.getenv("MAX_SAME_SIDE", "3"))          # max open longs (and max open shorts); 0 = no cap
EXCHANGE_TP = os.getenv("EXCHANGE_TP", "true").lower() in ("1", "true", "yes")  # take-profit order on KuCoin


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


_learner = None


def learner():
    """Signal journal + outcome labels + learned filter (learner.py); lives next to the state file."""
    global _learner
    if _learner is None:
        _learner = Learner("KB", os.getenv("LEARN_DIR") or os.path.dirname(os.path.abspath(STATE_PATH)),
                           STEP_SECONDS[INTERVAL])
    return _learner


def _open_ms(c):
    return pd.to_datetime(c["timestamps"]).values.astype("datetime64[ms]").astype("int64")


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


def start_cooldown(state, sym, pos, pnl_pct, reason):
    """After a stop or a losing close, block re-entry on that coin for COOLDOWN_BARS candles."""
    if COOLDOWN_BARS <= 0 or (pnl_pct >= 0 and "stop" not in reason):
        return
    until = time.time() + COOLDOWN_BARS * STEP_SECONDS[INTERVAL]
    state.setdefault("cooldown", {})[sym] = {"until": until, "dir": pos["dir"]}
    log(f"{sym}: cooldown {COOLDOWN_BARS} candle(s) for {'any side' if COOLDOWN_ANY_SIDE else ('LONG' if pos['dir'] == 1 else 'SHORT')} "
        f"after {reason} ({pnl_pct:+.2%})")


def cooldown_block(state, sym, d):
    cd = state.get("cooldown", {}).get(sym)
    if not cd or time.time() >= cd["until"]:
        return None
    if COOLDOWN_ANY_SIDE or cd["dir"] == d:
        return f"cooldown after a loss ({(cd['until'] - time.time()) / 60:.0f} min left)"
    return None


def record_trade(state, sym, pos, exit_price, reason):
    state["trades"].append({"symbol": sym, "dir": pos["dir"], "lev": pos["lev"], "entry": pos["entry"],
                            "exit": exit_price, "pnl_pct": pos["dir"] * (exit_price / pos["entry"] - 1) - 2 * FUT_FEE,
                            "reason": reason, "closed": datetime.now(timezone.utc).isoformat()})
    start_cooldown(state, sym, pos, state["trades"][-1]["pnl_pct"], reason)


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
                                   "take_order_id": None, "opened_candle": opened_candle}
        self.place_take(sym, state["positions"][sym])
        return state["positions"][sym]

    def place_take(self, sym, pos):
        """Take-profit-market on KuCoin, so it fills the moment price gets there (not at the next hourly check)."""
        if not EXCHANGE_TP:
            return
        try:
            pos["take_order_id"] = self.api.exchange_take(sym, pos["dir"], pos["lots"], pos["lev"], pos["take"])["orderId"]
            log(f"{sym}: take-profit order on exchange @ {pos['take']:.4f}")
        except Exception as e:
            pos["take_order_id"] = None
            log(f"{sym}: take-profit order failed, using the hourly software take-profit instead: {e}")

    def tidy(self, sym, pos):
        """Remove the leftover stop / take-profit after a position is gone (they must not hit a later position)."""
        for k in ("stop_order_id", "take_order_id"):
            if pos.get(k):
                try:
                    self.api.cancel(pos[k])
                except Exception:
                    pass
        try:
            self.api.cancel_stop_orders(sym)
        except Exception as e:
            log(f"{sym}: could not clear leftover stop orders: {e}")

    def close(self, state, sym, price, reason):
        pos = state["positions"][sym]
        self.api.market(sym, "sell" if pos["dir"] == 1 else "buy", pos["lots"], pos["lev"], reduce_only=True)
        state["positions"].pop(sym)                      # closed -> forget, then tidy the stop / TP orders
        self.tidy(sym, pos)
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
            broker.tidy(sym, pos)
            last = float(candles[sym]["close"].iloc[-1]) if sym in candles else pos["entry"]
            won = pos["dir"] * (last - pos["entry"]) > 0
            reason = "take-profit on exchange" if won else "stop on exchange (or liquidation/manual)"
            record_trade(state, sym, pos, pos["take"] if won else pos["stop"], reason)
            log(f"{sym}: position no longer on exchange -> recorded as {reason}")
        elif EXCHANGE_TP and not state["positions"][sym].get("take_order_id"):
            broker.place_take(sym, state["positions"][sym])          # positions opened before this version
    untracked = [s for s in ex if s not in state["positions"]]
    if untracked:
        log(f"WARNING: positions on the exchange that this bot did not open (left untouched): {untracked}")
    for sym in list(state["positions"]):                 # software take-profit (only if no TP order on KuCoin)
        c = candles.get(sym)
        if c is None or state["positions"][sym].get("take_order_id"):
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

    # 1) forecast every coin in one batch, so all decisions are made right after the candle close
    t0 = time.time()
    try:
        forecasts = expected_returns(predictor, candles, PRED_LEN, SAMPLES)
    except Exception as e:
        log(f"batch forecast failed ({e}), forecasting coin by coin")
        forecasts = {}
        for sym, c in candles.items():
            try:
                forecasts[sym] = expected_return(predictor, c, PRED_LEN, SAMPLES)
            except Exception as e2:
                log(f"{sym}: forecast error: {e2}")
    log(f"forecast {len(forecasts)} coins in {time.time() - t0:.0f}s")

    # outcomes of earlier signals whose holding window has now passed (learner.py)
    L = learner()
    try:
        n = L.label(candles, _open_ms, fetch=lambda s: fetch(s, INTERVAL, n=120))
        if n:
            log(f"learner: labelled {n} past signal(s)")
    except Exception as e:
        log(f"learner labelling failed: {e}")
    journal = []

    # 2) decide, strongest forecast first (so the best signals get the free slots)
    for sym in sorted(forecasts, key=lambda s: -abs(forecasts[s])):
        c, price, er = candles[sym], prices[sym], forecasts[sym]
        side_sig = 1 if er >= ENTRY_THRESHOLD else (-1 if er <= -ENTRY_THRESHOLD else 0)
        jr = dict(sym=sym, t=int(_open_ms(c)[-1]) + STEP_SECONDS[INTERVAL] * 1000, close=price, side=side_sig,
                  er=er, tp_pct=TAKE_PCT, sl_pct=STOP_PCT, hold=PRED_LEN, fee=2 * FUT_FEE,
                  action="signal" if side_sig else "none", p=None, feats={})
        journal.append(jr)
        try:
            pos = state["positions"].get(sym)
            exit_now, enter_dir = target_action(er, pos["dir"] if pos else 0, ENTRY_THRESHOLD)
            ctx = None              # RSI / open-interest readings (filters.py); OI also logs spikes
            if filters.OI_FILTER or enter_dir:
                ctx = filters.evaluate(sym, c, STEP_SECONDS[INTERVAL])
            try:
                ctx_j = ctx or filters.context(c, None, STEP_SECONDS[INTERVAL])
            except Exception:
                ctx_j = {}
            jr["feats"] = make_features(side_sig or (1 if er >= 0 else -1), er, jr["t"], dict(ctx_j, close=price))
            if pos and side_sig == pos["dir"]:
                jr["action"] = "skip: already in a trade"
            log(f"{sym}: price {price:.4f} | forecast {er:+.2%} | "
                f"{'EXIT ' if exit_now else ''}{('LONG' if enter_dir == 1 else 'SHORT') if enter_dir else ('HOLD' if not exit_now else '')}"
                + (f" | {filters.describe(ctx)}" if ctx else ""))
            why = (cooldown_block(state, sym, enter_dir) or filters.entry_block(enter_dir, ctx)) if enter_dir else None
            if enter_dir and not why:
                why, jr["p"] = L.gate(jr["feats"])
                if jr["p"] is not None:
                    log(f"{sym}: learned expected return {jr['p']:+.2%}")
            if why:                 # exit above still happens; only the new entry is vetoed
                log(f"{sym}: {'LONG' if enter_dir == 1 else 'SHORT'} entry blocked: {why}")
                jr["action"] = f"blocked: {why}"
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
                    jr["action"] = "skip: daily loss limit" if halted else "skip: no free slot"
                    continue
                same = sum(1 for p in state["positions"].values() if p["dir"] == enter_dir)
                if MAX_SAME_SIDE and same >= MAX_SAME_SIDE:
                    log(f"{sym}: entry skipped (already {same} {'longs' if enter_dir == 1 else 'shorts'}, MAX_SAME_SIDE)")
                    jr["action"] = "skip: max same side"
                    continue
                plan = choose_leverage_and_notional(equity, exposure(state, prices), er,
                                                    ENTRY_THRESHOLD, STOP_PCT, RISK_FRAC)
                if not plan or plan["notional"] < MIN_NOTIONAL:
                    log(f"{sym}: entry skipped (no room under exposure cap)")
                    jr["action"] = "skip: exposure cap"
                    continue
                if plan["notional"] / plan["leverage"] > available * 0.95:
                    log(f"{sym}: entry skipped (not enough free margin)")
                    jr["action"] = "skip: margin"
                    continue
                pos = broker.open(state, sym, enter_dir, price, plan["notional"], plan["leverage"],
                                  str(c["timestamps"].iloc[-1]))
                jr["action"] = "opened"
                log(f"{sym}: OPENED {'LONG' if enter_dir == 1 else 'SHORT'} {plan['notional']:.2f} USDT "
                    f"@ {pos['entry']:.4f} | leverage {plan['leverage']}x (strength {plan['strength']:.2f}) "
                    f"| stop {pos['stop']:.4f} take {pos['take']:.4f}")
            elif enter_dir and sym in untracked:
                jr["action"] = "skip: untracked position on coin"
        except Exception as e:
            log(f"{sym}: cycle error: {e}")
            jr["action"] = "error"
        save_state(state)
    save_state(state)
    try:
        L.record(journal)
        L.maybe_retrain()
    except Exception as e:
        log(f"learner failed: {e}")


def main():
    if MODE == "live":
        if os.getenv("CONFIRM_LIVE") != "YES_I_ACCEPT_THE_RISK":
            raise SystemExit("MODE=live needs CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK. Run paper mode first.")
        log("*** LIVE FUTURES MODE: real leveraged orders will be placed (CROSS margin) ***")
        broker = LiveFutures()
    else:
        log("PAPER FUTURES MODE: simulated fills only")
        broker = PaperFutures()
    log(f"entry filters: {filters.summary()} | cooldown {COOLDOWN_BARS}c after a loss | "
        f"max {MAX_SAME_SIDE or 'unlimited'} per side | take-profit {'on exchange' if EXCHANGE_TP else 'software'}")
    log(f"limits: max leverage {MAX_LEVERAGE}x | max total exposure {MAX_TOTAL_LEVERAGE}x equity | "
        f"margin/position <= {MAX_MARGIN_FRAC:.0%} | stop {STOP_PCT:.0%} | daily loss limit {DAILY_LOSS_LIMIT:.0%}")
    predictor = load_predictor()
    state = load_state()
    learner()
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
