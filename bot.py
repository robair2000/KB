"""Kronos trading bot for KuCoin spot.  Default MODE=paper (no real orders).

    python bot.py              # runs forever, one decision per closed candle
    touch STOP                 # kill switch: bot exits at the next cycle
"""
import json
import os
import time
from datetime import datetime, timezone

import filters
from kucoin import INTERVAL_SECONDS, KuCoinLive, fetch_candles
from strategy import decide, expected_return, load_predictor

# ---------------------------------------------------------------- config
PAIRS = [p.strip() for p in os.getenv("PAIRS", "BTC-USDT,ETH-USDT,SOL-USDT,LINK-USDT,DOGE-USDT").split(",")]
MODE = os.getenv("MODE", "paper").lower()            # paper | live
INTERVAL = os.getenv("INTERVAL", "1hour")
PRED_LEN = int(os.getenv("PRED_LEN", "6"))           # candles to forecast
SAMPLES = int(os.getenv("SAMPLES", "3"))             # forecast paths averaged
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.008"))  # forecast gain needed (> fees)
FEE = float(os.getenv("FEE", "0.001"))               # per side
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0005"))    # per side
STOP_PCT = float(os.getenv("STOP_PCT", "0.03"))
TAKE_PCT = float(os.getenv("TAKE_PCT", "0.06"))
RISK_FRAC = float(os.getenv("RISK_FRAC", "0.01"))    # equity risked per trade
MAX_POS_FRAC = float(os.getenv("MAX_POS_FRAC", "0.25"))  # max equity in one position
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))  # stop opening trades
START_CASH = float(os.getenv("START_CASH", "1000"))  # paper only
MIN_NOTIONAL = 10.0
STATE_PATH = os.getenv("STATE_PATH", "state.json")


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


# ---------------------------------------------------------------- sizing
def position_notional(equity, cash, stop_pct=STOP_PCT, risk_frac=RISK_FRAC, max_frac=MAX_POS_FRAC):
    """USDT to deploy so that hitting the stop loses ~risk_frac of equity (capped)."""
    notional = equity * risk_frac / stop_pct
    return max(0.0, min(notional, equity * max_frac, cash))


# ---------------------------------------------------------------- state
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


def equity_of(state, prices):
    return state["cash"] + sum(p["qty"] * prices.get(sym, p["entry"]) for sym, p in state["positions"].items())


# ---------------------------------------------------------------- execution
class Broker:
    def __init__(self, mode):
        self.live = mode == "live"
        self.api = KuCoinLive() if self.live else None

    def buy(self, state, sym, price, notional):
        fill = price * (1 + SLIPPAGE)
        if self.live:
            self.api.market_buy(sym, notional)
            time.sleep(2)
            qty = self.api.base_balance(sym.split("-")[0])
            state["cash"] = self.api.usdt_balance()
        else:
            qty = notional * (1 - FEE) / fill
            state["cash"] -= notional
        state["positions"][sym] = {"qty": qty, "entry": fill,
                                   "stop": fill * (1 - STOP_PCT), "take": fill * (1 + TAKE_PCT),
                                   "opened": datetime.now(timezone.utc).isoformat()}
        return qty, fill

    def sell(self, state, sym, price, reason):
        pos = state["positions"].pop(sym)
        fill = price * (1 - SLIPPAGE)
        if self.live:
            self.api.market_sell(sym, pos["qty"])
            time.sleep(2)
            state["cash"] = self.api.usdt_balance()
        else:
            state["cash"] += pos["qty"] * fill * (1 - FEE)
        pnl = (fill / pos["entry"] - 1) - 2 * FEE
        state["trades"].append({"symbol": sym, "entry": pos["entry"], "exit": fill, "pnl_pct": pnl,
                                "reason": reason, "closed": datetime.now(timezone.utc).isoformat()})
        return fill, pnl


# ---------------------------------------------------------------- one cycle
def run_cycle(state, predictor, broker):
    today = datetime.now(timezone.utc).date().isoformat()
    prices, candles = {}, {}
    for sym in PAIRS:
        try:
            candles[sym] = fetch_candles(sym, INTERVAL, n=450)
            prices[sym] = float(candles[sym]["close"].iloc[-1])
        except Exception as e:
            log(f"{sym}: data error: {e}")
    if broker.live:
        state["cash"] = broker.api.usdt_balance()
    equity = equity_of(state, prices)
    if state["day"] != today:
        state["day"], state["day_start_equity"] = today, equity
    halted = equity < state["day_start_equity"] * (1 - DAILY_LOSS_LIMIT)
    log(f"equity {equity:.2f} USDT | cash {state['cash']:.2f} | open {list(state['positions'])}"
        + ("  ** DAILY LOSS LIMIT: no new entries **" if halted else ""))

    for sym in PAIRS:
        if sym not in candles:
            continue
        c, price = candles[sym], prices[sym]
        pos = state["positions"].get(sym)
        try:
            # 1) hard exits first: stop / take-profit measured on the last closed candle
            if pos:
                low, high = float(c["low"].iloc[-1]), float(c["high"].iloc[-1])
                if low <= pos["stop"]:
                    fill, pnl = broker.sell(state, sym, pos["stop"], "stop")
                    log(f"{sym}: STOP hit, sold @ {fill:.4f} ({pnl:+.2%})")
                    continue
                if high >= pos["take"]:
                    fill, pnl = broker.sell(state, sym, pos["take"], "take-profit")
                    log(f"{sym}: TAKE-PROFIT, sold @ {fill:.4f} ({pnl:+.2%})")
                    continue
            # 2) Kronos forecast
            er = expected_return(predictor, c, PRED_LEN, SAMPLES)
            action = decide(er, ENTRY_THRESHOLD)
            ctx = None              # RSI / open-interest readings (filters.py); OI also logs spikes
            if filters.OI_FILTER or (not pos and action == "ENTER"):
                ctx = filters.evaluate(sym, c, INTERVAL_SECONDS[INTERVAL])
            log(f"{sym}: price {price:.4f} | forecast {er:+.2%} over {PRED_LEN} candles | {action}"
                + (f" | {filters.describe(ctx)}" if ctx else ""))
            if pos and action == "EXIT":
                fill, pnl = broker.sell(state, sym, price, "signal")
                log(f"{sym}: SOLD on signal @ {fill:.4f} ({pnl:+.2%})")
            elif not pos and action == "ENTER":
                why = filters.entry_block(1, ctx)
                if why:
                    log(f"{sym}: entry blocked: {why}")
                    continue
                if halted or len(state["positions"]) >= MAX_POSITIONS:
                    log(f"{sym}: entry skipped (limit reached)")
                    continue
                notional = position_notional(equity, state["cash"])
                if notional < MIN_NOTIONAL:
                    log(f"{sym}: entry skipped (size {notional:.2f} < {MIN_NOTIONAL})")
                    continue
                qty, fill = broker.buy(state, sym, price, notional)
                log(f"{sym}: BOUGHT {qty:.6f} @ {fill:.4f} ({notional:.2f} USDT) "
                    f"stop {state['positions'][sym]['stop']:.4f}")
        except Exception as e:
            log(f"{sym}: cycle error: {e}")
        save_state(state)
    save_state(state)


def main():
    if os.getenv("MARKET", "spot").lower() == "futures":
        from futures_bot import main as futures_main     # futures: long/short, cross margin, leverage
        return futures_main()
    if MODE == "live":
        if os.getenv("CONFIRM_LIVE") != "YES_I_ACCEPT_THE_RISK":
            raise SystemExit("MODE=live needs CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK. Run paper mode first.")
        log("*** LIVE MODE: real orders will be placed ***")
    else:
        log("PAPER MODE: simulated fills only")
    log(f"entry filters: {filters.summary()}")
    predictor = load_predictor()
    broker = Broker(MODE)
    state = load_state()
    step = INTERVAL_SECONDS[INTERVAL]
    while True:
        if os.path.exists("STOP"):
            log("STOP file found, exiting.")
            break
        run_cycle(state, predictor, broker)
        wait = step - (time.time() % step) + 20      # wake 20s after the next candle closes
        log(f"sleeping {wait/60:.1f} min")
        time.sleep(wait)


if __name__ == "__main__":
    main()
