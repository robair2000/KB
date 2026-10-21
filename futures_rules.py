"""Futures config + pure sizing/leverage rules (shared by bot and backtest, no network).

HOW THE BOT "CHOOSES" LEVERAGE
Leverage never decides how much you risk. Risk per trade is fixed by the stop distance:
    notional = equity * RISK_FRAC * strength / STOP_PCT        (strength 1..2 from forecast size)
Leverage is then the smallest value that keeps the margin of one position under
MAX_MARGIN_FRAC of equity, capped by MAX_LEVERAGE and by a liquidation-distance check.
Because the account is CROSS margin, total exposure is also capped:
    sum(notional of all positions) <= MAX_TOTAL_LEVERAGE * equity
"""
import math
import os

MMR = float(os.getenv("MMR", "0.005"))                  # approx maintenance margin rate (conservative guess)
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "3"))      # hard cap on leverage per position
MAX_TOTAL_LEVERAGE = float(os.getenv("MAX_TOTAL_LEVERAGE", "3"))  # cap on total notional / equity (cross!)
MAX_MARGIN_FRAC = float(os.getenv("MAX_MARGIN_FRAC", "0.25"))     # max equity used as margin per position
LIQ_BUFFER = float(os.getenv("LIQ_BUFFER", "2.0"))      # liquidation must be >= this many stop-distances away
FUT_FEE = float(os.getenv("FUT_FEE", "0.0006"))         # taker fee per side (check your KuCoin fee tier)
FUNDING_8H = float(os.getenv("FUNDING_8H", "0.0001"))   # backtest/paper assumption: longs pay, shorts receive


def choose_leverage_and_notional(equity, current_notional, er, entry_threshold, stop_pct, risk_frac):
    """Return {'leverage', 'notional', 'strength'} or None if there is no room for a new position."""
    if equity <= 0:
        return None
    strength = min(max(abs(er) / entry_threshold, 1.0), 2.0)
    notional = equity * risk_frac * strength / stop_pct
    notional = min(notional, equity * MAX_TOTAL_LEVERAGE - current_notional)
    if notional <= 0:
        return None
    lev = max(1, math.ceil(notional / (equity * MAX_MARGIN_FRAC)))
    lev = min(lev, MAX_LEVERAGE)
    while lev > 1 and (1.0 / lev - MMR) < LIQ_BUFFER * stop_pct:
        lev -= 1
    notional = min(notional, equity * MAX_MARGIN_FRAC * lev)
    return {"leverage": lev, "notional": notional, "strength": strength}


def target_action(er, held_dir, threshold):
    """held_dir: +1 long, -1 short, 0 flat.  Returns (exit_now, enter_dir)."""
    exit_now = (held_dir == 1 and er <= 0) or (held_dir == -1 and er >= 0)
    enter_dir = 0
    if held_dir == 0 or exit_now:
        if er >= threshold:
            enter_dir = 1
        elif er <= -threshold:
            enter_dir = -1
    return exit_now, enter_dir


def cross_liquidation_price(direction, qty, entry, wallet_cash, mmr=MMR):
    """Price at which a single cross-margin position is liquidated (account equity == mmr * notional)."""
    if direction == 1:
        return max(0.0, (qty * entry - wallet_cash) / (qty * (1 - mmr)))
    return (wallet_cash + qty * entry) / (qty * (1 + mmr))
