"""Walk-forward backtest of the exact same signal + risk rules the bot uses.

    python backtest.py --symbol BTC-USDT --candles 1500 --step 6
    python backtest.py --csv my_candles.csv        # offline (columns: timestamps,open,high,low,close,volume,amount)

The model only ever sees candles up to the decision point (no look-ahead).
"""
import argparse

import numpy as np
import pandas as pd

from bot import FEE, MAX_POS_FRAC, RISK_FRAC, SLIPPAGE, STOP_PCT, TAKE_PCT
from strategy import decide, expected_return


def run_backtest(candles, predictor, pred_len=6, samples=3, step=6, lookback=400,
                 entry_threshold=0.008, start_cash=1000.0, verbose=True):
    c = candles.reset_index(drop=True)
    cash, pos, trades, equity_curve = start_cash, None, [], []
    start = lookback
    for j in range(start, len(c)):
        row = c.iloc[j]
        close = float(row["close"])
        # hard exits on this candle
        if pos:
            if row["low"] <= pos["stop"]:
                fill, why = pos["stop"] * (1 - SLIPPAGE), "stop"
            elif row["high"] >= pos["take"]:
                fill, why = pos["take"] * (1 - SLIPPAGE), "take"
            else:
                fill = None
            if fill:
                cash += pos["qty"] * fill * (1 - FEE)
                trades.append(fill / pos["entry"] - 1 - 2 * FEE)
                pos = None
        # decision point
        if (j - start) % step == 0:
            er = expected_return(predictor, c.iloc[: j + 1], pred_len, samples, lookback)
            action = decide(er, entry_threshold)
            if pos and action == "EXIT":
                fill = close * (1 - SLIPPAGE)
                cash += pos["qty"] * fill * (1 - FEE)
                trades.append(fill / pos["entry"] - 1 - 2 * FEE)
                pos = None
            elif not pos and action == "ENTER":
                eq = cash
                notional = max(0.0, min(eq * RISK_FRAC / STOP_PCT, eq * MAX_POS_FRAC, cash))
                if notional >= 10:
                    fill = close * (1 + SLIPPAGE)
                    pos = {"qty": notional * (1 - FEE) / fill, "entry": fill,
                           "stop": fill * (1 - STOP_PCT), "take": fill * (1 + TAKE_PCT)}
                    cash -= notional
        equity_curve.append(cash + (pos["qty"] * close if pos else 0.0))
        if verbose and (j - start) % (step * 20) == 0:
            print(f"  {j - start}/{len(c) - start} candles, equity {equity_curve[-1]:.2f}", flush=True)
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    first_close = float(c["close"].iloc[start])
    return {
        "candles_tested": len(eq),
        "return_pct": (eq[-1] / start_cash - 1) * 100,
        "buy_and_hold_pct": (float(c["close"].iloc[-1]) / first_close - 1) * 100,
        "trades": len(trades),
        "win_rate_pct": (np.mean([t > 0 for t in trades]) * 100) if trades else 0.0,
        "avg_trade_pct": (np.mean(trades) * 100) if trades else 0.0,
        "max_drawdown_pct": float(((eq - peak) / peak).min() * 100),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC-USDT")
    ap.add_argument("--interval", default="1hour")
    ap.add_argument("--candles", type=int, default=1500, help="history length incl. 400 lookback")
    ap.add_argument("--step", type=int, default=6, help="decide every N candles (bigger = faster)")
    ap.add_argument("--csv", help="load candles from CSV instead of KuCoin")
    ap.add_argument("--model", default=None, help="mini | small | base")
    a = ap.parse_args()

    from strategy import load_predictor
    if a.csv:
        df = pd.read_csv(a.csv, parse_dates=["timestamps"])
    else:
        from kucoin import fetch_candles
        df = fetch_candles(a.symbol, a.interval, n=a.candles)
    print(f"Backtesting {a.symbol}: {len(df)} candles, deciding every {a.step} candles")
    res = run_backtest(df, load_predictor(a.model), step=a.step)
    print("\n=== RESULT (fees + slippage included) ===")
    for k, v in res.items():
        print(f"{k:>18}: {v:.2f}" if isinstance(v, float) else f"{k:>18}: {v}")
    print("\nIf return_pct does not clearly beat buy_and_hold_pct (and drawdown is tolerable), do NOT go live.")
