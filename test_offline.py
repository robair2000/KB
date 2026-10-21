"""Offline logic test: synthetic candles + stub predictor (no network, no model download)."""
import numpy as np
import pandas as pd

import bot
from backtest import run_backtest
from strategy import decide, expected_return


class StubPredictor:
    """Pretends to forecast continuation of the last 12-candle trend."""
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
        close = df["close"].values
        drift = (close[-1] / close[-13] - 1) / 12
        path = close[-1] * (1 + drift) ** np.arange(1, pred_len + 1)
        out = pd.DataFrame({"open": path, "high": path, "low": path, "close": path,
                            "volume": 0.0, "amount": 0.0})
        out.index = y_timestamp
        return out


def synthetic(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, n)))
    ts = pd.date_range("2026-01-01", periods=n, freq="1h")
    spread = close * 0.003
    return pd.DataFrame({"timestamps": ts, "open": close, "high": close + spread, "low": close - spread,
                         "close": close, "volume": 10.0, "amount": close * 10})


df = synthetic()
er = expected_return(StubPredictor(), df.iloc[:500])
print("expected_return:", round(er, 4), "->", decide(er))
assert decide(0.02) == "ENTER" and decide(-0.01) == "EXIT" and decide(0.003) == "HOLD"

n = bot.position_notional(equity=1000, cash=1000)
print("position size for 1000 USDT equity:", n)
assert abs(n - 250) < 1e-6          # 1%/3% = 333 -> capped at 25%
assert bot.position_notional(1000, cash=50) == 50

res = run_backtest(df, StubPredictor(), step=6, verbose=False)
print(res)
assert res["max_drawdown_pct"] <= 0 and res["trades"] >= 0
print("ALL OFFLINE CHECKS PASSED")
