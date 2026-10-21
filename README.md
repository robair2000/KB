# Kronos KuCoin bot

Spot bot that forecasts candles with the official [Kronos](https://github.com/shiyu-coder/Kronos)
model and trades long/flat on KuCoin. **Paper mode by default.**

## Layout
- `model/`        official Kronos code (MIT, copied from shiyu-coder/Kronos). Weights download automatically from Hugging Face on first run.
- `strategy.py`   loads Kronos, forecasts, turns forecast into ENTER / EXIT / HOLD
- `kucoin.py`     candles + signed live orders
- `bot.py`        trading loop, risk rules, paper/live broker
- `backtest.py`   walk-forward backtest using the same rules
- `test_offline.py` logic test with no network

## Order of use
1. `pip install -r requirements.txt`
2. `python backtest.py --symbol BTC-USDT --candles 1500 --step 6`   (does it beat buy-and-hold after fees?)
3. `python bot.py` in `MODE=paper` for several weeks; read `state.json` / logs.
4. Only if 2 and 3 look good: `MODE=live CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK` plus KUCOIN_API_KEY / KUCOIN_API_SECRET / KUCOIN_PASSPHRASE
   (create the API key with trade permission only, NO withdrawal permission, IP-restricted if possible).

Kill switch: create a file named `STOP` next to `bot.py`.

## Env vars
MODE, KRONOS_MODEL (mini|small|base), PAIRS, INTERVAL, PRED_LEN, SAMPLES, ENTRY_THRESHOLD,
STOP_PCT, TAKE_PCT, RISK_FRAC, MAX_POS_FRAC, MAX_POSITIONS, DAILY_LOSS_LIMIT, FEE, SLIPPAGE, START_CASH.

Not financial advice. A model forecast is not a guarantee of profit.
