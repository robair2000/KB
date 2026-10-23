# Kronos multi-coin limit-order bot — KuCoin Futures, long + short

A Render background worker. Every 1-hour close (`TIMEFRAME`) it forecasts the next candle for a basket of
KuCoin USDT perpetuals with [Kronos](https://github.com/shiyu-coder/Kronos), ranks the signals,
and opens the best ones, long or short. Take-profits are always maker limits; entries and stops
use limit or market orders as configured (`ENTRY_MODE`, `STOP_ORDER_TYPE`).

## How it picks coins

* `SYMBOLS` set → trade exactly that list.
* `SYMBOLS` empty → every `UNIVERSE_REFRESH_BARS` candles, take the most liquid USDT perpetuals
  (24h turnover ≥ `MIN_VOLUME_USDT`, spread ≤ `MAX_SPREAD_PCT`, not in `BLACKLIST`), up to `UNIVERSE_SIZE`,
  plus `ALWAYS_INCLUDE`.
* Each candle, every coin gets `SAMPLE_PATHS` forecasts. A coin signals LONG or SHORT when the mean
  forecast move ≥ `MIN_EDGE_LONG` / `MIN_EDGE_SHORT` and ≥ `MIN_AGREEMENT` of paths agree.
* Signals are ranked by `RANK_BY` (`sharpe` = edge ÷ path disagreement, `edge`, or `agreement`) and
  the top ones are opened up to `MAX_POSITIONS`, `MAX_SAME_SIDE` and `MAX_TOTAL_NOTIONAL_USDT`.

## Entry filters (`filters.py`)

Kronos only sees OHLCV. These rules can veto a **new** entry; they never close a trade, and an
exit on a model flip still happens. Every blocked signal is logged, e.g. `LINK LONG blocked: RSI 78 > 75`.

| Rule | Switch | Blocks |
|------|--------|--------|
| RSI | `RSI_FILTER` | longs above `RSI_LONG_MAX`, shorts below `RSI_SHORT_MIN` |
| Trend | `EMA_FILTER` | longs below the `EMA_PERIOD` EMA, shorts above it |
| Bollinger | `BB_FILTER` (off by default) | longs closing above the upper band, shorts below the lower |
| OI spike | `OI_SPIKE_MODE` = `avoid` | any entry for `OI_SPIKE_BARS` candles after a sudden OI jump or drop |
| | `OI_SPIKE_MODE` = `follow` | entries against the price direction of the spike candle |
| OI positioning | `OI_FILTER`, `OI_MIN_CHANGE` | longs while price fell and OI rose over `OI_LOOKBACK_BARS` (shorts building), shorts while price rose and OI rose (longs building) |

**OI spikes** are judged per coin: a candle's OI change is a spike when it is `OI_SPIKE_Z` standard
deviations away from that coin's last `OI_SPIKE_WINDOW` candles and at least `OI_SPIKE_MIN_PCT`.
A 2% hourly OI drop is a big event for BTC but routine for a small altcoin; a fixed threshold
would miss the first or fire constantly on the second. Every spike in the universe is posted to the
webhook (`OI_SPIKE_ALERT`) with a plain reading: OI up + price up = longs piling in, OI up + price
down = shorts piling in, OI down + price down = longs liquidated, OI down + price up = shorts squeezed.

KuCoin Futures publishes no open-interest history, so OI is read from `OI_EXCHANGE`
(`binanceusdm` by default, or `bybit`), public data, no keys. Coins listed there as 1000x
(`1000PEPE`) are matched automatically. If a coin has no OI there, the OI rules are skipped for it
(or the coin is skipped with `OI_REQUIRED=true`). OI is sampled every `OI_PERIOD` and only readings
from before each candle close are used, so the backtest has no look-ahead.

Changing `TIMEFRAME` on a running bot is handled (bar counters in `state.json` are rescaled and
cooldowns cleared), but it is cleanest to switch while flat.

## How each trade runs

1. **Entry** (`ENTRY_MODE`):
   - `limit`: post-only limit `ENTRY_PULLBACK` of the way to the forecast low (long) / high (short).
     Cheapest (maker fee), but often misses the trades where price runs straight away.
   - `market`: buys/sells at once at the touch. Always fills, pays the taker fee and the spread.
   - `limit_then_market` (default): the pullback limit first; if it hasn't filled after
     `ENTRY_CHASE_SECONDS`, the rest is taken at market, **unless** price has run so far that the
     take-profit no longer clears the taker fee, in which case the trade is dropped.
   Every entry must leave the take-profit clear of the round-trip fees + `MIN_NET_EDGE`
   (maker + maker for limit entries, taker + maker for market ones).
2. **Take-profit**: reduce-only post-only limit at `TP_CAPTURE` of the forecast move (maker).
3. **Exchange stop** (`EXCHANGE_STOP`): reduce-only stop resting on KuCoin at `HARD_STOP_PCT`, so the
   position is protected even if the bot is down. A stop-**market** with `STOP_ORDER_TYPE=market`
   (default), a stop-limit with `STOP_ORDER_TYPE=limit`.
4. **Soft stop** at `STOP_PCT`: closed at market right away (`STOP_ORDER_TYPE=market`), or with a
   limit at the touch (`limit`).
5. **Max hold / model flip** (`MAX_HOLD_BARS`, `EXIT_ON_FLIP`): a limit at the touch first; if it
   hasn't filled after `EXIT_CHASE_SECONDS`, the rest is closed at market.

> ⚠️ A market stop always fills but can slip in a fast move. A stop-limit can be skipped over
> entirely if price gaps past its limit, leaving you in the trade. With leverage, keep
> `HARD_STOP_PCT` well inside your liquidation distance either way.

## Tiny-priced coins (PEPE, SHIB, …)

* Sizes are whole contracts using each market's own multiplier (e.g. 1 BTC contract = 0.001 BTC,
  1 PEPE contract = 1,000,000 PEPE), and prices are rounded to each market's tick (down to 1e-9).
* Prices are rescaled to about 1000 before they go into Kronos. Its normaliser divides by
  `std + 1e-5`, which would otherwise flatten a 0.000004 price series into a constant line.
* Logs and alerts print full decimals (`0.000003921`), never `0.00` or scientific notation.

## Settings

Every setting is an environment variable. All of them are listed, with explanations, in
`.env.example` and pre-filled in `render.yaml`, so they all appear in the Render dashboard.
They are grouped as:

| Group | Covers |
|-------|--------|
| Exchange | mode, KuCoin keys |
| Universe | coin selection, ranking, max positions |
| Futures | leverage, isolated/cross margin, stop trigger price |
| Direction | allow long / allow short |
| Model | Kronos size, lookback, horizon, sampling, batch size |
| Signal | edge thresholds, entry/TP placement, fees, exit on flip |
| Sizing | fixed USDT or % of equity, caps |
| Risk | stops, hold time, cooldown, daily loss / trade limits, kill switch, trading hours |
| Ops | polling, state file, alert webhook, heartbeat |

To add a setting, add a row to `VARS` in `config.py`, then run `python make_render.py` to regenerate
`render.yaml` and `.env.example`.

## Modes

| `MODE` | What happens | Needs |
|--------|--------------|-------|
| `paper` | Real KuCoin prices, simulated fills, 1000 USDT play money | nothing |
| `live`  | Real money | `API_KEY`, `API_SECRET`, `API_PASSWORD` (passphrase), `CONFIRM_LIVE=yes` |

KuCoin Futures has no testnet, so go paper → live with small `NOTIONAL_USDT`.
Create the API key with **Futures trading only, no withdrawals**, ideally IP-restricted.
Dedicate the futures account to the bot. It leaves positions it didn't open alone, but they count
toward `MAX_POSITIONS`, and it cancels its own stop orders on a coin when it closes a trade there.

## Deploy on Render

1. Push this folder to a GitHub repo.
2. Render → **New → Blueprint** → select the repo. You get a Frankfurt background worker
   (Standard, 2 GB) with a 1 GB disk for `state.json`.
3. Deploy in `paper` mode and watch the logs: one line per candle with the universe size,
   forecast time and signals.
4. For live, enter the secrets in the dashboard, set `MODE=live` and `CONFIRM_LIVE=yes`.

## Backtest first

```bash
bash build.sh
python backtest.py --bars 400                                   # auto universe (top 8)
python backtest.py --bars 400 --symbols BTC/USDT:USDT,PEPE/USDT:USDT
```

It replays the same ranking, slots and limit-order rules with a conservative fill model, then
prints per-coin forecast hit-rate and correlation, plus P&L by side, exit reason and coin. Coins
where the correlation is about zero have no edge: blacklist them.

Kronos runs once per bar and the same forecasts are then traded in **every entry mode** (`limit`,
`market`, `limit_then_market`), **with and without the filters**: six results side by side, with
taker fees and `--spread` slippage on every market fill. Narrow it with `--entry` and `--filters`.
`limit_then_market` uses 5-minute candles to judge whether the limit would have filled before the chase. It also shows, per rule, how the
signals it blocked moved on the next candle: a rule earns its place only if its blocked signals
did worse than the ones that passed. Turn off any rule that doesn't. Binance keeps 30 days of OI
history and the spike detector needs `OI_SPIKE_WINDOW` candles of it first, so on 1h keep `--bars`
under about 550.

Not financial advice. Kronos is a research model and leveraged futures can lose more than the
trade size. Run paper mode first and start small.
