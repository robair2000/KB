"""Walk-forward backtest of the bot's exact rules across several coins.

    python backtest.py --bars 300                              # auto universe (same filters as the bot)
    python backtest.py --bars 300 --symbols BTC/USDT:USDT,PEPE/USDT:USDT
    python backtest.py --bars 300 --filters off                # Kronos signal only

Kronos runs once; the same forecasts are then traded in every entry mode (limit / market /
limit_then_market) with and without the entry filters (RSI / EMA / Bollinger / open interest, see
filters.py), so every comparison is like for like. Market fills pay TAKER_FEE plus --spread slippage.
Binance keeps only 30 days of OI history and the spike detector needs OI_SPIKE_WINDOW candles of it
before the first trading bar: with OI_EXCHANGE=binanceusdm on 1h keep --bars under ~550.

Uses the same signal, ranking (RANK_BY), MAX_POSITIONS / MAX_SAME_SIDE slots, entry, TP and stop
settings as bot.py. Fill model (deliberately conservative):
  * an entry limit fills only if a later candle trades strictly THROUGH it
  * a market entry fills at the next candle's open plus --spread (taker fee)
  * a take-profit fills only if a later candle trades strictly through it (maker fee)
  * if stop and TP are both touched in one candle, the stop is assumed first
  * a market stop fills at the stop price (or the open, if the candle gapped past it) plus --spread,
    taker fee; a limit stop fills at the worse of stop and close plus --spread
  * max-hold / flip exits fill at that candle's close plus --spread
Sizing is NOTIONAL_USDT per trade (fixed) so results are in USDT.
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from bot import LONG, NAME, allowed, plan_entry, score, signal_side, worth_it
from broker import PaperBroker, make_exchange
from filters import OIFeed, context, entry_block, oi_bars_needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=300)
    ap.add_argument("--symbols", default=",".join(C.SYMBOL_LIST))
    ap.add_argument("--top", type=int, default=min(C.UNIVERSE_SIZE, 8), help="auto universe size")
    ap.add_argument("--spread", type=float, default=0.0003,
                    help="slippage per side on market fills, stops and touch exits (0.0003 = 0.03%%)")
    ap.add_argument("--entry", choices=["all", "limit", "market", "limit_then_market"], default="all",
                    help="entry mode(s) to replay; all = compare the three side by side")
    ap.add_argument("--filters", choices=["both", "on", "off"], default="both",
                    help="trade the forecasts with filters, without, or both side by side")
    a = ap.parse_args()

    from forecaster import Forecaster
    br = PaperBroker(make_exchange(C.EXCHANGE), 0, C.MAKER_FEE)
    syms = [s for s in a.symbols.split(",") if s] or [
        s for s, v in br.universe() if v >= C.MIN_VOLUME_USDT][:a.top]
    need = C.LOOKBACK + a.bars + C.MAX_HOLD_BARS + 2
    data = {}
    for s in syms:
        df = br.candles(s, C.TIMEFRAME, need)
        if len(df) >= C.LOOKBACK + 50:
            data[s] = df.set_index("ts")
    common = sorted(set.intersection(*(set(d.index) for d in data.values())))
    data = {s: d.loc[common].reset_index() for s, d in data.items()}
    n = len(common)
    start = max(n - a.bars - C.MAX_HOLD_BARS - 1, C.LOOKBACK - 1)
    print(f"{len(data)} coins: {', '.join(s.split('/')[0] for s in data)} | trading bars {start}..{n - 2}")

    oi = {}
    if C.OI_FILTER and a.filters != "off":
        feed = OIFeed()
        tf = C.TF_MINUTES * 60_000
        for s, d in data.items():
            oi[s] = feed.history(s, int(d.ts.iloc[start]) - oi_bars_needed() * tf)
        have = [s.split("/")[0] for s, v in oi.items() if v is not None]
        print(f"open interest from {C.OI_EXCHANGE}: {len(have)}/{len(data)} coins"
              + (f" ({', '.join(have)})" if have else " - OI rules will not fire"))

    model = Forecaster(C.MODEL_NAME, C.TOKENIZER_NAME, C.MAX_CONTEXT, C.TORCH_THREADS, C.SEED)
    steps, preds, gated = [], [], []
    for i in range(start, n - 1):                                # pass 1: forecasts (the slow part)
        win = {s: d.iloc[i - C.LOOKBACK + 1:i + 1] for s, d in data.items()}
        fcs = model.forecast_many(win, C.TF_MINUTES, C.SAMPLE_PATHS, C.TEMPERATURE, C.TOP_P, C.HORIZON_BARS,
                                  C.TOP_K, C.FORECAST_BATCH)
        sigs = {s: signal_side(fc) for s, fc in fcs.items()}
        ctxs = {s: context(win[s], oi.get(s)) for s, v in sigs.items() if v and allowed(v)}
        steps.append((i, fcs, sigs, ctxs))
        for s, fc in fcs.items():
            real = data[s].close[i + 1] / fc.last_close - 1
            preds.append((s, fc.exp_return, real, fc.up_share))
            if s in ctxs:
                why = entry_block(sigs[s], ctxs[s])
                gated.append(((why or "passed").split(" (")[0], sigs[s] * real))
        if (i - start) % 20 == 0:
            print(f"  forecast bar {i - start}/{n - 1 - start}")

    p = pd.DataFrame(preds, columns=["symbol", "fc", "real", "up"])
    print("\n=== forecast quality (next close) ===")
    for s, g in p.groupby("symbol"):
        print(f"{s:22s} hit {(np.sign(g.fc) == np.sign(g.real)).mean():.1%}  corr {g.fc.corr(g.real):+.3f}")
    print(f"{'ALL':22s} hit {(np.sign(p.fc) == np.sign(p.real)).mean():.1%}  corr {p.fc.corr(p.real):+.3f}")

    if gated:
        g = pd.DataFrame(gated, columns=["rule", "move"])
        g["rule"] = g.rule.str.replace(r"[-+]?\d+(\.\d+)?%?", "#", regex=True)
        print("\n=== what the filters did to Kronos signals (next-candle move in the signal's direction) ===")
        print(g.groupby("rule").move.agg(count="count", avg_move="mean",
                                         right=lambda m: (m > 0).mean()).to_string(
            formatters={"avg_move": "{:+.3%}".format, "right": "{:.0%}".format}))
        print("A rule helps if its blocked signals have a worse avg_move than 'passed'.")

    fmodes = {"both": ["off", "on"], "on": ["on"], "off": ["off"]}[a.filters]
    emodes = ["limit", "market", "limit_then_market"] if a.entry == "all" else [a.entry]
    fine = {}
    if "limit_then_market" in emodes and C.TF_MINUTES > 5:
        k = (n - start + 3) * C.TF_MINUTES // 5 + 12
        for s in data:
            try:
                f = br.candles(s, "5m", k)
                f["h"] = f.ts // (C.TF_MINUTES * 60_000) * (C.TF_MINUTES * 60_000)
                fine[s] = {h: g.reset_index(drop=True) for h, g in f.groupby("h")}
            except Exception as e:
                print(f"  5m candles {s}: {e} (limit_then_market falls back to a market entry at the open)")

    results = {(e, f): simulate(steps, data, fine, a, f == "on", e) for e in emodes for f in fmodes}

    print(f"\n=== strategy (stops: {C.STOP_ORDER_TYPE}, fees maker {C.MAKER_FEE:.2%} / taker {C.TAKER_FEE:.2%}, "
          f"slippage {a.spread:.2%}/side) ===")
    print(f"{'entry':18s} {'filters':7s} {'signals':>7s} {'filled':>6s} {'trades':>6s} {'win':>5s} "
          f"{'avg':>8s} {'pnl USDT':>9s} {'max DD':>8s}")
    for (e, f), (t, placed) in results.items():
        if t.empty:
            print(f"{e:18s} {f:7s} {placed:7d} {'0%':>6s} {0:6d}")
            continue
        eq = t.pnl.cumsum()
        print(f"{e:18s} {f:7s} {placed:7d} {len(t) / max(placed, 1):6.0%} {len(t):6d} {(t.ret > 0).mean():5.0%} "
              f"{t.ret.mean():+8.3%} {t.pnl.sum():+9.2f} {(eq - eq.cummax()).min():8.2f}")
        t.to_csv(f"backtest_trades_{e}_{f}.csv", index=False)
    print("(signals = entries attempted, filled = share that became trades; trades -> backtest_trades_<entry>_<filters>.csv)")

    key = (C.ENTRY_MODE if C.ENTRY_MODE in emodes else emodes[0], "on" if "on" in fmodes else "off")
    t = results[key][0]
    if not t.empty:
        print(f"\n--- your settings: entry {key[0]}, filters {key[1]} ---")
        print(t.groupby(["side", "how", "why"]).ret.agg(["count", "mean"]).to_string())
        print(t.groupby("symbol").pnl.agg(["count", "sum"]).sort_values("sum").to_string())
    if all(r[0].empty for r in results.values()):
        print("no trades - loosen MIN_EDGE_* / MIN_AGREEMENT or ENTRY_PULLBACK")


def exit_check(p, o, h, l, c, spread, force=False):
    """(price, reason, exit fee) if the position closes in this candle, else None.
    Stop is assumed hit before TP when both are touched."""
    sd = p["side"]
    soft = p["entry"] * (1 - sd * C.STOP_PCT)
    if (l <= soft) if sd == LONG else (h >= soft):
        if C.STOP_ORDER_TYPE == "market":              # fills near the trigger, or at the open on a gap
            gap = (o <= soft) if sd == LONG else (o >= soft)
            return (o if gap else soft) * (1 - sd * spread), "stop", C.TAKER_FEE
        return (min(c, soft) if sd == LONG else max(c, soft)) * (1 - sd * spread), "stop", C.MAKER_FEE
    if (h > p["target"]) if sd == LONG else (l < p["target"]):
        return p["target"], "tp", C.MAKER_FEE
    if force:
        return c * (1 - sd * spread), "time/flip", C.MAKER_FEE
    return None


def simulate(steps, data, fine, a, use_filters, mode):
    """Pass 2: trade the stored forecasts with the bot's rules. Returns (trades, entries attempted).
    limit: fills only if the next candle trades through it; exits start the candle after.
    market: fills at the next open + slippage; that whole candle counts for exits.
    limit_then_market: the limit gets the first 5m candle of the next hour (the bot waits
    ENTRY_CHASE_SECONDS, 2 min by default, so this is generous to the limit); otherwise market at
    that 5m close if the TP still clears taker fees. Exits are then checked on the rest of the hour's 5m candles."""
    trades, pos, pend, cool, placed = [], {}, {}, {}, 0
    tf = C.TF_MINUTES * 60_000

    def close(s, p, px, why, fee_out, ts):
        sd = p["side"]
        r = sd * (px - p["entry"]) / p["entry"] - p["fee_in"] - fee_out
        trades.append(dict(symbol=s, side=NAME[sd], how=p["how"], open=p["opened"], close=ts, ret=r, why=why,
                           pnl=r * C.NOTIONAL_USDT))
        if why == "stop":
            cool[s] = i + 1 + C.COOLDOWN_BARS
        del pos[s]

    for i, fcs, sigs, ctxs in steps:
        for s, p in pos.items():
            p["bars"] += 1
            if (C.EXIT_ON_FLIP and sigs.get(s) == -p["side"]) or p["bars"] >= C.MAX_HOLD_BARS:
                p["force"] = True
        slots = C.MAX_POSITIONS - len(pos) - len(pend)
        same = {1: 0, -1: 0}
        for p in list(pos.values()) + list(pend.values()):
            same[p["side"]] += 1
        cands = sorted(((score(fcs[s]), s, v) for s, v in sigs.items()
                        if v and allowed(v) and s not in pos and s not in pend and i >= cool.get(s, -1)
                        and not (use_filters and entry_block(v, ctxs.get(s)))),
                       reverse=True)
        for sc, s, side in cands:
            if slots <= 0:
                break
            if C.MAX_SAME_SIDE and same[side] >= C.MAX_SAME_SIDE:
                continue
            last = fcs[s].last_close
            e, t = plan_entry(fcs[s], side, last * (1 - a.spread / 2), last * (1 + a.spread / 2), last * 1e-7,
                              market=mode == "market")
            if e:
                pend[s] = dict(side=side, entry=e, target=t, placed=i)
                slots -= 1; same[side] += 1; placed += 1

        for s in list(pend):                                     # candle i+1: entries
            c, p = data[s].iloc[i + 1], pend[s]
            sd = p["side"]
            new = None
            sub = fine.get(s, {}).get(int(c.ts)) if mode == "limit_then_market" else None
            if mode == "limit":
                if (sd == LONG and c.low < p["entry"]) or (sd != LONG and c.high > p["entry"]):
                    new = dict(entry=p["entry"], fee_in=C.MAKER_FEE, how="limit", rest=None)
                elif i + 1 - p["placed"] < C.ENTRY_TTL_BARS:
                    continue
            elif mode == "limit_then_market" and sub is not None and len(sub):
                f0 = sub.iloc[0]
                if (sd == LONG and f0.low < p["entry"]) or (sd != LONG and f0.high > p["entry"]):
                    new = dict(entry=p["entry"], fee_in=C.MAKER_FEE, how="limit", rest=sub.iloc[1:])
                else:
                    px = f0.close * (1 + sd * a.spread)
                    if worth_it(sd, px, p["target"], market=True):
                        new = dict(entry=px, fee_in=C.TAKER_FEE, how="chased", rest=sub.iloc[1:])
            else:                                                # market, or no 5m data
                px = c.open * (1 + sd * a.spread)
                if worth_it(sd, px, p["target"], market=True):
                    new = dict(entry=px, fee_in=C.TAKER_FEE, how="market", rest="all")
            del pend[s]
            if new:
                pos[s] = dict(side=sd, target=p["target"], bars=0, force=False, opened=c.timestamp, **new)

        for s in list(pos):                                      # candle i+1: exits
            c, p = data[s].iloc[i + 1], pos[s]
            if p["opened"] == c.timestamp:
                rest = p.pop("rest", None)
                if isinstance(rest, pd.DataFrame):               # rest of the entry hour, 5m by 5m
                    for f in rest.itertuples():
                        hit = exit_check(p, f.open, f.high, f.low, f.close, a.spread)
                        if hit:
                            close(s, p, *hit, c.timestamp)
                            break
                    continue
                if rest != "all":                                # limit fill: timing unknown, skip
                    continue
            hit = exit_check(p, c.open, c.high, c.low, c.close, a.spread, p["force"])
            if hit:
                close(s, p, *hit, c.timestamp)
    return pd.DataFrame(trades), placed


if __name__ == "__main__":
    main()
