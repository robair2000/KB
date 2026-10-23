"""Kronos wrapper: N independent forecast paths per coin, many coins per batch.

Prices are rescaled to ~1000 before they reach Kronos. Its normaliser divides by (std + 1e-5),
so a coin trading at 0.000004 (std ~1e-7) would otherwise be flattened to a constant line.
"""
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from fmt import fp

KRONOS_DIR = os.getenv("KRONOS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_repo"))
sys.path.insert(0, KRONOS_DIR)
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402  (from the Kronos repo)

log = logging.getLogger("forecaster")
PRICE = ["open", "high", "low", "close"]


@dataclass
class Forecast:
    last_close: float
    close: float          # mean forecast close at the horizon
    high: float           # mean of each path's highest high over the horizon
    low: float            # mean of each path's lowest low over the horizon
    exp_return: float     # close / last_close - 1
    up_share: float       # share of paths ending above last_close
    close_std: float      # dispersion of path closes, relative to last_close

    def __str__(self):
        return (f"fc {fp(self.close)} ({self.exp_return:+.3%}) range {fp(self.low)}-{fp(self.high)} "
                f"up {self.up_share:.0%} disp {self.close_std:.3%}")


def prepare(candles):
    """-> (scaled df for Kronos, x timestamps, price scale, last close)."""
    last = float(candles["close"].iloc[-1])
    scale = 1000.0 / last
    df = candles[PRICE + ["volume"]].reset_index(drop=True).astype("float64")
    df[PRICE] = df[PRICE] * scale
    vmed = float(df["volume"].median()) or 1.0
    df["volume"] = df["volume"] * (1000.0 / vmed)                  # keep volume in a sane float32 range
    df["amount"] = df["volume"] * df[PRICE].mean(axis=1)
    ts = pd.Series(pd.to_datetime(candles["timestamp"]).reset_index(drop=True))
    return df.astype("float32"), ts, scale, last


class Forecaster:
    def __init__(self, model_name, tokenizer_name, max_context=512, threads=0, seed=-1):
        if threads:
            torch.set_num_threads(threads)
        if seed >= 0:
            torch.manual_seed(seed); np.random.seed(seed)
        log.info("loading %s + %s", model_name, tokenizer_name)
        tok = KronosTokenizer.from_pretrained(tokenizer_name)
        mdl = Kronos.from_pretrained(model_name)
        tok.eval(); mdl.eval()
        self.predictor = KronosPredictor(mdl, tok, device="cpu", max_context=max_context)

    @torch.no_grad()
    def forecast_many(self, candles_by_sym, tf_minutes=15, paths=8, T=1.0, top_p=0.9, horizon=1,
                      top_k=0, batch=32):
        """{symbol: candles} -> {symbol: Forecast}. All candle frames must have the same length."""
        prepped = {s: prepare(c) for s, c in candles_by_sym.items()}
        jobs = [(s, k) for s in prepped for k in range(paths)]
        step = pd.Timedelta(minutes=tf_minutes)
        out = defaultdict(list)
        for i in range(0, len(jobs), max(batch, 1)):
            chunk = jobs[i:i + batch]
            dfs = [prepped[s][0] for s, _ in chunk]
            xts = [prepped[s][1] for s, _ in chunk]
            yts = [pd.Series([prepped[s][1].iloc[-1] + step * (h + 1) for h in range(horizon)]) for s, _ in chunk]
            preds = self.predictor.predict_batch(df_list=dfs, x_timestamp_list=xts, y_timestamp_list=yts,
                                                 pred_len=horizon, T=T, top_k=top_k, top_p=top_p,
                                                 sample_count=1, verbose=False)
            for (s, _), p in zip(chunk, preds):
                sc = prepped[s][2]
                out[s].append((p["close"].iloc[-1] / sc, p["high"].max() / sc, p["low"].min() / sc))

        result = {}
        for s, rows in out.items():
            a = np.array(rows, dtype="float64")
            last = prepped[s][3]
            c = float(a[:, 0].mean())
            result[s] = Forecast(last_close=last, close=c,
                                 high=max(float(a[:, 1].mean()), c, last), low=min(float(a[:, 2].mean()), c, last),
                                 exp_return=c / last - 1, up_share=float((a[:, 0] > last).mean()),
                                 close_std=float(a[:, 0].std() / last))
        return result
