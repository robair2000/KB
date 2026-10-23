"""Kronos forecasting -> trading signal. Spot only (long or flat)."""
import os

import pandas as pd

# model name -> (tokenizer, max_context)
MODELS = {
    "mini": ("NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-Tokenizer-base", 512),
}


def load_predictor(name=None, device=None):
    """Load the official Kronos tokenizer + model from Hugging Face."""
    from model import Kronos, KronosPredictor, KronosTokenizer  # local ./model folder
    name = name or os.getenv("KRONOS_MODEL", "small")
    tok_id, max_ctx = MODELS[name]
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model = Kronos.from_pretrained(f"NeoQuasar/Kronos-{name}")
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)


def expected_return(predictor, candles, pred_len=6, sample_count=3, lookback=400):
    """Forecast `pred_len` future candles; return expected % move of the close.

    `candles` must contain only CLOSED candles, oldest -> newest.
    """
    x = candles.tail(lookback).reset_index(drop=True)
    step = x["timestamps"].diff().dropna().median()
    last_ts = x["timestamps"].iloc[-1]
    y_ts = pd.Series(pd.date_range(last_ts + step, periods=pred_len, freq=step))
    pred = predictor.predict(
        df=x[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=x["timestamps"],
        y_timestamp=y_ts,
        pred_len=pred_len, T=1.0, top_p=0.9, sample_count=sample_count, verbose=False)
    last_close = float(x["close"].iloc[-1])
    return float(pred["close"].iloc[-1]) / last_close - 1.0


def expected_returns(predictor, candles_by_sym, pred_len=6, sample_count=3, lookback=400, batch=None):
    """Forecast many coins at once with Kronos predict_batch -> {symbol: expected % move}.

    Same result as calling expected_return() per coin, but all coins share one model pass,
    so every decision is ready a few minutes after the candle close instead of coin by coin.
    Coins are grouped by history length (predict_batch needs equal lengths).
    """
    batch = batch or int(os.getenv("FORECAST_BATCH", "16"))
    prepared = {}
    for sym, candles in candles_by_sym.items():
        x = candles.tail(lookback).reset_index(drop=True)
        step = x["timestamps"].diff().dropna().median()
        y_ts = pd.Series(pd.date_range(x["timestamps"].iloc[-1] + step, periods=pred_len, freq=step))
        prepared[sym] = (x, y_ts)
    groups = {}
    for sym, (x, _) in prepared.items():
        groups.setdefault(len(x), []).append(sym)
    out = {}
    for syms in groups.values():
        for i in range(0, len(syms), batch):
            chunk = syms[i:i + batch]
            preds = predictor.predict_batch(
                df_list=[prepared[s][0][["open", "high", "low", "close", "volume", "amount"]] for s in chunk],
                x_timestamp_list=[prepared[s][0]["timestamps"] for s in chunk],
                y_timestamp_list=[prepared[s][1] for s in chunk],
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=sample_count, verbose=False)
            for s, p in zip(chunk, preds):
                out[s] = float(p["close"].iloc[-1]) / float(prepared[s][0]["close"].iloc[-1]) - 1.0
    return out


def decide(exp_ret, entry_threshold=0.008, exit_threshold=0.0):
    """ENTER if forecast gain beats fees+margin, EXIT if forecast turns negative, else HOLD."""
    if exp_ret >= entry_threshold:
        return "ENTER"
    if exp_ret <= exit_threshold:
        return "EXIT"
    return "HOLD"
