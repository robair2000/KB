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


def decide(exp_ret, entry_threshold=0.008, exit_threshold=0.0):
    """ENTER if forecast gain beats fees+margin, EXIT if forecast turns negative, else HOLD."""
    if exp_ret >= entry_threshold:
        return "ENTER"
    if exp_ret <= exit_threshold:
        return "EXIT"
    return "HOLD"
