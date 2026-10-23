"""Learn from the bot's own decisions.

  1. JOURNAL  every candle, every coin: forecast, filter readings (RSI / EMA / momentum / OI)
              and what the bot did (opened, blocked + reason, skipped + reason, nothing).
  2. LABELS   once a signal's holding window has passed, its outcome is computed from the candles
              that followed, with the bot's own take-profit / stop / max-hold / fees. Blocked and
              skipped signals are labelled too, so you learn whether the filters actually help.
  3. TRAINING every RETRAIN_HOURS a small regression model learns the EXPECTED RETURN of a signal
              (net of fees) from its readings. Expected return, not win rate: with small take-profits
              and wider stops most trades "win", so what matters is how big the losses are.
              It is tested on the most recent data it never saw and only used if the signals it
              keeps did better there than taking every signal.
  4. FILTER   LEARNED_FILTER=shadow (default): log the model's expected return, block nothing.
              LEARNED_FILTER=on: also skip signals whose expected return is below LEARNED_MIN_EDGE.
              LEARNED_FILTER=off: no model at all (journal + labels still run while LEARN=true).

Files (in LEARN_DIR, default = folder of the bot's state file, i.e. the Render disk):
  signals.jsonl  one line per coin per candle      labels.jsonl  outcome per signal
  pending.json   signals still waiting for outcome model.json    current model + its test scores
  report.json    last training report (also logged)

Report from a shell:  python learner.py report /var/data
Only numpy is needed; this file is identical in the KB and KB15 repos.
"""
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np


def _b(name, default):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


LEARN = _b("LEARN", "true")
LEARNED_FILTER = os.getenv("LEARNED_FILTER", "shadow").strip().lower()      # off | shadow | on
LEARNED_MIN_EDGE = float(os.getenv("LEARNED_MIN_EDGE", "0.0"))     # min expected net return to trade
RETRAIN_HOURS = float(os.getenv("RETRAIN_HOURS", "168"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "500"))
TRAIN_DAYS = int(os.getenv("TRAIN_DAYS", "60"))
LABEL_GIVE_UP_HOURS = 72

FEATURES = ["edge", "agree", "disp", "rsi_s", "ema_s", "mom_s", "oi_chg", "oi_flow", "spike", "is_long",
            "hour_sin", "hour_cos"]


def _log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] learner | {msg}", flush=True)


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def make_features(side, er, ts_ms, ctx=None, agree=None, disp=None, mom=None):
    """Signal readings, turned so that positive always means 'in favour of this trade side'.
    side: +1 long / -1 short. er: forecast return. ctx: the bot's filter readings (any missing -> None)."""
    ctx = ctx or {}
    s = side or 1
    close, ema_v, rsi_v = _num(ctx.get("close")), _num(ctx.get("ema")), _num(ctx.get("rsi"))
    px, oi = _num(ctx.get("px_chg")), _num(ctx.get("oi_chg"))
    mom = _num(mom) if mom is not None else _num(ctx.get("mom_chg"))
    mom = mom if mom is not None else px
    hour = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).hour
    return {
        "edge": s * er if er is not None else None,
        "agree": (agree if s == 1 else 1 - agree) if agree is not None else None,
        "disp": _num(disp),
        "rsi_s": s * (rsi_v - 50) / 50 if rsi_v is not None else None,
        "ema_s": s * (close / ema_v - 1) if close and ema_v else None,
        "mom_s": s * mom if mom is not None else None,
        "oi_chg": oi,
        "oi_flow": s * oi * (1 if (px or 0) >= 0 else -1) if oi is not None else None,
        "spike": 1.0 if ctx.get("oi_spike") else 0.0,
        "is_long": 1.0 if s == 1 else 0.0,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }


class Learner:
    def __init__(self, bot_name, data_dir, tf_seconds, clock=time.time):
        self.bot, self.dir, self.tf_ms, self.clock = bot_name, data_dir or ".", int(tf_seconds) * 1000, clock
        os.makedirs(self.dir, exist_ok=True)
        self.f_sig = os.path.join(self.dir, "signals.jsonl")
        self.f_lab = os.path.join(self.dir, "labels.jsonl")
        self.f_pend = os.path.join(self.dir, "pending.json")
        self.f_model = os.path.join(self.dir, "model.json")
        self.f_report = os.path.join(self.dir, "report.json")
        self.pending = self._read_json(self.f_pend, [])
        self.model = self._read_json(self.f_model, None) if LEARNED_FILTER != "off" else None
        if self.model and self.model.get("kind") != "ridge":
            self.model = None                                  # model from an older learner version
        self._model_mtime = self._mtime(self.f_model)
        if LEARN:
            m = self.model
            _log(f"journal on in {self.dir} | {len(self.pending)} signals awaiting outcome | filter {LEARNED_FILTER}"
                 + (f" | model from {m['trained_at']} (n={m['n_train']}, kept signals "
                       f"{m['test']['kept_ret']:+.2%} vs all {m['test']['base_ret']:+.2%})" if m else " | no model yet"))

    # ----------------------------------------------------------- io helpers --
    @staticmethod
    def _read_json(path, default):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return default

    @staticmethod
    def _write_json(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)

    @staticmethod
    def _mtime(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0

    def _append(self, path, rows):
        with open(path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    # --------------------------------------------------------------- journal --
    def record(self, rows):
        """rows: dicts with sym, t (candle close ms), close, side (-1/0/1), er, tp_pct, sl_pct, hold,
        fee, action, feats, p. Signals with a side are queued for labelling."""
        if not LEARN or not rows:
            return
        out = []
        for r in rows:
            r = dict(r, bot=self.bot, id=f"{self.bot}:{r['sym']}:{r['t']}")
            out.append(r)
            if r.get("side"):
                self.pending.append({k: r[k] for k in ("id", "sym", "t", "close", "side", "tp_pct", "sl_pct",
                                                       "hold", "fee")})
        self._append(self.f_sig, out)
        self._write_json(self.f_pend, self.pending)

    # ---------------------------------------------------------------- labels --
    def label(self, candles_by_sym, ts_of, fetch=None, max_fetch=10):
        """Fill in outcomes whose holding window has fully passed.
        candles_by_sym: frames already loaded this cycle. ts_of(frame) -> np.array of candle OPEN ms.
        fetch(sym) -> frame, for coins not loaded this cycle (at most max_fetch per call)."""
        if not LEARN or not self.pending:
            return 0
        now = self.clock() * 1000
        done, keep, fetched = [], [], 0
        cache = dict(candles_by_sym)
        for p in self.pending:
            if now < p["t"] + p["hold"] * self.tf_ms + 60_000:
                keep.append(p)
                continue
            c = cache.get(p["sym"])
            if c is None and fetch and fetched < max_fetch:
                fetched += 1
                try:
                    c = cache[p["sym"]] = fetch(p["sym"])
                except Exception:
                    c = cache[p["sym"]] = None
            res = self._outcome(p, c, ts_of) if c is not None else None
            if res is None:
                if now - p["t"] > LABEL_GIVE_UP_HOURS * 3_600_000:
                    done.append({"id": p["id"], "outcome": "nodata"})
                else:
                    keep.append(p)
                continue
            done.append(dict(id=p["id"], **res))
        if done:
            self._append(self.f_lab, done)
            self.pending = keep
            self._write_json(self.f_pend, keep)
        return len(done)

    def _outcome(self, p, c, ts_of):
        ts = ts_of(c)
        i0 = int(np.searchsorted(ts, p["t"]))                 # first candle opening at/after the signal
        bars = c.iloc[i0:i0 + p["hold"]]
        if len(bars) < p["hold"] or (len(ts) > i0 and ts[i0] != p["t"]):
            return None
        e, s = p["close"], p["side"]
        tp, sl = e * (1 + s * p["tp_pct"]), e * (1 - s * p["sl_pct"])
        for k, (_, b) in enumerate(bars.iterrows()):
            hi, lo = float(b["high"]), float(b["low"])
            if (s == 1 and lo <= sl) or (s == -1 and hi >= sl):              # stop first if both touched
                return {"outcome": "stop", "ret": -p["sl_pct"] - p["fee"], "bars": k + 1}
            if (s == 1 and hi >= tp) or (s == -1 and lo <= tp):
                return {"outcome": "tp", "ret": p["tp_pct"] - p["fee"], "bars": k + 1}
        r = s * (float(bars["close"].iloc[-1]) / e - 1) - p["fee"]
        return {"outcome": "time", "ret": r, "bars": len(bars)}

    # ----------------------------------------------------------------- model --
    def _refresh_model(self):
        m = self._mtime(self.f_model)
        if m != self._model_mtime:
            self._model_mtime, self.model = m, self._read_json(self.f_model, None)

    def predict(self, feats):
        """Model's expected net return for a signal, or None when there is no accepted model."""
        if LEARNED_FILTER == "off":
            return None
        self._refresh_model()
        m = self.model
        if not m or m.get("kind") != "ridge":
            return None
        x = np.array([_num(feats.get(f)) if _num(feats.get(f)) is not None else np.nan for f in m["features"]])
        x = np.nan_to_num((x - np.array(m["mean"])) / np.array(m["std"]))
        return float(m["bias"] + x @ np.array(m["w"]))

    def gate(self, feats):
        """(block_reason or None, expected return). Blocks only with LEARNED_FILTER=on and an accepted model."""
        e = self.predict(feats)
        if LEARNED_FILTER == "on" and e is not None and e < LEARNED_MIN_EDGE:
            return f"learned filter: expected return {e:+.2%} < {LEARNED_MIN_EDGE:+.2%}", e
        return None, e

    # -------------------------------------------------------------- training --
    def _dataset(self):
        since = (self.clock() - TRAIN_DAYS * 86400) * 1000
        labels = {}
        if os.path.exists(self.f_lab):
            with open(self.f_lab) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("outcome") in ("tp", "stop", "time"):
                        labels[r["id"]] = r
        rows = []
        if os.path.exists(self.f_sig):
            with open(self.f_sig) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("t", 0) < since or not r.get("side") or r["id"] not in labels:
                        continue
                    rows.append((r, labels[r["id"]]))
        rows.sort(key=lambda x: x[0]["t"])
        return rows

    def maybe_retrain(self, force=False):
        if not LEARN or LEARNED_FILTER == "off":
            return None
        last = self._read_json(self.f_report, {}).get("at_ts", 0)
        if not force and self.clock() - last < RETRAIN_HOURS * 3600:
            return None
        rep = train(self._dataset())
        rep["at_ts"] = self.clock()
        rep["at"] = datetime.fromtimestamp(rep["at_ts"], timezone.utc).isoformat(timespec="minutes")
        self._write_json(self.f_report, rep)
        _log(rep["summary"])
        if rep.get("accepted"):
            self._write_json(self.f_model, rep["model"])
            self._refresh_model()
        return rep


# ---------------------------------------------------------------- training --
def _ridge(Z, y, lam):
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w, float(ym)


def _rank_corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def train(rows, min_edge=None):
    """rows: [(signal, label)] oldest first. Fit on the older 80%, judge on the newest 20%."""
    min_edge = LEARNED_MIN_EDGE if min_edge is None else min_edge
    n = len(rows)
    if n < MIN_TRAIN_SAMPLES:
        return {"accepted": False, "n": n,
                "summary": f"training skipped: {n}/{MIN_TRAIN_SAMPLES} labelled signals so far"}
    X = np.array([[_num(r["feats"].get(f)) if _num(r["feats"].get(f)) is not None else np.nan
                   for f in FEATURES] for r, _ in rows], dtype=float)
    ret = np.array([l["ret"] for _, l in rows])
    y = np.clip(ret, -0.05, 0.05)                                  # one freak candle shouldn't steer the fit
    cut = int(n * 0.8)
    with warnings.catch_warnings():                  # columns a bot never fills (e.g. KB has no agreement)
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X[:cut], axis=0)
        std = np.nanstd(X[:cut], axis=0)
    mean = np.where(np.isnan(mean), 0, mean)
    std = np.where((std > 0) & ~np.isnan(std), std, 1)
    Z = np.nan_to_num((X - mean) / std)
    # shrinkage chosen on the last quarter of the training part (still older than the test part)
    v = int(cut * 0.75)
    best = max((_rank_corr(Z[v:cut] @ _ridge(Z[:v], y[:v], lam)[0], ret[v:cut]), lam)
               for lam in (1.0, 10.0, 100.0, 1000.0))
    lam = best[1]
    w, b = _ridge(Z[:cut], y[:cut], lam)
    e_te, r_te = Z[cut:] @ w + b, ret[cut:]
    sel = e_te >= min_edge
    test = {"n": int(len(r_te)), "rank_corr": _rank_corr(e_te, r_te), "base_win": float((r_te > 0).mean()),
            "base_ret": float(r_te.mean()), "kept": int(sel.sum()),
            "kept_win": float((r_te[sel] > 0).mean()) if sel.any() else None,
            "kept_ret": float(r_te[sel].mean()) if sel.any() else None}
    accepted = (test["n"] >= 100 and test["rank_corr"] >= 0.05 and 20 <= test["kept"] < test["n"]
                and test["kept_ret"] is not None and test["kept_ret"] > test["base_ret"] + 0.0002)
    model = {"kind": "ridge", "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "w": w.tolist(),
             "bias": b, "lambda": lam, "n_train": cut, "test": test,
             "trained_at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    top = sorted(zip(FEATURES, w), key=lambda kv: -abs(kv[1]))[:4]
    summary = (f"trained on {cut}, tested on {test['n']} newest signals: rank corr {test['rank_corr']:+.2f} | all: "
               f"win {test['base_win']:.0%} avg {test['base_ret']:+.2%} | expected >= {min_edge:+.2%}: "
               f"{test['kept']} kept"
               + (f", win {test['kept_win']:.0%} avg {test['kept_ret']:+.2%}" if test["kept"] else "")
               + f" | strongest: {', '.join(f'{k} {v * 100:+.2f}%' for k, v in top)} | "
               + ("ACCEPTED" if accepted else "not better than taking every signal - previous model kept"))
    return {"accepted": accepted, "n": n, "test": test, "model": model, "summary": summary}


# ------------------------------------------------------------------ report --
def report(data_dir):
    """How did each kind of decision turn out? Answers 'do my filters help?'."""
    lab = {}
    with open(os.path.join(data_dir, "labels.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("outcome") in ("tp", "stop", "time"):
                lab[r["id"]] = r
    groups = {}
    with open(os.path.join(data_dir, "signals.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in lab and r.get("side"):
                key = r["action"].split(":")[0] + (":" + r["action"].split(":")[1].strip().split(" ")[0]
                                                   if r["action"].startswith("blocked") else "")
                groups.setdefault(key, []).append(lab[r["id"]]["ret"])
    print(f"{'decision':32s} {'n':>6s} {'win':>6s} {'avg ret':>9s}")
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        print(f"{k:32s} {len(v):6d} {(v > 0).mean():6.0%} {v.mean():+9.2%}")
    rep = os.path.join(data_dir, "report.json")
    if os.path.exists(rep):
        print("\nlast training:", json.load(open(rep))["summary"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        print("usage: python learner.py report <data dir>")
