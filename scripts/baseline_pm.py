"""Polymarket baselines: no-change and LightGBM (two-stage: zero-move classifier + quantile regressors on hand-crafted features).
All fitted on the train split only. Writes prediction JSONL in the eval contract for the frozen eval sets (val / test_id / test_ood,
all horizons, Task J and Task S) and for the intervention prompts (Task S, 1h), plus out/eval/pm_baselines_r2.json (alarm input:
out-of-sample R^2 of E[dmid_1h] on the full test_id split, Task J) and out/eval/pm_baselines.md.
  data/processed/polymarket/preds/{nochange,lgbm}.jsonl, {name}_interventions.jsonl
"""
import sys, os, json, time, argparse
import numpy as np
import polars as pl
import lightgbm as lgb

from eventworld.pm_state import ROOT, DELTAS
from eventworld import metrics_pm as M

PRED = f"{ROOT}/preds"
TOPICS = ["sports", "politics", "crypto", "macro", "entertainment", "other", "weather"]
BASE_FEATS = ["mid", "spread", "logD", "N1h", "F1h_rel", "logV1h", "age_d", "ttl_d", "ttl_missing", "log_since", "is_mid_band", "is_anchor", "d", "logq", "qD", "w_pos_rel", "w_n_prev", "w_flow_rel"] + [f"topic_{t}" for t in TOPICS]
J_FEATS = ["N_all", "F_all_rel", "logV_all", "own_N", "own_F_rel", "logown_V"]
S_FEATS = ["own_N", "own_F_rel", "logown_V", "rho"]


def features(df):
    D1 = pl.col("D") + 1.0
    return df.with_columns([
        pl.col("D").log1p().alias("logD"), (pl.col("F1h") / D1).alias("F1h_rel"), pl.col("V1h").log1p().alias("logV1h"),
        pl.col("ttl_d").fill_nan(None).fill_null(-1.0).alias("ttl_d"), pl.col("ttl_d").fill_nan(None).is_null().cast(pl.Float64).alias("ttl_missing"),
        pl.col("since_last_s").clip(0).cast(pl.Float64).log1p().alias("log_since"), (pl.col("band") == "mid").cast(pl.Float64).alias("is_mid_band"),
        (pl.col("kind") == "anchor").cast(pl.Float64).alias("is_anchor"), pl.col("d").cast(pl.Float64), pl.col("q").log1p().alias("logq"),
        (pl.col("w_pos").fill_nan(0.0).fill_null(0.0) / (pl.col("D") + 100.0)).alias("w_pos_rel"), pl.col("w_n_prev").clip(0).cast(pl.Float64),
        (pl.col("w_flow").fill_nan(0.0).fill_null(0.0) / D1).alias("w_flow_rel"),
        (pl.col("F_all") / D1).alias("F_all_rel"), pl.col("V_all").log1p().alias("logV_all"), (pl.col("own_F") / D1).alias("own_F_rel"), pl.col("own_V").log1p().alias("logown_V"),
        pl.col("rho").fill_nan(0.0).fill_null(0.0), pl.col("N_all").cast(pl.Float64), pl.col("own_N").cast(pl.Float64), pl.col("N1h").cast(pl.Float64),
    ] + [(pl.col("topic") == t).cast(pl.Float64).alias(f"topic_{t}") for t in TOPICS])


def load(delta):
    pts = pl.read_parquet(f"{ROOT}/points.parquet")
    sp = pl.read_parquet(f"{ROOT}/splits.parquet").select("point_id", "split")
    tg = pl.read_parquet(f"{ROOT}/targets_{delta}.parquet").drop("own")
    return features(pts.join(sp, on="point_id", how="inner").join(tg, on="point_id", how="inner"))


def X(df, task):
    cols = BASE_FEATS + (J_FEATS if task == "J" else S_FEATS)
    return np.nan_to_num(df.select(cols).to_numpy().astype(np.float64)), cols


class NoChange:
    def fit(self, tr, task):
        self.pz = tr.group_by("kind", "band", "rho_bin").agg(pl.col("is_zero").mean().alias("pz_hat"))
        self.oth = tr.group_by("kind").agg(pl.col("N_o").mean().alias("N_hat"), pl.col("F_o").mean().alias("F_hat"), pl.col("V_o").mean().alias("V_hat"))
        return self

    def predict(self, te, task):
        j = te.join(self.pz, on=["kind", "band", "rho_bin"], how="left").join(self.oth, on="kind", how="left")
        n = te.height
        return {"q50": np.zeros(n), "q10": np.zeros(n), "q90": np.zeros(n), "p_zero": j["pz_hat"].fill_null(0.5).to_numpy(), "N": j["N_hat"].to_numpy(), "F": j["F_hat"].to_numpy(), "V": j["V_hat"].to_numpy()}


class LGBM:
    P = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=32)

    def fit(self, tr, task):
        x, _ = X(tr, task); y = tr["dmid"].to_numpy(); z = tr["is_zero"].to_numpy()
        self.clf = lgb.LGBMClassifier(**self.P).fit(x, z)
        self.q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, **self.P).fit(x[~z], y[~z]) for a in [0.1, 0.5, 0.9]}
        self.oth = {k: lgb.LGBMRegressor(objective="l1", **self.P).fit(x, tr[f"{k}_o"].to_numpy()) for k in ["N", "F", "V"]}
        return self

    def predict(self, te, task):
        x = X(te, task)[0]
        q = np.sort(np.stack([self.q[a].predict(x) for a in [0.1, 0.5, 0.9]], 1), axis=1)
        return {"q50": q[:, 1], "q10": q[:, 0], "q90": q[:, 2], "p_zero": self.clf.predict_proba(x)[:, 1], **{k: self.oth[k].predict(x) for k in ["N", "F", "V"]}}


def make(name, delta):
    return {"nochange": NoChange, "lgbm": LGBM}[name]()


def write_preds(fh, df, task, p, sample_ids, variant=None):
    pb = df["p_buy"].to_numpy(); ps = df["p_sell"].to_numpy()
    for i, sid in enumerate(sample_ids):
        pred = {"p_buy": float(np.clip(pb[i] + p["q50"][i], 0, 1)), "p_sell": float(np.clip(ps[i] + p["q50"][i], 0, 1)), "p_zero": float(np.clip(p["p_zero"][i], 0, 1)),
                "q10": float(p["q10"][i]), "q50": float(p["q50"][i]), "q90": float(p["q90"][i])}
        if task == "S":
            pred["others"] = {"N": float(max(p["N"][i], 0)), "F": float(p["F"][i]), "V": float(max(p["V"][i], 0))}
        fh.write(json.dumps({"sample_id": sid, "task": task, "variant": variant or "factual", "pred": pred, "samples": [], "logprob": None}) + "\n")


def intervention_frame(base, variant):
    """Apply the intervention to the own-action features of Task S rows (base = factual 1h rows of intervention points)."""
    if variant in ("factual", "null"):
        return base
    if variant == "remove":
        return base.with_columns(pl.lit(0).alias("own_N"), pl.lit(0.0).alias("own_F"), pl.lit(0.0).alias("own_V"), pl.lit(0.0).alias("rho"), pl.lit("control").alias("kind"), pl.lit(0).cast(pl.Int8).alias("d"), pl.lit(0.0).alias("q"), pl.lit(0.0).alias("qD"), pl.lit("control").alias("rho_bin"))
    if variant == "reverse":
        return base.with_columns((-pl.col("own_F")).alias("own_F"), (-pl.col("d")).alias("d"))
    if variant == "scale":
        return base.with_columns((pl.col("own_F") / 10).alias("own_F"), (pl.col("own_V") / 10).alias("own_V"), (pl.col("rho") / 10).alias("rho"), (pl.col("q") / 10).alias("q"), (pl.col("qD") / 10).alias("qD"),
                                 pl.when(pl.col("rho") / 10 < 0.2).then(pl.lit("0.1-0.2")).when(pl.col("rho") / 10 < 0.5).then(pl.lit("0.2-0.5")).when(pl.col("rho") / 10 < 1).then(pl.lit("0.5-1")).otherwise(pl.lit(">=1")).alias("rho_bin"))
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--names", default="nochange,lgbm"); a = ap.parse_args()
    t0 = time.time(); os.makedirs(PRED, exist_ok=True)
    idx = pl.read_parquet(f"{ROOT}/eval/eval_index.parquet").filter(pl.col("split").is_in(["val", "test_id", "test_ood"])).select("sample_id", "point_id", "task", "delta", "split")
    inter = pl.read_parquet(f"{ROOT}/eval/interventions.parquet").select("sample_id", "point_id", "variant", "split")
    r2 = {}; lines = ["# Polymarket floor baselines", ""]
    fhs = {n: (open(f"{PRED}/{n}.jsonl", "w"), open(f"{PRED}/{n}_interventions.jsonl", "w")) for n in a.names.split(",")}
    for delta in DELTAS:
        df = load(delta); tr = df.filter(pl.col("split") == "train"); te_full = df.filter(pl.col("split") == "test_id")
        print(f"[{delta}] train rows {tr.height:,}, test_id rows {te_full.height:,}  ({time.time() - t0:.0f}s)")
        for name in a.names.split(","):
            for task in ["J", "S"]:
                t1 = time.time(); model = make(name, delta).fit(tr, task)
                ev = idx.filter((pl.col("delta") == delta) & (pl.col("task") == task)).join(df, on="point_id", how="inner")
                p = model.predict(ev, task); write_preds(fhs[name][0], ev, task, p, ev["sample_id"].to_list())
                if task == "J":
                    pf = model.predict(te_full, "J"); e = (1 - pf["p_zero"]) * pf["q50"]
                    r2[f"{name}|{delta}"] = M.r2(e, te_full["dmid"].to_numpy())
                    lines.append(f"- {name} {delta} Task J: full test_id R^2(E[dmid]) = {r2[f'{name}|{delta}']:.4f}, MAE {np.mean(np.abs(e - te_full['dmid'].to_numpy())):.5f} vs no-change {np.mean(np.abs(te_full['dmid'].to_numpy())):.5f}; fit+predict {time.time() - t1:.0f}s")
                if task == "S" and delta == "1h":
                    base = inter.filter(pl.col("variant") == "factual").select("point_id").join(df, on="point_id", how="inner")
                    for variant in ["factual", "null", "remove", "reverse", "scale"]:
                        fr = intervention_frame(base, variant); pv = model.predict(fr, "S")
                        write_preds(fhs[name][1], fr, "S", pv, [f"{pid}|S|1h|{variant}" for pid in fr["point_id"].to_list()], variant)
                print(f"  {name} {task} done ({time.time() - t1:.0f}s)")
    for f1, f2 in fhs.values():
        f1.close(); f2.close()
    os.makedirs("out/eval", exist_ok=True)
    json.dump({k: v for k, v in r2.items() if k.endswith("|1h")}, open("out/eval/pm_baselines_r2.json", "w"), indent=1)
    lines += ["", f"- alarm threshold: R^2 > 0.15 on dmid_1h; max observed {max(v for k, v in r2.items() if k.endswith('|1h')):.4f}", f"- runtime {time.time() - t0:.0f}s; predictions under `{PRED}/`"]
    open("out/eval/pm_baselines.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
