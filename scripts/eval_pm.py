"""Polymarket scorer. Usage:
  python scripts/eval_pm.py --pred <pred.jsonl> [--pred ...] --name <run> [--interventions <pred_int.jsonl>] [--rollout <pred_roll.jsonl>]
Each --pred file holds prediction lines {"sample_id","task","variant","pred":{...}} for samples in eval_{split}_K{0,3}.jsonl (any splits / deltas / tasks mixed;
sample_ids are K-independent, so score K=0 and K=3 predictions under different --name).
Writes out/eval/pm_<name>.md and out/eval/pm_<name>.json.
Reports per (split, task, delta): p_zero Brier / log-loss, MAE / CRPS / direction accuracy on non-zero windows, skill vs no-change, 21-bin MAE,
control p_zero vs base rate, actor PnL-sign accuracy, price/quantile self-consistency, others' N/F/V MAE (task SF only), Err_S - Err_J,
horizon monotonicity; interventions: sign agreement, Spearman monotonicity, magnitude ratio, null spurious rate.
"""
import sys, os, json, math, argparse, time
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import ROOT
from eventworld import metrics_pm as M

KEYS = ["p_buy", "p_sell", "p_zero", "q10", "q50", "q90"]


def load_preds(files):
    rows = []
    for f in files:
        for line in open(f):
            r = json.loads(line); p = r.get("pred") or {}
            ok = all(k in p and p[k] is not None for k in KEYS)
            vals = {k: float(p[k]) for k in KEYS} if ok else {}
            ok = ok and all(math.isfinite(v) for v in vals.values())
            if ok:  # project onto physical ranges (prices/probabilities in [0,1], Delta-mid quantiles in [-1,1], quantiles sorted); in-range predictions are unchanged
                for k in ("p_buy", "p_sell", "p_zero"):
                    vals[k] = min(max(vals[k], 0.0), 1.0)
                vals["q10"], vals["q50"], vals["q90"] = sorted(min(max(vals[k], -1.0), 1.0) for k in ("q10", "q50", "q90"))
            row = {"sample_id": r["sample_id"], "variant": r.get("variant") or "factual", "parsed": ok}
            for k in KEYS:
                row[k] = vals[k] if ok else None
            o = p.get("others") if ok else None
            for k in ["N", "F", "V"]:
                row[f"others_{k}"] = float(o[k]) if (o and k in o) else None
            rows.append(row)
    return pl.DataFrame(rows, schema_overrides={k: pl.Float64 for k in KEYS + ["others_N", "others_F", "others_V"]})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", action="append", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--interventions", action="append", default=[]); ap.add_argument("--rollout", action="append", default=[])
    a = ap.parse_args(); t0 = time.time()
    idx = pl.read_parquet(f"{ROOT}/eval/eval_index.parquet")
    pred = load_preds(a.pred)
    df = idx.join(pred.drop("variant"), on="sample_id", how="inner").with_columns((pl.col("q90") - pl.col("q10")).alias("width"))
    print(f"predictions {pred.height:,}, matched eval samples {df.height:,}, parsed {df['parsed'].mean():.4f}")
    res = {"name": a.name, "n_pred": pred.height, "n_matched": df.height, "blocks": {}, "horizon": {}, "gap": {}, "interventions": {}, "rollout": None}
    for (split, task, delta), g in sorted(df.group_by("split", "task", "delta"), key=lambda kv: kv[0]):
        res["blocks"][f"{split}|{task}|{delta}"] = M.score_block(g)
    for (split, task), g in sorted(df.group_by("split", "task"), key=lambda kv: kv[0]):
        res["horizon"][f"{split}|{task}"] = M.horizon_monotonicity(g)
    for split in df["split"].unique().sort().to_list():
        for delta in ["5m", "1h", "24h"]:
            bj = res["blocks"].get(f"{split}|J|{delta}", {}); bs = res["blocks"].get(f"{split}|S|{delta}", {})
            if "mae" in bj and "mae" in bs:
                res["gap"][f"{split}|{delta}"] = {"err_J_mae": bj["mae"], "err_S_mae": bs["mae"], "err_S_minus_J": bs["mae"] - bj["mae"], "crps_S_minus_J": bs["crps"] - bj["crps"], "brier_S_minus_J": bs["p_zero_brier"] - bj["p_zero_brier"]}
    if a.interventions:
        inter = pl.read_parquet(f"{ROOT}/eval/interventions.parquet")
        pi = load_preds(a.interventions)
        ji = inter.join(pi.drop("variant"), on="sample_id", how="inner")
        for split, g in ji.group_by("split"):
            res["interventions"][split[0]] = M.intervention_metrics(g)
    if a.rollout:
        truth = pl.DataFrame([{"chain_id": r["chain_id"], "step": r["step"], "sample_id": r["sample_id"], "y_mid_t": r["target"]["mid_t"], "y_mid_end": r["target"]["mid_end"]} for r in map(json.loads, open(f"{ROOT}/eval/rollout_K0.jsonl"))])  # truth is K-independent
        pr = load_preds(a.rollout).join(truth.select("sample_id", "chain_id", "step"), on="sample_id").filter(pl.col("parsed"))
        res["rollout"] = M.rollout_metrics(pr.select("chain_id", "step", "q50", "q10", "q90"), truth.select("chain_id", "step", "y_mid_t", "y_mid_end")).to_dicts()
    os.makedirs("out/eval", exist_ok=True)
    json.dump(res, open(f"out/eval/pm_{a.name}.json", "w"), indent=1, default=float)
    # markdown
    cols = ["n", "parsed", "p_zero_brier", "p_zero_logloss", "n_nonzero", "mae", "mae_nochange", "skill_vs_nochange", "crps", "mae_bin", "direction_acc", "direction_acc_covered", "coverage_80", "r2_all", "pnl_sign_acc", "control_p_zero_pred", "control_p_zero_true", "control_mean_abs_q50", "others_N_mae", "others_F_mae", "others_V_mae"]
    lines = [f"# Polymarket eval: {a.name}", "", f"predictions {pred.height:,}; matched {df.height:,}; runtime {time.time() - t0:.0f}s", "", "## L1 / L2 per (split, task, horizon)", "",
             "| block | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, b in res["blocks"].items():
        lines.append(f"| {k} | " + " | ".join("" if b.get(c) is None else (f"{b[c]:.4f}" if isinstance(b[c], float) else str(b[c])) for c in cols) + " |")
    lines += ["", "## Err_S - Err_J (MAE on non-zero windows)", "", "| split/horizon | Err_J | Err_S | S-J | CRPS S-J | Brier S-J |", "|---|---|---|---|---|---|"]
    for k, g in res["gap"].items():
        lines.append(f"| {k} | {g['err_J_mae']:.4f} | {g['err_S_mae']:.4f} | {g['err_S_minus_J']:+.4f} | {g['crps_S_minus_J']:+.4f} | {g['brier_S_minus_J']:+.4f} |")
    lines += ["", "## Horizon sweep monotonicity (per point over 5m -> 1h -> 24h)", "", "| split/task | n | p_zero non-increasing | width non-decreasing |", "|---|---|---|---|"]
    for k, h in res["horizon"].items():
        if h:
            lines.append(f"| {k} | {h['n_points']} | {h['p_zero_monotone_rate']:.3f} | {h['width_monotone_rate']:.3f} |")
    if res["interventions"]:
        lines += ["", "## L3 interventions (Task S, 1h, mid-band anchors)", ""]
        for split, m in res["interventions"].items():
            lines.append(f"- **{split}**: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
    if res["rollout"]:
        lines += ["", "## Rollout (24 x 1h, true-log protocol)", "", "| step | MAE path | MAE no-change | coverage 80 |", "|---|---|---|---|"] + [f"| {r['step']} | {r['mae_path']:.4f} | {r['mae_path_nochange']:.4f} | {r['coverage_80']:.3f} |" for r in res["rollout"]]
    open(f"out/eval/pm_{a.name}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"-> out/eval/pm_{a.name}.md")


if __name__ == "__main__":
    main()
