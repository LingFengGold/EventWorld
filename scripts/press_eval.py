"""Score press-pipeline predictions with the unchanged scorer scripts/eval_dip.py, pointed at the press transitions.

eval_dip.py reads its references from data/processed/diplomacy/transitions.parquet (module constant D). This wrapper imports it, sets
eval_dip.D = data/processed/diplomacy_press and calls its main(), so scoring behaviour is identical. Task mapping: SM is scored like S
(next state only), SFM like SF (OTHERS block resolved by the engine + the model's own state). Afterwards it adds a summary split by the
paper's interaction levels (high = movement phases) and by whether the acting power had any message in its view.

Usage (from the repo root):
  python scripts/press_eval.py --task SM --pred out/eval/local/<tag>/press_SM_K0_zs.jsonl --out out/eval/local/<tag>/press_SM_K0_zs
  python scripts/press_eval.py --task SM --pred ... --out ... --paired_with out/eval/local/<tag>/press_S_K0_zs   # paired difference vs S
Writes eval_dip's tables.md / metrics.json / per_sample.parquet plus press_summary.json in --out.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import eval_dip

PD = "data/processed/diplomacy_press"
SCORER_TASK = {"J": "J", "S": "S", "SM": "S", "SF": "SF", "SFM": "SF"}


def summary(d, task):
    keys = ["phase_exact", "unit_f1"] if SCORER_TASK[task] in ("J", "S") else ["state_exact_direct", "state_exact_factored", "others_top1", "parsed"]
    hi = d.phase_type == "M"
    out = {"n": int(len(d)), "n_high": int(hi.sum()), "n_low": int((~hi).sum())}
    for k in keys:
        out[k] = float(d[k].mean()); out[f"{k}_high"] = float(d.loc[hi, k].mean()); out[f"{k}_low"] = float(d.loc[~hi, k].mean())
    if "n_msgs" in d:
        has = d.n_msgs > 0
        out["n_with_msgs"] = int(has.sum())
        for k in keys[:3]:
            out[f"{k}_with_msgs"] = float(d.loc[has, k].mean()); out[f"{k}_no_msgs"] = float(d.loc[~has, k].mean())
            out[f"{k}_high_with_msgs"] = float(d.loc[has & hi, k].mean())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=list(SCORER_TASK)); ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--paired_with", default=None, help="another press_eval output dir on the same samples")
    a = ap.parse_args()
    eval_dip.D = PD
    sys.argv = ["eval_dip.py", "--task", SCORER_TASK[a.task], "--pred", a.pred, "--out", a.out]
    eval_dip.main()
    d = pd.read_parquet(f"{a.out}/per_sample.parquet")
    meta = pd.DataFrame([{k: v for k, v in json.loads(l).items() if k not in ("prompt", "target")} for l in open(f"{PD}/eval_text/press_test_{a.task}_K0.jsonl")])
    cols = ["sample_id", "interaction"] + (["n_msgs", "msgs_truncated"] if a.task != "J" else [])
    d = d.merge(meta[cols], on="sample_id", how="left")
    s = summary(d, a.task)
    s["n_expected"] = int(len(meta)); s["coverage"] = len(d) / len(meta)
    if a.paired_with:   # paired difference (this run minus the other run) on the shared sample ids, for every metric both runs have
        b = pd.read_parquet(f"{a.paired_with}/per_sample.parquet")
        s["paired"] = {"vs": a.paired_with}
        for k in [k for k in ("phase_exact", "unit_f1", "state_exact_direct", "state_exact_factored", "others_top1") if k in d and k in b]:
            m = d[["sample_id", "phase_type", k] + (["n_msgs"] if "n_msgs" in d else [])].merge(b[["sample_id", k]].rename(columns={k: "other"}), on="sample_id")
            def pdiff(x):
                diff = x[k] - x["other"]
                return {"n": int(len(x)), "diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None}
            s["paired"][k] = {"all": pdiff(m), "high": pdiff(m[m.phase_type == "M"]), "low": pdiff(m[m.phase_type != "M"])}
            if "n_msgs" in m:
                s["paired"][k]["with_msgs"] = pdiff(m[m.n_msgs > 0]); s["paired"][k]["high_with_msgs"] = pdiff(m[(m.n_msgs > 0) & (m.phase_type == "M")])
    json.dump(s, open(f"{a.out}/press_summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
