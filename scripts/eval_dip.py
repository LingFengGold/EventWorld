"""Score Diplomacy prediction files. Usage:
  eval_dip.py --task J  --pred preds.jsonl [--pop 7p] --out out/eval/dip/<run>
  eval_dip.py --task S  --pred preds.jsonl --out ...          (pred lines: sample_id "<transition>:<POWER>", samples=[text,...] or pred.text)
  eval_dip.py --task CF --pred_fact fact.jsonl --pred_cf cf.jsonl --out ...   (cf lines keyed by cf_id)
Prediction JSONL: {"sample_id": ..., "pred": {"text": ...}, "samples": [ {"text": ...}, ... ] }
"""
import argparse, json, os, sys
import polars as pl, pandas as pd
from eventworld.metrics_dip import score_task_s_direct, score_task_j, score_task_s, score_counterfactual
D = "data/processed/diplomacy"

def load_preds(fn, key="sample_id"):
    out = {}
    for line in open(fn):
        r = json.loads(line)
        texts = [s["text"] if isinstance(s, dict) else s for s in r.get("samples", [])] or ([r["pred"]["text"]] if r.get("pred") else [])
        out[r[key]] = texts
    return out

def rows_for(sample_ids):
    T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(list(sample_ids))).collect().to_pandas().set_index("sample_id")
    def mk(r):
        return dict(state=json.loads(r["state_json"]), orders=json.loads(r["orders_json"]), results=json.loads(r["results_json"]), next_state=json.loads(r["next_state_json"]),
                    phase_type=r["phase_type"], map=r["map"], rules=["NO_CHECK"] if r["rules_no_check"] else [], pop=r["pop"], tier=r["difficulty_tier"], year=int(r["phase_name"][1:5]), game_id=r["game_id"])
    return {sid: mk(r) for sid, r in T.iterrows()}

def summarize(df, out_dir, by):
    os.makedirs(out_dir, exist_ok=True)
    num = df.select_dtypes("number")
    overall = num.mean().to_dict(); overall["n"] = len(df)
    if "ref_size_bucket" in df:  # counterfactual headline metrics: change-set F1 where the intervention actually changes the outcome, spurious rate where it does not
        ne = df[df["ref_size_bucket"] != "0"]; em = df[df["ref_size_bucket"] == "0"]
        overall["cf_f1_ref_nonempty"] = float(ne["cf_f1"].mean()) if len(ne) else float("nan"); overall["cf_exact_ref_nonempty"] = float(ne["cf_exact"].mean()) if len(ne) else float("nan")
        overall["insensitive_ref_nonempty"] = float(ne["insensitive"].mean()) if len(ne) else float("nan")
        overall["spurious_ref_empty"] = float(em["spurious"].mean()) if len(em) else float("nan"); overall["n_ref_nonempty"] = int(len(ne))
    lines = ["# overall", pd.Series(overall).round(4).to_string(), ""]
    for b in by:
        if b in df:
            lines += [f"# by {b}", df.groupby(b)[num.columns].mean().round(4).assign(n=df.groupby(b).size()).to_markdown(), ""]
    open(f"{out_dir}/tables.md", "w").write("\n".join(lines)); json.dump(overall, open(f"{out_dir}/metrics.json", "w"), indent=1)
    df.to_parquet(f"{out_dir}/per_sample.parquet", index=False); print("\n".join(lines[:2]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--pred"); ap.add_argument("--pred_fact"); ap.add_argument("--pred_cf"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.task == "J":
        preds = load_preds(a.pred); rows = rows_for(preds)
        recs = []
        for sid, texts in preds.items():
            if sid not in rows: continue
            r = rows[sid]; s = score_task_j(texts[0] if texts else "", r)
            s.update(sample_id=sid, pop=r["pop"], phase_type=r["phase_type"], tier=r["tier"], year_bucket="1901" if r["year"] == 1901 else ("1902-04" if r["year"] <= 1904 else "1905+")); recs.append(s)
        summarize(pd.DataFrame(recs), a.out, ["pop", "phase_type", "tier", "year_bucket"])
    elif a.task in ("S", "SF"):  # S = main definition (next state only, state metrics); SF = factored diagnostic (OTHERS block + state)
        preds = load_preds(a.pred); base = {k.rsplit(":", 1)[0] for k in preds}; rows = rows_for(base)
        recs = []
        for sid, texts in preds.items():
            tid, power = sid.rsplit(":", 1)
            if tid not in rows: continue
            r = rows[tid]; s = score_task_s_direct(texts[0] if texts else "", r) if a.task == "S" else score_task_s(texts, r, power, r["rules"])
            s.update(sample_id=sid, power=power, pop=r["pop"], phase_type=r["phase_type"], tier=r["tier"]); recs.append(s)
        summarize(pd.DataFrame(recs), a.out, ["pop", "phase_type", "tier", "power"])
    elif a.task == "CF":
        fact = load_preds(a.pred_fact); cf = load_preds(a.pred_cf, key="cf_id")
        meta = pd.read_parquet(f"{D}/counterfactuals.parquet").set_index("cf_id")
        rows = rows_for({c.rsplit(":", 2)[0] for c in cf}); recs = []
        for cid, texts in cf.items():
            if cid not in meta.index: continue
            m = meta.loc[cid]; sid = m.sample_id; fkey = f"{sid}:{m.power}" if f"{sid}:{m.power}" in fact else sid
            if fkey not in fact or sid not in rows: continue
            s = score_counterfactual(fact[fkey][0], texts[0], rows[sid]["next_state"], json.loads(m.ref_delta_json))
            s.update(cf_id=cid, scheme=m.scheme, tier=rows[sid]["tier"], ref_size_bucket="0" if m.ref_delta_size == 0 else ("1-2" if m.ref_delta_size <= 2 else "3+")); recs.append(s)
        summarize(pd.DataFrame(recs), a.out, ["scheme", "tier", "ref_size_bucket"])

if __name__ == "__main__":
    main()
