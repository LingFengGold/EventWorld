"""Assemble the Hugging Face dataset folder: the processed benchmarks (data/processed/...) and every model output reported in the paper
(out/...), with the same relative paths the code uses, so `hf download ... --local-dir <repo>` drops them in place. Files are copied (--link: hard links instead); the folder also gets the dataset card tools/hf_dataset_card.md as README.md.
Usage: python tools/prepare_hf.py --src <directory with data/ and out/> --dst <upload folder> [--link]
"""
import argparse, fnmatch, os, shutil

DIP = "data/processed/diplomacy"; PRESS = "data/processed/diplomacy_press"; PM = "data/processed/polymarket"
SFT_DIP = [f"dip7p_{n}k" for n in (5, 20, 80, 320)] + ["dip7p_20k_sf", "dip7p_20k_ksweep"]
SFT_PM = [f"pm_{n}k" for n in (5, 20, 80, 320)] + ["pm_20k_sf", "pm_20k_ksweep"]
DATA = [  # (path or glob relative to --src)
    *[f"{DIP}/{f}" for f in ("transitions.parquet", "games_index.parquet", "eval_common_7p.parquet", "eval_common_2p.parquet", "rollout_chains.parquet",
                             "counterfactuals.parquet", "order_freq_train.parquet")],
    *[f"{DIP}/eval_text/{g}" for g in ("common_7p_*.jsonl", "common_2p_*_K0.jsonl", "demos_dip_*.jsonl", "sub1k_7p_J_K*.jsonl", "sub1k_7p_S_K*.jsonl",
                                       "sub1k_7p_SF_K*.jsonl", "sub1k_7p_CF_*_K0.jsonl")],
    *[f"{DIP}/sft/{n}{v}.json" for n in SFT_DIP for v in ("", "_val")],
    f"{DIP}/sft_agent/agent_*.json",
    *[f"{PRESS}/{f}" for f in ("transitions.parquet", "games_index.parquet", "eval_text/press_test_*.jsonl")],
    *[f"{PRESS}/sft/press_20k_{c}{v}.json" for c in ("plain", "msg", "sf", "sfmsg") for v in ("", "_val")],
    *[f"{PM}/{f}" for f in ("points.parquet", "splits.parquet", "markets.parquet", "market_splits.parquet", "targets_5m.parquet", "targets_1h.parquet",
                            "targets_24h.parquet", "response_curve.parquet", "response_curve_stats.parquet")],
    *[f"{PM}/preds/{n}{s}.jsonl" for n in ("nochange", "lgbm") for s in ("", "_interventions")],
    *[f"{PM}/eval/{g}" for g in ("eval_index.parquet", "interventions.parquet", "eval_test_id_K*.jsonl", "eval_test_ood_K*.jsonl", "interventions_K*.jsonl",
                                 "rollout_K*.jsonl", "rollout_J_K0.jsonl", "demos_*.jsonl", "sub1k_test_id_K*.jsonl", "sub1k_test_id_SF_K0.jsonl",
                                 "sub1k_test_ood_K0.jsonl")],
    *[f"{PM}/sft/{n}{v}.json" for n in SFT_PM for v in ("", "_val")],
]
LOCAL = ["qwen3-8b", "sft1.7b-20k-v4", "sft1.7b-pm20k", "qwen3-8b-dip20k", "qwen3-8b-pm20k", "qwen3-8b-dip320k", "qwen3-8b-pm320k",
         "sft0.6b-dip20k", "sft0.6b-pm20k", "sft4b-dip20k", "sft4b-pm20k", *[f"sft1.7b-{d}{n}k" for d in ("dip", "pm") for n in (5, 80, 320)],
         *[f"sft1.7b-{d}{n}k-scratch" for d in ("dip", "pm") for n in (20, 320)], "sft1.7b-dip20k-sf", "sft1.7b-pm20k-sf",
         "qwen3-1.7b-dip20k-v4-seed43", "qwen3-1.7b-dip20k-v4-seed44", "qwen3-1.7b-dip20k-sf-seed43", "qwen3-1.7b-dip20k-sf-seed44",
         "sft1.7b-dip20k-ksweep", "sft1.7b-pm20k-ksweep", *[f"sft1.7b-press20k-{c}" for c in ("plain", "msg", "sf", "sfmsg")]]
RENAME = {"out/eval/local/delta-qwen3-8b-dip20k-sf": "out/eval/local/qwen3-8b-dip20k-sf",          # run tags as used by the code
          "out/eval/api/leakage_recall/gpt-6-luna-1.jsonl": "out/eval/api/leakage_recall/gpt-6-luna.jsonl",
          "out/eval/api/leakage_recall/gpt-6-sol-1.jsonl": "out/eval/api/leakage_recall/gpt-6-sol.jsonl"}
OUTPUTS = ([f"out/eval/local/{t}" for t in LOCAL] + ["out/eval/local/delta-qwen3-8b-dip20k-sf", "out/eval/baselines"]
           + [f"out/eval/api/{t}" for t in ("gpt-6-luna", "gpt-6-sol_sub1k", "gpt-6-luna_press", "gpt-6-sol_press")]
           + ["out/eval/api/leakage_recall/gpt-6-luna-1.jsonl", "out/eval/api/leakage_recall/gpt-6-sol-1.jsonl"]
           + [f"out/rollout/{d}" for d in ("diplomacy_chains", "diplomacy_cr", "polymarket_rollout")]
           + ["out/agent/diplomacy_verifier", "out/agent/diplomacy_syndata"])
SKIP = ["*.part", "*pilot*", "*.log", "*.in.jsonl", "eval_real1k_lminit20k*", "traj_real2_pilot*", "real_n5.parquet"]   # partial / pilot / unreported runs


def skipped(rel):
    return any(fnmatch.fnmatch(os.path.basename(rel), p) or fnmatch.fnmatch(rel, f"*/{p}") for p in SKIP)


def place(src, dst, rel, n, link=False):
    out = rel
    for a, b in RENAME.items():
        if rel == a or rel.startswith(a + "/"):
            out = b + rel[len(a):]
    s, d = os.path.join(src, rel), os.path.join(dst, out)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.exists(d):
        return n
    if link:
        try:
            os.link(s, d); return n + 1
        except OSError:
            pass
    shutil.copy2(s, d)
    return n + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="."); ap.add_argument("--dst", required=True); ap.add_argument("--link", action="store_true"); a = ap.parse_args()
    n, missing = 0, []
    import glob
    for g in DATA:
        hits = sorted(glob.glob(os.path.join(a.src, g)))
        if not hits:
            missing.append(g)
        for h in hits:
            n = place(a.src, a.dst, os.path.relpath(h, a.src), n, a.link)
    for d in ("diplomacy/sft", "polymarket/sft", "diplomacy_press/sft", "diplomacy/sft_agent"):   # dataset_info.json restricted to the released sets
        import json
        p = os.path.join(a.src, f"data/processed/{d}/dataset_info.json")
        if os.path.exists(p):
            keep = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(a.dst, f"data/processed/{d}/*.json")) if not f.endswith("dataset_info.json")}
            info = {k: v for k, v in json.load(open(p)).items() if k in keep}
            os.makedirs(os.path.join(a.dst, f"data/processed/{d}"), exist_ok=True)
            json.dump(info, open(os.path.join(a.dst, f"data/processed/{d}/dataset_info.json"), "w"), indent=1)
    for o in OUTPUTS:
        p = os.path.join(a.src, o)
        if not os.path.exists(p):
            missing.append(o); continue
        files = [p] if os.path.isfile(p) else [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs]
        for f in files:
            rel = os.path.relpath(f, a.src)
            if not skipped(rel):
                n = place(a.src, a.dst, rel, n, a.link)
    card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_dataset_card.md")
    shutil.copy(card, os.path.join(a.dst, "README.md"))
    print(f"{n} files placed under {a.dst}")
    if missing:
        print("missing (not placed):", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
