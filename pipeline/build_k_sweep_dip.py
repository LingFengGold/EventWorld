"""History-length sweep (L2), Diplomacy: eval text for the sub1k subset at K in {1, 6, 10} (K = number of previous phases, each with board, orders
and results), with the same designated powers and renderers as dip_build_samples.py. K = 0 and 3 files already exist (sub1k_7p_{J,S}_K{0,3}).
Writes data/processed/diplomacy/eval_text/sub1k_7p_{J,S}_K{K}.jsonl."""
import json, sys
import polars as pl, pandas as pd
from eventworld.dip_serialize import render_task_j, render_task_j_target, render_task_s, render_task_s_direct_target
D = "data/processed/diplomacy"
ids = set(json.load(open("docs/sub1k_ids.json"))["diplomacy_transitions"])
common = pd.read_parquet(f"{D}/eval_common_7p.parquet"); common = common[common.sample_id.isin(ids)]
games = common.game_id.unique().tolist()
T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("game_id").is_in(games)).select(
    "sample_id", "game_id", "map", "rules_no_check", "phase_idx", "phase_type", "difficulty_tier", "state_json", "orders_json", "results_json", "next_state_json").collect().to_pandas()
by_game = {g: d.sort_values("phase_idx") for g, d in T.groupby("game_id")}
rows = T.set_index("sample_id")
for K in (1, 6, 10):
    with open(f"{D}/eval_text/sub1k_7p_J_K{K}.jsonl", "w") as fj, open(f"{D}/eval_text/sub1k_7p_S_K{K}.jsonl", "w") as fs:
        for c in common.itertuples():
            r = rows.loc[c.sample_id]
            st, o, res, ns = (json.loads(r[k]) for k in ("state_json", "orders_json", "results_json", "next_state_json"))
            rules = ["NO_CHECK"] if r["rules_no_check"] else []
            h = by_game[c.game_id]; h = h[h.phase_idx < r["phase_idx"]].tail(K)
            hist = [(json.loads(x.state_json), json.loads(x.orders_json), json.loads(x.results_json)) for x in h.itertuples()]
            fj.write(json.dumps(dict(sample_id=c.sample_id, task="J", K=K, pop="7p", phase_type=r["phase_type"], tier=r["difficulty_tier"],
                                     prompt=render_task_j(r["map"], rules, 7, st, o, history=hist), target=render_task_j_target(res, st, ns))) + "\n")
            for pw in json.loads(c.designated):
                fs.write(json.dumps(dict(sample_id=f"{c.sample_id}:{pw}", task="S", K=K, pop="7p", power=pw, phase_type=r["phase_type"], tier=r["difficulty_tier"],
                                         prompt=render_task_s(r["map"], rules, 7, hist, st, pw, o.get(pw)), target=render_task_s_direct_target(ns))) + "\n")
    print(K, sum(1 for _ in open(f"{D}/eval_text/sub1k_7p_J_K{K}.jsonl")), sum(1 for _ in open(f"{D}/eval_text/sub1k_7p_S_K{K}.jsonl")))
# consistency check: rebuilding K = 3 with this code must reproduce the existing sub1k K = 3 prompts
old = {json.loads(l)["sample_id"]: json.loads(l)["prompt"] for l in open(f"{D}/eval_text/sub1k_7p_J_K3.jsonl")}
c = common.iloc[0]; r = rows.loc[c.sample_id]; h = by_game[c.game_id]; h = h[h.phase_idx < r["phase_idx"]].tail(3)
new = render_task_j(r["map"], ["NO_CHECK"] if r["rules_no_check"] else [], 7, json.loads(r["state_json"]), json.loads(r["orders_json"]),
                    history=[(json.loads(x.state_json), json.loads(x.orders_json), json.loads(x.results_json)) for x in h.itertuples()])
print("K=3 prompt reproduced:", new == old[c.sample_id])
