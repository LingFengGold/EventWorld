"""Render SFT data (LLaMA-Factory alpaca format with a system column: system/instruction/output) for Diplomacy from the SFT id lists.
Format v3: the task prompt goes in the system role and the rendered input in the user turn, so training renders exactly like
`apply_chat_template([...], enable_thinking=False)` at inference. v1/v2 put both in the user turn and had no empty <think> block, a train/inference
mismatch that made the first output token unstable (OTHERS vs NEXT).
Usage: dip_build_sft.py --pop 7p --n_j 10000 --n_s 10000 --K 3 --name dip7p_20k
Also writes a 1k validation file from the val split and updates data/processed/diplomacy/sft/dataset_info.json.
"""
import argparse, json, os, sys, random
import polars as pl, pandas as pd
from eventworld.dip_serialize import render_task_j, render_task_j_target, render_task_s, render_task_s_direct_target, render_task_s_target
import hashlib

def k_of(sid, K):
    """K == -1 -> mixed: half the samples K=0, half K=3, by hash of the sample id (deterministic)."""
    if K >= 0:
        return K
    hv = int(hashlib.sha1(('K' + sid).encode()).hexdigest()[:8], 16)
    return (3 if hv % 2 else 0) if K == -1 else KSWEEP[hv % len(KSWEEP)]


KSWEEP = [0, 1, 3, 6, 10]   # K = -2: uniform over these history lengths (L2 history sweep)
from eventworld.prompts import SYS
D = "data/processed/diplomacy"

def load_rows(sample_ids):
    T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(list(sample_ids))).select(
        "sample_id", "game_id", "phase_idx", "map", "rules_no_check", "state_json", "orders_json", "results_json", "next_state_json").collect().to_pandas()
    return T.set_index("sample_id")

def history(game_rows, game_id, phase_idx, K):
    d = game_rows.get(game_id)
    if d is None or K == 0:
        return []
    d = d[d.phase_idx < phase_idx].tail(K)
    return [(json.loads(r.state_json), json.loads(r.orders_json), json.loads(r.results_json)) for r in d.itertuples()]

SF = False

def build(pop, ids_j, ids_s, K, npow):
    base_s = [i.rsplit(":", 1) for i in ids_s]
    need = set(ids_j) | {b for b, _ in base_s}
    T = load_rows(need)
    games = set(T.game_id)
    G = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("game_id").is_in(list(games))).select("game_id", "phase_idx", "state_json", "orders_json", "results_json").collect().to_pandas()
    game_rows = {g: d.sort_values("phase_idx") for g, d in G.groupby("game_id")}
    out = []
    for sid in ids_j:
        r = T.loc[sid]; st, o, res, ns = (json.loads(r[c]) for c in ("state_json", "orders_json", "results_json", "next_state_json"))
        rules = ["NO_CHECK"] if r["rules_no_check"] else []
        h = history(game_rows, r["game_id"], r["phase_idx"], k_of(sid, K))
        out.append({"system": SYS["dip_J"], "instruction": render_task_j(r["map"], rules, npow, st, o, history=h), "output": render_task_j_target(res, st, ns)})
    for sid, pw in base_s:
        r = T.loc[sid]; st, o, res, ns = (json.loads(r[c]) for c in ("state_json", "orders_json", "results_json", "next_state_json"))
        rules = ["NO_CHECK"] if r["rules_no_check"] else []
        h = history(game_rows, r["game_id"], r["phase_idx"], k_of(sid, K))
        if SF:   # --sf (factored Task S): same input, system dip_SF, target = the other powers' orders (OTHERS block), then NEXT/STATE
            out.append({"system": SYS["dip_SF"], "instruction": render_task_s(r["map"], rules, npow, h, st, pw, o.get(pw)), "output": render_task_s_target(o, pw, ns)})
        else:
            out.append({"system": SYS["dip_S"], "instruction": render_task_s(r["map"], rules, npow, h, st, pw, o.get(pw)), "output": render_task_s_direct_target(ns)})
    random.Random(0).shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pop", default="7p"); ap.add_argument("--n_j", type=int, default=10000); ap.add_argument("--n_s", type=int, default=10000)
    ap.add_argument("--sf", action="store_true", help="Task S half as the S-factored variant (other powers' orders, then the next state)")
    ap.add_argument("--K", type=int, default=-1, help="history length; -1 = mixed {0,3} by sample hash (protocol v3 default); -2 = uniform over {0,1,3,6,10}"); ap.add_argument("--name", required=True); a = ap.parse_args()
    global SF; SF = a.sf
    npow = 7 if a.pop == "7p" else 2
    ids_j = open(f"{D}/sft_lists/{a.pop}_J_all.txt").read().split()[:a.n_j]
    ids_s = open(f"{D}/sft_lists/{a.pop}_S_all.txt").read().split()[:a.n_s]
    data = build(a.pop, ids_j, ids_s, a.K, npow)
    json.dump(data, open(f"{D}/sft/{a.name}.json", "w"))
    # validation: 500 J + 500 S from the val split
    V = pl.scan_parquet(f"{D}/transitions.parquet").filter((pl.col("split") == "val") & (pl.col("pop") == a.pop)).select("sample_id", "state_json").collect().to_pandas().sample(n=1000, random_state=0)
    vj = list(V.sample_id[:500]); vs = []
    for sid, sj in zip(V.sample_id[500:], V.state_json[500:]):
        alive = [p for p, us in json.loads(sj)["units"].items() if us]
        vs.append(f"{sid}:{random.Random(sid).choice(alive)}")
    val = build(a.pop, vj, vs, a.K, npow)
    json.dump(val, open(f"{D}/sft/{a.name}_val.json", "w"))
    info_path = f"{D}/sft/dataset_info.json"
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for nm in (a.name, a.name + "_val"):
        info[nm] = {"file_name": f"{nm}.json", "columns": {"prompt": "instruction", "response": "output", "system": "system"}}  # system role = task prompt, user = rendered input: identical to the inference-time chat template
    json.dump(info, open(info_path, "w"), indent=1)
    lens = [len(d["instruction"]) + len(d["output"]) for d in data]
    print(f"{a.name}: {len(data)} train ({a.n_j} J + {a.n_s} S, K={a.K}), {len(val)} val; chars p50 {sorted(lens)[len(lens)//2]} max {max(lens)}")

if __name__ == "__main__":
    main()
