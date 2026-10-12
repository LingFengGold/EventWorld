"""Factored Task S (Task SF): the Task S input, but the model first predicts the other actors' behaviour and then the next state.
Diplomacy: OTHERS block (their orders) + NEXT/STATE, prompt identical to Task S (system prompt dip_SF). Polymarket: the TASK SF prompt (JSON with
the state fields plus others {N, F, V}: count, net flow, volume of the other wallets' orders in the hour), built with pm_build_eval's renderer.
Sub1k subsets. Writes diplomacy/eval_text/sub1k_7p_SF_K{0,3}.jsonl and polymarket/eval/sub1k_test_id_SF_K0.jsonl."""
import json, os, sys
import multiprocessing as mp
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.dip_serialize import render_task_s_target
D = "data/processed/diplomacy"


def dip():
    for K in (0, 3):   # K = 3: the same transitions with three phases of history (per-order accuracy with history, Table opp)
        ids = {json.loads(l)["sample_id"].rsplit(":", 1)[0] for l in open(f"{D}/eval_text/sub1k_7p_S_K{K}.jsonl")}
        T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(list(ids))).select("sample_id", "orders_json", "next_state_json").collect().to_pandas().set_index("sample_id")
        n = 0
        with open(f"{D}/eval_text/sub1k_7p_SF_K{K}.jsonl", "w") as f:
            for l in open(f"{D}/eval_text/sub1k_7p_S_K{K}.jsonl"):
                r = json.loads(l); tid, pw = r["sample_id"].rsplit(":", 1)
                r.update(task="SF", target=render_task_s_target(json.loads(T.loc[tid]["orders_json"]), pw, json.loads(T.loc[tid]["next_state_json"])))
                f.write(json.dumps(r) + "\n"); n += 1
        print(f"diplomacy SF K={K} rows", n)


def build(args):
    import pm_build_eval as B
    from eventworld.pm_state import MarketOrders, market_meta
    from eventworld import pm_serialize as ser
    mid, pts, tgs = args
    meta = market_meta(mid); m = MarketOrders(mid, meta=meta); out = []
    for row in pts:
        tg = tgs[row["point_id"]]; t = int(row["t"]); nh = int(row["n_hist"])
        actor = None if row["kind"] == "control" else {"wallet": int(row["wallet"]), "pos": row["w_pos"], "n_prev": int(row["w_n_prev"]), "flow": row["w_flow"]}
        own = [[int(a), int(b), float(c)] for a, b, c in zip(tg["own"]["dt"], tg["own"]["d"], tg["own"]["q"])]
        s = B.make_sample(m, meta, row, "SF", "1h", own, [], actor, 0)
        out.append((row["point_id"], ser.render(s)))
    return out


def pm():
    from eventworld.pm_state import ROOT
    E = f"{ROOT}/eval"; ids = set(json.load(open("docs/sub1k_ids.json"))["polymarket_points"])
    pts = pl.read_parquet(f"{ROOT}/points.parquet").filter(pl.col("point_id").is_in(list(ids))).to_dicts()
    tgs = {r["point_id"]: r for r in pl.read_parquet(f"{ROOT}/targets_1h.parquet").filter(pl.col("point_id").is_in(list(ids))).to_dicts()}
    by_m = {}
    for p in pts:
        by_m.setdefault(p["market_id"], []).append(p)
    with mp.get_context("spawn").Pool(32) as pool:
        P = dict(x for r in pool.map(build, [(mk, ps, {p["point_id"]: tgs[p["point_id"]] for p in ps}) for mk, ps in by_m.items()], chunksize=1) for x in r)
    n = 0
    with open(f"{E}/sub1k_test_id_SF_K0.jsonl", "w") as f:
        for l in open(f"{E}/sub1k_test_id_K0.jsonl"):
            r = json.loads(l)
            if r["task"] != "S":
                continue
            f.write(json.dumps({**r, "task": "SF", "sample_id": f"{r['point_id']}|SF|1h", "prompt": P[r["point_id"]]}) + "\n"); n += 1
    print("polymarket SF rows", n)


if __name__ == "__main__":
    dip(); pm()
