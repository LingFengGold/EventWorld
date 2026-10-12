"""History-length sweep (L2), Polymarket: eval prompts for the sub1k points (1,000 anchors + 333 controls), Delta = 1 h, Task J and S, at
K in {1, 6, 10} hours of history (K hourly bars + the orders of the last K hours, capped by pm_serialize.recent_cap), built with the same
MarketOrders / make_sample / render code as pm_build_eval.py. Also rebuilds K = 3 and checks it reproduces the frozen sub1k K = 3 prompts.
Writes data/processed/polymarket/eval/sub1k_test_id_K{K}.jsonl (same row format as the K = 0 / 3 files)."""
import json, os, sys
import multiprocessing as mp
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm_build_eval as B
from eventworld.pm_state import MarketOrders, market_meta, ROOT
from eventworld import pm_serialize as ser

E = f"{ROOT}/eval"; KS = (3, 1, 6, 10)


def build(args):
    mid, pts, tgs = args
    meta = market_meta(mid); m = MarketOrders(mid, meta=meta); out = []
    for row in pts:
        tg = tgs[row["point_id"]]; t = int(row["t"]); nh = int(row["n_hist"])
        actor = None if row["kind"] == "control" else {"wallet": int(row["wallet"]), "pos": row["w_pos"], "n_prev": int(row["w_n_prev"]), "flow": row["w_flow"]}
        own = [[int(a), int(b), float(c)] for a, b, c in zip(tg["own"]["dt"], tg["own"]["d"], tg["own"]["q"])]
        joint = [[int(m.ts[i] - t), int(m.wallet[i]), int(m.sign[i]), float(m.usd[i])] for i in range(nh, int(tg["n_end"]))]
        for task in ("J", "S"):
            for K in KS:
                s = B.make_sample(m, meta, row, task, "1h", own, joint, actor, K)
                out.append((K, row["point_id"], task, s["sample_id"], ser.render(s)))
    return out


def main():
    ids = set(json.load(open("docs/sub1k_ids.json"))["polymarket_points"])
    pts = pl.read_parquet(f"{ROOT}/points.parquet").filter(pl.col("point_id").is_in(list(ids))).to_dicts()
    tgs = {r["point_id"]: r for r in pl.read_parquet(f"{ROOT}/targets_1h.parquet").filter(pl.col("point_id").is_in(list(ids))).to_dicts()}
    by_m = {}
    for p in pts:
        by_m.setdefault(p["market_id"], []).append(p)
    with mp.get_context("spawn").Pool(32) as pool:
        res = [x for r in pool.map(build, [(mk, ps, {p["point_id"]: tgs[p["point_id"]] for p in ps}) for mk, ps in by_m.items()], chunksize=1) for x in r]
    P = {(K, sid): pr for K, _, _, sid, pr in res}
    base = [json.loads(l) for l in open(f"{E}/sub1k_test_id_K0.jsonl")]   # row template (targets, meta) from the frozen K = 0 file
    old3 = {json.loads(l)["sample_id"]: json.loads(l)["prompt"] for l in open(f"{E}/sub1k_test_id_K3.jsonl")}
    same = sum(P[(3, sid)] == pr for sid, pr in old3.items())
    print(f"K=3 rebuilt prompts identical to the frozen file: {same}/{len(old3)}")
    for K in (1, 6, 10):
        with open(f"{E}/sub1k_test_id_K{K}.jsonl", "w") as f:
            for r in base:
                f.write(json.dumps({**r, "K": K, "prompt": P[(K, r["sample_id"])]}) + "\n")
        L = [len(P[(K, r["sample_id"])]) for r in base]
        print(f"K={K}: {len(base)} rows, mean prompt chars {np.mean(L):.0f}, max {max(L)}")


if __name__ == "__main__":
    main()
