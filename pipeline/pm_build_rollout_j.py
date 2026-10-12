"""Task J version of the Polymarket rollout chains (same 500 test_id anchors x 24 hourly steps as rollout_K0.jsonl, K=0).
Each step's prompt lists ALL orders in the hour (JOINT block, execution prices hidden) instead of only the agent's own orders; state and targets
are identical to the Task S file. Step 0 also carries `direct24_prompt`: the Task J prompt at the chain start with horizon 24 h and every order
in the 24 hours. Built with the same MarketOrders / make_sample / render code as pm_build_eval.py.
Writes data/processed/polymarket/eval/rollout_J_K0.jsonl.
"""
import sys, os, json
import multiprocessing as mp
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm_build_eval as B
from eventworld.pm_state import MarketOrders, market_meta, ROOT, HOUR
from eventworld import pm_serialize as ser

E = f"{ROOT}/eval"


def build(args):
    mid, pts = args
    meta = market_meta(mid); m = MarketOrders(mid, meta=meta)
    out = []
    for row in pts:
        t, nh, w = int(row["t"]), int(row["n_hist"]), int(row["wallet"])
        for k in range(B.ROLL_STEPS):
            tk = t + k * HOUR
            nhk = nh if k == 0 else int(np.searchsorted(m.ts, tk, side="left"))
            st = m.state(tk, nhk, w); win = m.window(tk, nhk, HOUR, w)
            srow = {**{kk: st[kk] for kk in B.STATE_KEYS}, "t": tk, "n_hist": nhk, "point_id": f"{row['point_id']}|k{k}", "wallet": w}
            own = [[o[0], o[1], o[2]] for o in win["own"]]
            joint = [[int(m.ts[i] - tk), int(m.wallet[i]), int(m.sign[i]), float(m.usd[i])] for i in range(nhk, win["n_end"])]
            actor = {"wallet": w, "pos": st["w_pos"], "n_prev": st["w_n_prev"], "flow": st["w_flow"]}
            s = B.make_sample(m, meta, srow, "J", "1h", own, joint, actor, 0)
            rec = {"chain_id": row["point_id"], "step": k, "sample_id": s["sample_id"], "task": "J", "K": 0, "delta": "1h", "prompt": ser.render(s)}
            if k == 0:
                w24 = m.window(tk, nhk, 24 * HOUR, w)
                j24 = [[int(m.ts[i] - tk), int(m.wallet[i]), int(m.sign[i]), float(m.usd[i])] for i in range(nhk, w24["n_end"])]
                o24 = [[o[0], o[1], o[2]] for o in w24["own"]]
                s24 = B.make_sample(m, meta, srow, "J", "24h", o24, j24, actor, 0)
                rec["direct24_prompt"] = ser.render(s24)
            out.append(rec)
    return out


def main():
    S = [json.loads(l) for l in open(f"{E}/rollout_K0.jsonl")]
    cids = sorted({r["chain_id"] for r in S})
    pts = pl.read_parquet(f"{ROOT}/points.parquet").filter(pl.col("point_id").is_in(cids)).to_dicts()
    by_m = {}
    for p in pts:
        by_m.setdefault(p["market_id"], []).append(p)
    with mp.get_context("spawn").Pool(32) as pool:
        res = pool.map(build, list(by_m.items()), chunksize=1)
    J = {(r["chain_id"], r["step"]): r for x in res for r in x}
    n = 0
    with open(f"{E}/rollout_J_K0.jsonl", "w") as f:
        for r in S:   # same order, state and targets as the Task S file
            j = J[(r["chain_id"], r["step"])]
            f.write(json.dumps({**r, "task": "J", "sample_id": j["sample_id"], "prompt": j["prompt"], **({"direct24_prompt": j["direct24_prompt"]} if "direct24_prompt" in j else {})}) + "\n"); n += 1
    print(f"wrote {n} rows ({len(cids)} chains) -> {E}/rollout_J_K0.jsonl")


if __name__ == "__main__":
    main()
