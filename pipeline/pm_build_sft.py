"""Polymarket SFT data (protocol v3, binned output). LLaMA-Factory alpaca format with a system column.

Sampling from the TRAIN split only: points disjoint between Task J and Task S (hash of point_id), 80% anchors / 20% controls, horizon 1h for 70%,
5m and 24h for 15% each, history K in {0,3} 50/50 by hash. Prompts are the eval renderer's prompts with the JSON output line replaced by the binned
output spec (eventworld.pm_serialize.sft_prompt); targets are realised quantities only (eventworld.pm_serialize.sft_target): bin letter, point Delta-mid, end prices.
Validation: 1,000 samples from the val split built the same way. Usage:
  python pipeline/pm_build_sft.py --n 20000 --name pm_20k   [--n 320000 --name pm_320k]
Outputs data/processed/polymarket/sft/{name}.json, {name}_val.json, dataset_info.json.
"""
import sys, os, json, hashlib, argparse, time
import multiprocessing as mp
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import MarketOrders, market_meta, ROOT, DELTAS
from eventworld import pm_serialize as ser
from eventworld.prompts import SYS
import pm_build_eval as E

OUT = f"{ROOT}/sft"
def h(s, salt): return int(hashlib.sha1((salt + str(s)).encode()).hexdigest()[:8], 16)

def build_market(mid):
    W = E.WORK
    meta = market_meta(mid); m = MarketOrders(mid, meta=meta)
    rows = W["points"].filter(pl.col("market_id") == mid).to_dicts()
    out = []
    for row in rows:
        delta = row["delta"]; tg = W[f"tg_{delta}"].get(row["point_id"])
        if tg is None or tg["dmid"] is None or tg["p_buy_end"] is None or tg["p_sell_end"] is None:
            continue
        t = int(row["t"]); nh = int(row["n_hist"]); n_end = int(tg["n_end"])
        actor = None if row["kind"] == "control" else {"wallet": int(row["wallet"]), "pos": row["w_pos"], "n_prev": int(row["w_n_prev"]), "flow": row["w_flow"]}
        own = [[int(a), int(b), float(c)] for a, b, c in zip(tg["own"]["dt"], tg["own"]["d"], tg["own"]["q"])]
        joint = [[int(m.ts[i] - t), int(m.wallet[i]), int(m.sign[i]), float(m.usd[i])] for i in range(nh, n_end)]
        task = "SF" if (W.get("sf") and row["task"] == "S") else row["task"]   # --sf: the Task S half becomes the S-factored variant
        s = E.make_sample(m, meta, row, task, delta, own, joint, actor, int(row["K"]))
        out.append({"system": SYS[f"pm_{task}_sft"], "instruction": ser.sft_prompt(ser.render(s)), "output": ser.sft_target_sf(tg) if task == "SF" else ser.sft_target(tg),
                    "_meta": {"point_id": row["point_id"], "task": row["task"], "delta": delta, "K": int(row["K"]), "kind": row["kind"]}})
    return out

def select(pts, split, n, seed, ks=(0, 3)):
    rng = np.random.default_rng(seed)
    p = pts.filter(pl.col("split") == split)
    a = p.filter(pl.col("kind") == "anchor"); c = p.filter(pl.col("kind") == "control")
    na, nc = int(n * 0.8), n - int(n * 0.8)
    pick = pl.concat([a[rng.choice(a.height, size=min(na, a.height), replace=False).tolist()], c[rng.choice(c.height, size=min(nc, c.height), replace=False).tolist()]])
    ids = pick["point_id"].to_list()
    task = ["J" if h(i, "task") % 2 == 0 else "S" for i in ids]
    u = [h(i, "delta") % 100 for i in ids]; delta = ["1h" if x < 70 else ("5m" if x < 85 else "24h") for x in u]
    K = [3 if h(i, "K") % 2 else 0 for i in ids] if tuple(ks) == (0, 3) else [ks[h(i, "K") % len(ks)] for i in ids]
    return pick.with_columns(pl.Series("task", task), pl.Series("delta", delta), pl.Series("K", K))

def run(pick, nproc, sf=False):
    work = {"points": pick, "sf": sf}
    ids = pick["point_id"]
    for d in DELTAS:
        tg = pl.read_parquet(f"{ROOT}/targets_{d}.parquet").filter(pl.col("point_id").is_in(ids))
        work[f"tg_{d}"] = {r["point_id"]: r for r in tg.to_dicts()}
    mids = pick["market_id"].unique().sort().to_list()
    with mp.get_context("spawn").Pool(nproc, initializer=E._init, initargs=(work,)) as pool:
        res = pool.map(build_market, mids, chunksize=2)
    data = [r for x in res for r in x]
    np.random.default_rng(0).shuffle(data)
    return data

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--name", required=True); ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--sf", action="store_true", help="factored Task S: same points, the Task S half rendered as Task SF (target starts with the other wallets' N/F/V)")
    ap.add_argument("--ks", default="0,3", help="history lengths assigned by point hash; 0,3 = protocol v3; 0,1,3,6,10 = L2 history sweep")
    a = ap.parse_args(); ks = [int(x) for x in a.ks.split(",")]; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    pts = pl.read_parquet(f"{ROOT}/points.parquet").join(pl.read_parquet(f"{ROOT}/splits.parquet").select("point_id", "split"), on="point_id", how="inner")
    tr = run(select(pts, "train", int(a.n * 1.03), seed=0, ks=ks), a.nproc, a.sf)[:a.n]   # small oversample: a few points lack an end price and are dropped
    va = run(select(pts, "val", 1030, seed=1, ks=ks), a.nproc, a.sf)[:1000]
    for name, d in ((a.name, tr), (f"{a.name}_val", va)):
        json.dump([{k: v for k, v in x.items() if k != "_meta"} for x in d], open(f"{OUT}/{name}.json", "w"))
    info_path = f"{OUT}/dataset_info.json"; info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for nm in (a.name, f"{a.name}_val"):
        info[nm] = {"file_name": f"{nm}.json", "columns": {"prompt": "instruction", "response": "output", "system": "system"}}
    json.dump(info, open(info_path, "w"), indent=1)
    import collections
    c = collections.Counter((x["_meta"]["task"], x["_meta"]["delta"], x["_meta"]["K"], x["_meta"]["kind"]) for x in tr)
    zero = np.mean([x["output"].startswith("BIN K") for x in tr])
    print(f"{a.name}: {len(tr)} train, {len(va)} val; zero-bin share {zero:.3f}; mean prompt chars {np.mean([len(x['instruction']) for x in tr]):.0f}; {time.time()-t0:.0f}s")
    print("composition (task, delta, K, kind):", sorted(c.items())[:24])

if __name__ == "__main__":
    main()
