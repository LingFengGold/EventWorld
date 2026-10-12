"""P7: frozen evaluation sets, interventions, rollout chains and few-shot demonstrations for Polymarket (protocol v3).

History length K in {0, 3} (see eventworld/pm_serialize.py): every prompt file exists in a K=0 and a K=3 version; sample_ids are identical across K.
eval_{split}_K{K}.jsonl  : 3,000 random anchors + 1,000 random controls per split (seed 0), x Delta {5m,1h,24h} x task {J,S};
                           one line = {sample_id, point_id, split, kind, band, rho_bin, task, delta, K, prompt, target{...}, meta{...}}
interventions_K{K}.jsonl : 1,000 mid-band test_id anchors (+1,000 test_ood), Task S, Delta=1h; variants factual / remove / reverse / scale / null
                           with reference response from the train-only curve: ref = E_curve[dmid | A'] - E_curve[dmid | A]
rollout_K{K}.jsonl       : 500 test_id anchors x 24 hourly steps (true-log protocol): per step the true state, own orders, targets, Task S prompt
demos_{J,S}_K{K}.jsonl   : 3 fixed demonstrations per (task, K) for 3-shot prompting: train-split mid-band anchors at Delta=1h from markets that
                           appear in no eval file (seed 1); target = the JSON the model should output (end prices, realised Delta-mid as q50,
                           train-statistics dq10/dq90 around it and the train zero-move rate as p_zero, so the demo is calibrated, not degenerate)
Also writes eval_index.parquet (all sample rows without prompts; K-independent) and interventions.parquet.
"""
import sys, os, time, json
import multiprocessing as mp
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import MarketOrders, market_meta, ROOT, DELTAS, HOUR, delta_bin
from eventworld import pm_serialize as ser
from pm_response_curve import curve_lookup

N_ANCHOR, N_CONTROL, N_INTERV, N_ROLL, ROLL_STEPS, N_DEMO = 3000, 1000, 1000, 500, 24, 3
KS = [0, 3]
SPLITS = ["train", "val", "test_id", "test_ood"]
STATE_KEYS = ["p_buy", "p_sell", "mid", "spread", "D", "N1h", "F1h", "V1h", "age_d", "ttl_d", "since_last_s"]
TARGET_KEYS = ["p_buy_end", "p_sell_end", "mid_end", "spread_end", "dmid", "is_zero", "dbin", "N_all", "F_all", "V_all", "own_N", "own_F", "own_V", "own_dpos", "N_o", "F_o", "V_o", "D_end", "w_pos_end", "pnl_mtm", "rho"]
WORK = {}

def _init(work):
    global WORK
    WORK = work

def _nan(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x

def recent_within(m, t, n_hist, hours, cap):
    """History orders in [t - hours*1h, t) (block-strict: idx < n_hist), the `cap` most recent, oldest first."""
    lo = int(np.searchsorted(m.ts, int(t) - hours * HOUR, side="left"))
    lo = max(lo, int(n_hist) - cap, 0)
    return [{"ts": int(m.ts[i]), "wallet": int(m.wallet[i]), "sign": int(m.sign[i]), "usd": float(m.usd[i]), "p": float(m.p[i])} for i in range(lo, int(n_hist))]

def make_sample(m, meta, row, task, delta, own, joint, actor, K):
    st = {k: _nan(row[k]) for k in STATE_KEYS}
    hours = ser.K_HOURS[K]
    bars = [[b[0] if b[0] is not None else (st["mid"] or 0.5), b[1], b[2], b[3]] for b in m.hour_bars(row["t"], row["n_hist"], hours)] if hours else []
    recent = [[o["ts"] - int(row["t"]), o["wallet"], o["sign"], o["usd"], o["p"]] for o in recent_within(m, row["t"], row["n_hist"], hours, ser.recent_cap(K))] if hours else []
    return {"sample_id": f"{row['point_id']}|{task}|{delta}", "task": task, "delta": delta, "K": K, "t": int(row["t"]),
            "market": {"market_id": m.market_id, "question": meta["question"], "event_title": meta.get("event_title"), "topic": meta["topic"]},
            "state": st, "bars": bars, "recent": recent, "actor": actor, "own": own, "joint": joint}

def demo_target_text(tg, row, delta):
    """Calibrated demonstration output: end prices, realised Delta-mid as the median, train-statistics spread and zero-move rate."""
    st = WORK["stats"].get((row["kind"], row["band"], row["rho_bin"] if row["kind"] == "anchor" else "control", delta))
    dm = float(tg["dmid"]); r4 = lambda x: round(float(x), 4)
    out = {"p_buy": r4(tg["p_buy_end"]), "p_sell": r4(tg["p_sell_end"]), "p_zero": round(float(st["p_zero"]), 2),
           "q10": r4(dm + st["dq10"]), "q50": r4(dm), "q90": r4(dm + st["dq90"])}
    return json.dumps(out)

def build_market(mid):
    meta = market_meta(mid)
    m = MarketOrders(mid, meta=meta)
    pts = WORK["points"].filter(pl.col("market_id") == mid)
    out_eval, out_int, out_roll, out_demo = [], [], [], []
    for row in pts.to_dicts():
        t = int(row["t"]); nh = int(row["n_hist"])
        actor = None if row["kind"] == "control" else {"wallet": int(row["wallet"]), "pos": row["w_pos"], "n_prev": int(row["w_n_prev"]), "flow": row["w_flow"]}
        tg_all = {d: WORK[f"tg_{d}"].get(row["point_id"]) for d in DELTAS}
        for delta in DELTAS:
            tg = tg_all[delta]
            n_end = int(tg["n_end"])
            own = [[int(a), int(b), float(c)] for a, b, c in zip(tg["own"]["dt"], tg["own"]["d"], tg["own"]["q"])]
            joint = [[int(m.ts[i] - t), int(m.wallet[i]), int(m.sign[i]), float(m.usd[i])] for i in range(nh, n_end)]
            target = {k: _nan(tg[k]) for k in TARGET_KEYS}
            target["mid_t"] = row["mid"]
            base = {"point_id": row["point_id"], "split": row["split"], "kind": row["kind"], "band": row["band"], "rho_bin": row["rho_bin"], "delta": delta,
                    "market_id": mid, "t": t, "d": int(row["d"]), "qD": row["qD"], "target": target}
            if row.get("use_eval"):
                for task in ["J", "S"]:
                    for K in KS:
                        s = make_sample(m, meta, row, task, delta, own, joint, actor, K)
                        out_eval.append({**base, "sample_id": s["sample_id"], "task": task, "K": K, "prompt": ser.render(s)})
            if row.get("use_demo") and delta == "1h":
                for task in ["J", "S"]:
                    for K in KS:
                        s = make_sample(m, meta, row, task, delta, own, joint, actor, K)
                        out_demo.append({"sample_id": s["sample_id"], "point_id": row["point_id"], "market_id": mid, "task": task, "K": K, "delta": delta,
                                         "prompt": ser.render(s), "target": demo_target_text(tg, row, delta), "target_fields": target})
            if row.get("use_int") and delta == "1h":
                rho = float(tg["rho"]); d = int(row["d"])
                r0 = d * float(curve_lookup(WORK["curve"], row["band"], "1h", np.array([rho]))[0])
                r_scaled = d * float(curve_lookup(WORK["curve"], row["band"], "1h", np.array([rho / 10]))[0])
                variants = {"factual": (own, 0.0), "null": (own, 0.0), "remove": ([], -r0), "reverse": ([[a, -b, c] for a, b, c in own], -2 * r0),
                            "scale": ([[a, b, c / 10] for a, b, c in own], r_scaled - r0)}
                for var, (own_v, ref) in variants.items():
                    for K in KS:
                        s = make_sample(m, meta, row, "S", "1h", own_v, joint, actor, K)
                        s["sample_id"] = f"{row['point_id']}|S|1h|{var}"
                        out_int.append({**base, "sample_id": s["sample_id"], "task": "S", "K": K, "variant": var, "ref_response": ref, "r0": r0, "rho": rho, "prompt": ser.render(s)})
        if row.get("use_roll"):
            w = int(row["wallet"])
            for k in range(ROLL_STEPS):
                tk = t + k * HOUR
                nhk = nh if k == 0 else int(np.searchsorted(m.ts, tk, side="left"))
                st = m.state(tk, nhk, w)
                win = m.window(tk, nhk, HOUR, w)
                srow = {**{kk: st[kk] for kk in STATE_KEYS}, "t": tk, "n_hist": nhk, "point_id": f"{row['point_id']}|k{k}", "wallet": w,
                        "w_pos": st["w_pos"], "w_n_prev": st["w_n_prev"], "w_flow": st["w_flow"]}
                own = [[o[0], o[1], o[2]] for o in win["own"]]
                joint = [[int(m.ts[i] - tk), int(m.wallet[i]), int(m.sign[i]), float(m.usd[i])] for i in range(nhk, win["n_end"])]
                actor_k = {"wallet": w, "pos": st["w_pos"], "n_prev": st["w_n_prev"], "flow": st["w_flow"]}
                e = win["end"]
                dm = (e["mid"] - st["mid"]) if (e["mid"] is not None and st["mid"] is not None) else None
                for K in KS:
                    s = make_sample(m, meta, srow, "S", "1h", own, joint, actor_k, K)
                    out_roll.append({"chain_id": row["point_id"], "step": k, "sample_id": s["sample_id"], "split": row["split"], "market_id": mid, "t": tk, "task": "S", "K": K, "delta": "1h",
                                     "state": {kk: st[kk] for kk in STATE_KEYS + ["w_pos", "w_n_prev", "w_flow"]}, "own": own,
                                     "target": {"p_buy_end": e["p_buy"], "p_sell_end": e["p_sell"], "mid_end": e["mid"], "dmid": dm, "is_zero": (dm == 0) if dm is not None else None,
                                                "N_o": win["N_o"], "F_o": win["F_o"], "V_o": win["V_o"], "own_N": win["own_N"], "own_dpos": win["own_dpos"], "D_end": e["D"], "w_pos_end": e["w_pos"], "mid_t": st["mid"]},
                                     "prompt": ser.render(s)})
    return out_eval, out_int, out_roll, out_demo

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    pts = pl.read_parquet(f"{ROOT}/points.parquet").join(pl.read_parquet(f"{ROOT}/splits.parquet").select("point_id", "split"), on="point_id", how="inner")
    sel = []
    for split in SPLITS:
        for kind, n in [("anchor", N_ANCHOR), ("control", N_CONTROL)]:
            ids = pts.filter((pl.col("split") == split) & (pl.col("kind") == kind))["point_id"].to_numpy()
            sel.append(pl.DataFrame({"point_id": rng.choice(ids, size=min(n, len(ids)), replace=False), "use_eval": True}))
    use_eval = pl.concat(sel)
    ints = []
    for split in ["test_id", "test_ood"]:
        ids = pts.filter((pl.col("split") == split) & (pl.col("kind") == "anchor") & (pl.col("band") == "mid"))["point_id"].to_numpy()
        ints.append(pl.DataFrame({"point_id": rng.choice(ids, size=min(N_INTERV, len(ids)), replace=False), "use_int": True}))
    use_int = pl.concat(ints)
    ids = pts.filter((pl.col("split") == "test_id") & (pl.col("kind") == "anchor"))["point_id"].to_numpy()
    use_roll = pl.DataFrame({"point_id": rng.choice(ids, size=N_ROLL, replace=False), "use_roll": True})
    # few-shot demonstrations: train mid-band anchors from markets that appear in no eval / intervention / rollout file (seed 1)
    used_markets = set(pts.filter(pl.col("point_id").is_in(pl.concat([use_eval["point_id"], use_int["point_id"], use_roll["point_id"]])))["market_id"].to_list())
    cand = pts.filter((pl.col("split") == "train") & (pl.col("kind") == "anchor") & (pl.col("band") == "mid") & ~pl.col("market_id").is_in(list(used_markets))).sort("point_id")
    rng_demo = np.random.default_rng(1)
    demo_ids = []
    demo_markets = set()
    order = rng_demo.permutation(cand.height)
    for i in order:
        pid, mk = cand["point_id"][int(i)], cand["market_id"][int(i)]
        if mk in demo_markets:
            continue
        demo_ids.append(pid); demo_markets.add(mk)
        if len(demo_ids) == N_DEMO:
            break
    use_demo = pl.DataFrame({"point_id": demo_ids, "use_demo": True})
    work_pts = pts.join(use_eval, on="point_id", how="left").join(use_int, on="point_id", how="left").join(use_roll, on="point_id", how="left").join(use_demo, on="point_id", how="left")
    work_pts = work_pts.filter(pl.col("use_eval") | pl.col("use_int") | pl.col("use_roll") | pl.col("use_demo")).with_columns(pl.col("use_eval", "use_int", "use_roll", "use_demo").fill_null(False))
    ids = work_pts["point_id"]
    stats = pl.read_parquet(f"{ROOT}/response_curve_stats.parquet")
    work = {"points": work_pts, "curve": pl.read_parquet(f"{ROOT}/response_curve.parquet"),
            "stats": {(r["kind"], r["band"], r["rho_bin"], r["delta"]): r for r in stats.to_dicts()}}
    for d in DELTAS:
        tg = pl.read_parquet(f"{ROOT}/targets_{d}.parquet").filter(pl.col("point_id").is_in(ids))
        work[f"tg_{d}"] = {r["point_id"]: r for r in tg.to_dicts()}
    mids = work_pts["market_id"].unique().sort().to_list()
    print(f"selected points {work_pts.height:,} in {len(mids)} markets")
    with mp.get_context("spawn").Pool(48, initializer=_init, initargs=(work,)) as pool:
        res = pool.map(build_market, mids, chunksize=2)
    ev = [r for x in res for r in x[0]]; it = [r for x in res for r in x[1]]; ro = [r for x in res for r in x[2]]; dm = [r for x in res for r in x[3]]
    E = f"{ROOT}/eval"; os.makedirs(E, exist_ok=True)
    lines = ["# P7 Polymarket eval sets (protocol v3: K in {0, 3}, Task S = next state only)", ""]
    for K in KS:
        for split in SPLITS:
            rows = [r for r in ev if r["split"] == split and r["K"] == K]
            with open(f"{E}/eval_{split}_K{K}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            na = len({r["point_id"] for r in rows if r["kind"] == "anchor"}); nc = len({r["point_id"] for r in rows if r["kind"] == "control"})
            lines.append(f"- eval_{split}_K{K}.jsonl: {len(rows):,} samples ({na} anchors + {nc} controls) x 3 horizons x 2 tasks; mean prompt chars {np.mean([len(r['prompt']) for r in rows]):.0f}")
    idx = pl.DataFrame([{k: v for k, v in r.items() if k not in ("prompt", "target", "K")} | {f"y_{k}": v for k, v in r["target"].items()} for r in ev if r["K"] == KS[0]])
    idx.write_parquet(f"{E}/eval_index.parquet")
    for K in KS:
        with open(f"{E}/interventions_K{K}.jsonl", "w") as f:
            for r in it:
                if r["K"] == K:
                    f.write(json.dumps(r) + "\n")
        with open(f"{E}/rollout_K{K}.jsonl", "w") as f:
            for r in ro:
                if r["K"] == K:
                    f.write(json.dumps(r) + "\n")
        for task in ["J", "S"]:
            with open(f"{E}/demos_{task}_K{K}.jsonl", "w") as f:
                for r in dm:
                    if r["K"] == K and r["task"] == task:
                        f.write(json.dumps(r) + "\n")
    it0 = [r for r in it if r["K"] == KS[0]]
    pl.DataFrame([{k: v for k, v in r.items() if k not in ("prompt", "target", "K")} | {f"y_{k}": v for k, v in r["target"].items()} for r in it0]).write_parquet(f"{E}/interventions.parquet")
    refs = pl.DataFrame([{"variant": r["variant"], "ref": r["ref_response"], "split": r["split"]} for r in it0]).group_by("split", "variant").agg(pl.len(), pl.col("ref").mean().alias("mean_ref"), pl.col("ref").abs().mean().alias("mean_abs_ref")).sort("split", "variant")
    lines += [f"- interventions_K{{0,3}}.jsonl: {len(it0):,} rows each ({len({r['point_id'] for r in it0})} mid-band anchors x 5 variants, Task S, 1h)", refs.to_pandas().to_markdown(index=False, floatfmt=".5f"),
              f"- rollout_K{{0,3}}.jsonl: {len([r for r in ro if r['K'] == KS[0]]):,} rows each ({len({r['chain_id'] for r in ro})} chains x {ROLL_STEPS} hourly steps, test_id, true-log protocol)",
              f"- demos_{{J,S}}_K{{0,3}}.jsonl: {N_DEMO} demonstrations each from train markets {sorted({r['market_id'] for r in dm})} (points {sorted({r['point_id'] for r in dm})}), Delta=1h, mid band",
              "", "example prompt (Task S, 1h, K=3):", "```", next(r["prompt"] for r in ev if r["task"] == "S" and r["delta"] == "1h" and r["kind"] == "anchor" and r["K"] == 3), "```",
              "example prompt (Task S, 1h, K=0):", "```", next(r["prompt"] for r in ev if r["task"] == "S" and r["delta"] == "1h" and r["kind"] == "anchor" and r["K"] == 0), "```",
              "example demonstration target (S, K=0):", "```", next(r["target"] for r in dm if r["task"] == "S" and r["K"] == 0), "```",
              f"- runtime {time.time() - t0:.0f}s; outputs under `{E}/`"]
    open("out/build/pm_eval.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
