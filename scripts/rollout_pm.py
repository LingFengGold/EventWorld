"""L1 multi-step consistency (3/3): 24-step hourly rollouts in Polymarket (Task S, K=0, 1 h steps).

500 chains x 24 hourly steps from test_id anchors (data/processed/polymarket/eval/rollout_K0.jsonl, true-log protocol: every step shows the true
flow of the previous hour and the agent's own orders of the current hour).
  open loop   : the recorded prompt, i.e. the true prices at the start of every hour (single-step accuracy along the chain).
  closed loop : the prices in the STATE line are the model's own predictions from the previous step; everything else is recorded.
                Fed back: mid = previous prompt mid + predicted median change q50; spread = predicted p_buy - p_sell (clipped to [0, 1]).
  direct 24h  : one Task S prediction at the chain start with horizon 24 h and the agent's orders over the full 24 h.
Metrics per step k (price level at t0 + k h): MAE of the median, CRPS from the three quantiles, 80% interval coverage, for both loops;
closed/open MAE ratio; MAE(direct 24 h) / MAE(closed loop at step 24).
Task J (--task J): the same chains, states and targets, but each prompt lists every order in the hour (rollout_J_K0.jsonl, built by
pm_build_rollout_j.py); the direct 24 h prompt then lists every order in the 24 hours.
Usage: rollout_pm.py --model api:gpt-6-sol|<dir> --tag <name> [--task S|J] [--binned] [--n 500]
"""
import argparse, json, os, re, sys
import numpy as np, pandas as pd
from eventworld.metrics_pm import crps_from_quantiles
from eventworld.rollout_predict import predict
E = "data/processed/polymarket/eval"; OUT = "out/rollout/polymarket_rollout"; STEPS = 24
STATE_RE = re.compile(r"STATE p_buy \S+ p_sell \S+ mid \S+ spread \S+")


def with_prices(prompt, p_buy, p_sell):
    mid, spr = (p_buy + p_sell) / 2, p_buy - p_sell
    return STATE_RE.sub(f"STATE p_buy {p_buy:.4f} p_sell {p_sell:.4f} mid {mid:.4f} spread {spr:.4f}", prompt, count=1)


def direct_24h_prompt(chain):
    """Step-0 prompt turned into the 24 h Task S prompt: horizon, the agent's orders over all 24 steps, and the task sentence."""
    p = chain[0]["prompt"].replace("| horizon 1h", "| horizon 24h").replace("after 1 hour.", "after 24 hours.")
    own = [(k * 3600 + dt, d, q) for k, r in enumerate(chain) for dt, d, q in r["own"]]
    act = "ACTION A: none" if not own else f"ACTION A ({len(own)} orders, dt side usd): " + " | ".join(f"{dt}s {'BUY' if d > 0 else 'SELL'} ${q:.2f}" for dt, d, q in own)
    return re.sub(r"ACTION A[^\n]*", act, p, count=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--binned", action="store_true"); ap.add_argument("--task", default="S", choices=["S", "J"])
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--concurrency", type=int, default=16); a = ap.parse_args()
    O = f"{OUT}/{a.tag}"; os.makedirs(O, exist_ok=True); log = f"{O}/infer.log"
    rows = [json.loads(l) for l in open(f"{E}/rollout_K0.jsonl" if a.task == "S" else f"{E}/rollout_J_K0.jsonl")]
    chains = {}
    for r in rows:
        chains.setdefault(r["chain_id"], []).append(r)
    cids = sorted(chains)[:a.n]
    for c in cids:
        chains[c].sort(key=lambda r: r["step"])
    kw = dict(binned=a.binned, concurrency=a.concurrency, log=log)
    # open loop and direct 24 h: all prompts are known in advance, one batch
    batch = [{"sample_id": f"{c}|{k}|open", "task": a.task, "prompt": chains[c][k]["prompt"]} for c in cids for k in range(STEPS)]
    batch += [{"sample_id": f"{c}|direct24", "task": a.task, "prompt": direct_24h_prompt(chains[c]) if a.task == "S" else chains[c][0]["direct24_prompt"]} for c in cids]
    P_open = predict(a.model, batch, f"{O}/polymarket_open", **kw)
    # closed loop: step by step
    cur = {c: (chains[c][0]["state"]["p_buy"], chains[c][0]["state"]["p_sell"]) for c in cids}
    P_closed = {}
    for k in range(STEPS):
        b = []
        for c in cids:
            pb, ps = cur[c]
            if pb is None or ps is None:
                continue
            b.append({"sample_id": f"{c}|{k}|closed", "task": a.task, "prompt": chains[c][k]["prompt"] if k == 0 else with_prices(chains[c][k]["prompt"], pb, ps)})
        P = predict(a.model, b, f"{O}/polymarket_closed_step{k + 1}", **kw); P_closed.update(P)
        for c in cids:
            pr = P.get(f"{c}|{k}|closed")
            pb, ps = cur[c]
            if pr is None or pb is None or pr.get("q50") is None:
                cur[c] = (None, None); continue   # unparseable: the chain stops (scored as missing from here on)
            mid = min(max((pb + ps) / 2 + pr["q50"], 0.0), 1.0)
            spr = max(0.0, (pr.get("p_buy") or pb) - (pr.get("p_sell") or ps))
            cur[c] = (min(1.0, mid + spr / 2), max(0.0, mid - spr / 2))
        print(f"closed step {k + 1}/{STEPS} done", flush=True)
    recs = []
    for c in cids:
        ch = chains[c]
        mid_closed = (ch[0]["state"]["p_buy"] + ch[0]["state"]["p_sell"]) / 2 if ch[0]["state"]["p_buy"] is not None else np.nan
        for k in range(STEPS):
            y = ch[k]["target"]["mid_end"]
            for mode, P, base in (("open", P_open, ch[k]["state"]["mid"]), ("closed", P_closed, mid_closed)):
                pr = P.get(f"{c}|{k}|{mode}") or {}
                ok = all(pr.get(q) is not None for q in ("q10", "q50", "q90")) and base is not None and not np.isnan(base) and y is not None
                rec = {"chain_id": c, "step": k + 1, "mode": mode, "ok": ok}
                if ok:
                    q10, q50, q90 = base + pr["q10"], base + pr["q50"], base + pr["q90"]
                    rec.update(ae=abs(q50 - y), crps=float(crps_from_quantiles([q10], [q50], [q90], [y])[0]),
                               cover=float(q10 <= y <= q90), ae_nochange=abs(ch[0]["state"]["mid"] - y) if ch[0]["state"]["mid"] is not None else np.nan)
                if mode == "closed":
                    mid_closed = min(max(q50, 0.0), 1.0) if ok else np.nan
                recs.append(rec)
        pr = P_open.get(f"{c}|direct24") or {}
        y24, m0 = ch[-1]["target"]["mid_end"], ch[0]["state"]["mid"]
        if pr.get("q50") is not None and y24 is not None and m0 is not None:
            recs.append({"chain_id": c, "step": 24, "mode": "direct24", "ok": True, "ae": abs(m0 + pr["q50"] - y24),
                         "cover": float(m0 + pr["q10"] <= y24 <= m0 + pr["q90"])})
    df = pd.DataFrame(recs); df.to_parquet(f"{O}/per_step.parquet", index=False)
    g = df[df["mode"] != "direct24"].groupby(["step", "mode"]).agg(MAE=("ae", "mean"), CRPS=("crps", "mean"), cov80=("cover", "mean"), ok=("ok", "mean")).unstack("mode")
    g.columns = [f"{m}_{mode}" for m, mode in g.columns]
    g["MAE_closed/open"] = g.MAE_closed / g.MAE_open
    g["MAE_nochange_from_t0"] = df[(df["mode"] == "closed") & df.ok].groupby("step").ae_nochange.mean()
    d24 = df[df["mode"] == "direct24"]
    sel = [1, 2, 3, 6, 12, 18, 24]
    lines = [f"# Polymarket 24-step rollout: {a.tag} (Task {a.task}, K=0, 1 h steps, {len(cids)} chains)", "", g.loc[sel].round(4).to_markdown(), "",
             f"direct 24 h: MAE {d24.ae.mean():.4f}, cov80 {d24.cover.mean():.3f} (n={len(d24)}); MAE(direct 24 h) / MAE(closed step 24) = {d24.ae.mean() / g.loc[24, 'MAE_closed']:.3f}"]
    open(f"{O}/tables.md", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
