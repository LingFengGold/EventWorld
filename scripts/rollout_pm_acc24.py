"""Polymarket rollouts scored by Acc at 24 h: does the mid-price fall, stay, or rise between the chain start and hour 24?
Compares the 24-step closed-loop rollout with the direct one-shot 24 h forecast, both read with the Acc rule of the
single-step results (Table 2): the class is "stays" if the predicted probability of no change is >= 0.5, otherwise the sign of the
predicted median change, or of the change implied by the predicted buy and sell prices when the median change is zero.
  direct 24 h : the rule applied to the single 24 h prediction (p_zero, q50, p_buy, p_sell), exactly as in Table 2.
  rollout     : the predicted 24 h change is the rolled-out mid at hour 24 minus the start mid (mid + q50 fed back every hour, clipped
                to [0, 1], as scored in rollout_pm.py); the probability of no change over 24 h is the product of the 24 hourly p_zero;
                when the rolled-out mid never moved, the direction falls back to the buy and sell prices predicted at hour 24.
Unparsed predictions and broken closed-loop chains count as wrong, as in Table 2. True class: the recorded mid at hour 24 against the
start mid (stays only if equal). Paired bootstrap over chains (2,000 resamples) for rollout minus direct.
Reads the raw outputs out/rollout/polymarket_rollout/<tag>/polymarket_{open,closed_step*}.out.jsonl (kept out of git) and writes
out/rollout/polymarket_rollout/<tag>/acc24.json for every tag given (default: every tag whose raw outputs exist here).
Usage: rollout_pm_acc24.py [--tags sft8b sft8b_J]
"""
import argparse, glob, json, os
import numpy as np
E = "data/processed/polymarket/eval"; R = "out/rollout/polymarket_rollout"; STEPS = 24; N = 500; TOL = 1e-9


def sgn(x):
    return 0 if abs(x) < TOL else int(np.sign(x))


def load(path):
    return {r["sample_id"]: (r.get("pred") or {}) for r in map(json.loads, open(path))}


def parsed(p):
    return all(p.get(k) is not None for k in ("q10", "q50", "q90"))


def chains(task):
    ch = {}
    for r in map(json.loads, open(f"{E}/rollout_K0.jsonl" if task == "S" else f"{E}/rollout_J_K0.jsonl")):
        ch.setdefault(r["chain_id"], []).append(r)
    return {c: sorted(ch[c], key=lambda r: r["step"]) for c in sorted(ch)[:N]}   # the 500 chains of rollout_pm.py


def score(tag):
    task = "J" if tag.endswith("_J") else "S"
    P = load(f"{R}/{tag}/polymarket_open.out.jsonl")
    for k in range(1, STEPS + 1):
        P.update(load(f"{R}/{tag}/polymarket_closed_step{k}.out.jsonl"))
    yt, yd, yc = [], [], []
    for c, ch in chains(task).items():
        m0, y24 = ch[0]["state"]["mid"], ch[-1]["target"]["mid_end"]
        yt.append(sgn(y24 - m0))
        # direct 24 h forecast, Table 2 rule (quantiles sorted, unparsed = wrong)
        p = P.get(f"{c}|direct24") or {}
        if parsed(p) and p.get("p_zero") is not None:
            q50 = sorted([p["q10"], p["q50"], p["q90"]])[1]
            pm = (p["p_buy"] + p["p_sell"]) / 2 - m0 if p.get("p_buy") is not None and p.get("p_sell") is not None else 0.0
            yd.append(0 if p["p_zero"] >= 0.5 else (int(np.sign(q50)) if q50 != 0 else sgn(pm)))
        else:
            yd.append(None)
        # closed-loop rollout
        mid, pz, last = (ch[0]["state"]["p_buy"] + ch[0]["state"]["p_sell"]) / 2, 1.0, None
        for k in range(STEPS):
            last = P.get(f"{c}|{k}|closed") or {}
            if not parsed(last):
                last = None; break
            mid = min(max(mid + last["q50"], 0.0), 1.0); pz *= last["p_zero"] if last.get("p_zero") is not None else 0.0
        if last is None:
            yc.append(None); continue
        pm = (last["p_buy"] + last["p_sell"]) / 2 - m0 if last.get("p_buy") is not None and last.get("p_sell") is not None else 0.0
        yc.append(0 if pz >= 0.5 else (sgn(mid - m0) if sgn(mid - m0) != 0 else sgn(pm)))
    yt = np.array(yt)
    hd = np.array([v is not None and v == y for v, y in zip(yd, yt)], float)
    hc = np.array([v is not None and v == y for v, y in zip(yc, yt)], float)
    rng = np.random.default_rng(0)
    bs = [(hc[i] - hd[i]).mean() for i in (rng.integers(0, len(yt), len(yt)) for _ in range(2000))]
    share = lambda v, x: 100 * float(np.mean([a == x for a in v]))
    return {"tag": tag, "task": task, "n": int(len(yt)), "acc_closed": 100 * hc.mean(), "acc_direct": 100 * hd.mean(),
            "closed_minus_direct": 100 * (hc - hd).mean(), "ci95": [100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))],
            "true_share": {"fall": share(yt, -1), "stay": share(yt, 0), "rise": share(yt, 1)},
            "pred_stay_closed": share(yc, 0), "pred_stay_direct": share(yd, 0),
            "parsed_closed": 100 * float(np.mean([v is not None for v in yc])), "parsed_direct": 100 * float(np.mean([v is not None for v in yd]))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*"); a = ap.parse_args()
    tags = a.tags or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{R}/*/polymarket_closed_step24.out.jsonl"))
    for t in tags:
        r = score(t)
        json.dump(r, open(f"{R}/{t}/acc24.json", "w"), indent=1)
        print(f"{t:14s} Task {r['task']}  Acc rollout {r['acc_closed']:.1f}  direct {r['acc_direct']:.1f}  diff {r['closed_minus_direct']:+.1f} "
              f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  predicts stay: rollout {r['pred_stay_closed']:.1f} direct {r['pred_stay_direct']:.1f}  "
              f"true fall/stay/rise {r['true_share']['fall']:.1f}/{r['true_share']['stay']:.1f}/{r['true_share']['rise']:.1f}")


if __name__ == "__main__":
    main()
