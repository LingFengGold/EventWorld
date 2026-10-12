"""Agent usefulness (1/2): the world model as a pre-execution verifier in Diplomacy, following Word2World Section 7.1.

200 starting boards (movement phases of the 7p common test set; same construction as the CR starts), one agent power per start. A
trajectory is H = 10 phases in the true engine. The opponents always play DipNet SL greedy (deterministic). At every movement phase the
agent draws B candidate order sets for its own power: candidate 0 is DipNet SL greedy (the no-verification baseline), candidates 1..B-1
are DipNet SL samples at temperature 0.5 (duplicates removed). A scorer predicts the board after the phase for each candidate, and the
candidate with the highest predicted value is executed in the true engine. Retreat and adjustment phases are played greedily.

Scorers
  engine   : the true engine with the opponents' actual (greedy) orders. The opponents are deterministic, so this is exact one-phase
             lookahead: the upper bound of any world model used this way.
  WM, J    : the world model given the candidate and the opponents' orders (it only adjudicates; Task J prompt, K = 0).
  WM, S    : the world model given only the candidate (Task S prompt, K = 0); it must imagine the opponents. This is how an actor would
             use an event world model.
B = 1 is the greedy baseline (no verification), identical for every scorer.

Value of a board for the agent: projected supply centers (owned centers not occupied by another power's unit, plus unowned centers
occupied by the agent's units; after a fall movement this is the ownership the adjustment phase will record), plus 0.1 per agent unit
that is not dislodged (tie-break). An unparseable or physically invalid predicted board scores -inf; if every candidate does, greedy is
played.
Metrics per (scorer, B): mean dSC after H phases (true engine), success rate (dSC >= 0), paired difference to B = 1 with its standard
error, and one-phase decision regret (engine value of the best candidate minus engine value of the chosen one, averaged over movement
phases) together with the share of phases in which the scorer picked the engine-optimal candidate.

Optimisation pressure: each run also records, per movement phase, the scorer's predicted one-phase gain of the chosen candidate
(predicted value minus the true value of the current board) and the realised gain in the engine; over-estimate = predicted - realised.
Usage: agent_verifier_dip.py --scorer engine --tag engine                       (CPU only; also writes the B = 1 baseline)
       agent_verifier_dip.py --scorer <model dir> --tag sft1.7b --modes S,J     (one GPU through eventworld/rollout_predict.py)
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from eventworld.dip_engine import step, is_valid_board, _seed_game, _map
from eventworld.dip_serialize import render_task_j, render_task_s
from eventworld.rollout_predict import predict
from rollout_dip import pred_to_state, for_phase
from rollout_cr import starts as cr_starts, sc
OUT = "out/agent/diplomacy_verifier"; H = 10; TEMP = 0.5


def value(st, p):
    """Projected supply centers of p on board st (+0.1 per non-dislodged unit)."""
    scs = _map("standard").scs
    occ = {}
    for q, us in (st.get("units") or {}).items():
        for u in us or []:
            if not u.startswith("*"):
                occ[u.split()[1].split("/")[0]] = q
    own = set((st.get("centers") or {}).get(p, []) or [])
    proj = sum(1 for c in own if occ.get(c, p) == p) + sum(1 for c in scs if c not in own and occ.get(c) == p)
    units = sum(1 for u in (st.get("units") or {}).get(p, []) or [] if not u.startswith("*"))
    return proj + 0.1 * units


_DN = None
def _dipnet():
    global _DN
    if _DN is None:
        from eventworld.dipnet import dipnet_orders_batch
        _DN = dipnet_orders_batch
    return _DN


def dn(st, m, rules, powers=None, temperature=0.0):
    return _dipnet()(_seed_game(st, m, rules).get_state(), powers=powers, map_name=m, rules=rules, temperature=temperature)


def prepare(job):
    """CPU part of one movement phase for one trajectory: opponents' orders, candidates, true next board and value of each candidate."""
    st, m, rules, agent, B = job["state"], job["map"], job["rules"], job["agent"], job["B"]
    base = dn(st, m, rules)
    cands, seen = [], set()
    for i in range(max(B, 1) * 3):   # a few extra draws so that B distinct candidates are usually reached
        o = base.get(agent, []) if i == 0 else dn(st, m, rules, powers=[agent], temperature=TEMP).get(agent, [])
        k = tuple(sorted(o))
        if k not in seen:
            seen.add(k); cands.append(list(o))
        if len(cands) >= B:
            break
    nexts, vals = [], []
    for c in cands:
        ns, _ = step(st, dict(base, **{agent: c}), m, rules)
        nexts.append(ns); vals.append(value(ns, agent))
    return {"base": base, "cands": cands, "nexts": nexts, "vals": vals}


def greedy_step(job):
    st, m, rules = job["state"], job["map"], job["rules"]
    return step(st, dn(st, m, rules), m, rules)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scorer", required=True, help="'engine' or a world-model path / api:<deployment>")
    ap.add_argument("--tag", required=True); ap.add_argument("--modes", default="S,J"); ap.add_argument("--budgets", default="2,4,10")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    S = cr_starts(a.n); budgets = [int(b) for b in a.budgets.split(",")]
    engine = a.scorer == "engine"
    modes = ["E"] if engine else a.modes.split(",")
    if engine:
        budgets = sorted(set([1] + budgets))
    O = f"{OUT}/{a.tag}"; os.makedirs(O, exist_ok=True); log = f"{O}/infer.log"
    T = {(s["sid"], mo, B): {"s": s, "mode": mo, "B": B, "state": s["state"], "done": False, "regret": [], "opt": [], "greedy": [], "ncand": [], "invalid": 0, "pg": [], "tg": []}
         for s in S for mo in modes for B in budgets}
    pool = Pool(a.workers)
    for h in range(H):
        act = [k for k, t in T.items() if not t["done"]]
        for k in act:
            if T[k]["state"]["name"][-1:] not in ("M", "R", "A"):
                T[k]["done"] = True
        act = [k for k in act if not T[k]["done"]]
        if not act:
            break
        mv = [k for k in act if T[k]["state"]["name"].endswith("M")]
        other = [k for k in act if k not in set(mv)]
        jobs = lambda ks: [{"state": T[k]["state"], "map": T[k]["s"]["map"], "rules": T[k]["s"]["rules"], "agent": T[k]["s"]["agent"], "B": T[k]["B"]} for k in ks]
        for k, ns in zip(other, pool.map(greedy_step, jobs(other), chunksize=4)):
            T[k]["state"] = ns
        prep = dict(zip(mv, pool.map(prepare, jobs(mv), chunksize=2)))
        choice, pred = {}, {}
        if engine:
            for k, p in prep.items():
                choice[k] = int(np.argmax(p["vals"])); pred[k] = p["vals"][choice[k]]
        else:
            rows = []
            for k, p in prep.items():
                if len(p["cands"]) == 1:
                    continue
                s, st = T[k]["s"], T[k]["state"]
                for i, c in enumerate(p["cands"]):
                    if T[k]["mode"] == "J":
                        pr = render_task_j(s["map"], s["rules"], 7, st, dict(p["base"], **{s["agent"]: c}))
                    else:
                        pr = render_task_s(s["map"], s["rules"], 7, None, st, s["agent"], c)
                    rows.append({"sample_id": f"{s['sid']}|{T[k]['mode']}|{T[k]['B']}|{i}", "task": T[k]["mode"], "prompt": pr})
            P = predict(a.scorer, rows, f"{O}/diplomacy_h{h}", concurrency=a.concurrency, log=log) if rows else {}
            for k, p in prep.items():
                s = T[k]["s"]; scores = []
                for i in range(len(p["cands"])):
                    text = P.get(f"{s['sid']}|{T[k]['mode']}|{T[k]['B']}|{i}", {}).get("text", "") if len(p["cands"]) > 1 else ""
                    first = text.strip().splitlines()[0] if text.strip() else ""
                    name = first.split()[1] if first.startswith("NEXT ") and len(first.split()) > 1 else None
                    ns = pred_to_state(text, name, T[k]["state"].get("homes")) if name else None
                    ok = ns is not None
                    if ok:
                        try:
                            ok = is_valid_board(ns, s["map"])
                        except Exception:
                            ok = False
                    scores.append(value(ns, s["agent"]) if ok else -np.inf)
                    T[k]["invalid"] += (not ok) and len(p["cands"]) > 1
                choice[k] = int(np.argmax(scores)) if np.isfinite(max(scores)) else 0
                pred[k] = scores[choice[k]] if np.isfinite(scores[choice[k]]) else np.nan
        for k, p in prep.items():
            c = choice[k]; best = max(p["vals"])
            T[k]["regret"].append(best - p["vals"][c]); T[k]["opt"].append(p["vals"][c] >= best - 1e-9)
            T[k]["greedy"].append(c == 0); T[k]["ncand"].append(len(p["cands"]))
            v0 = value(T[k]["state"], T[k]["s"]["agent"])   # predicted vs realised one-phase gain of the chosen candidate (optimisation pressure)
            if len(p["cands"]) > 1 and np.isfinite(pred.get(k, np.nan)):
                T[k]["pg"].append(pred[k] - v0); T[k]["tg"].append(p["vals"][c] - v0)
            T[k]["state"] = p["nexts"][c]
        print(f"phase {h + 1}/{H}: {len(act)} active trajectories, {len(mv)} at a movement phase", flush=True)
    pool.close()
    recs = [{"sid": t["s"]["sid"], "agent": t["s"]["agent"], "mode": t["mode"], "B": t["B"], "dsc": sc(t["state"], t["s"]["agent"]) - sc(t["s"]["state"], t["s"]["agent"]),
             "regret": float(np.mean(t["regret"])) if t["regret"] else np.nan, "opt_rate": float(np.mean(t["opt"])) if t["opt"] else np.nan,
             "greedy_rate": float(np.mean(t["greedy"])) if t["greedy"] else np.nan, "ncand": float(np.mean(t["ncand"])) if t["ncand"] else np.nan,
             "invalid": t["invalid"], "pred_gain": float(np.mean(t["pg"])) if t["pg"] else np.nan,
             "true_gain": float(np.mean(t["tg"])) if t["tg"] else np.nan, "n_choices": len(t["pg"])} for t in T.values()]
    df = pd.DataFrame(recs); df.to_parquet(f"{O}/per_start.parquet", index=False)
    summarize(a.tag)


def summarize(tag):
    """Table for one scorer, paired against the B = 1 baseline written by the engine run."""
    df = pd.read_parquet(f"{OUT}/{tag}/per_start.parquet")
    base_p = f"{OUT}/engine/per_start.parquet"
    base = pd.read_parquet(base_p) if os.path.exists(base_p) else df
    base = base[base.B == 1][["sid", "dsc"]].rename(columns={"dsc": "dsc_b1"})
    lines = [f"# Diplomacy pre-execution verifier: {tag} (H={H}, opponents DipNet SL greedy, candidates = greedy + DipNet T={TEMP} samples)", "",
             "| Scorer | B | mean dSC | success (dSC>=0) | dSC − B=1 (SE) | one-phase regret | picks engine-optimal | picks greedy | candidates | invalid predicted boards | predicted gain | realised gain | over-estimate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    b1 = base.dsc_b1
    lines.append(f"| greedy (no verification) | 1 | {b1.mean():.2f} | {(b1 >= 0).mean():.3f} | – | – | – | – | 1 | – | – | – | – |")
    for (mo, B), g in df[df.B > 1].groupby(["mode", "B"]):
        g = g.merge(base, on="sid"); d = g.dsc - g.dsc_b1
        name = {"E": "engine (upper bound)", "J": "WM, Task J", "S": "WM, Task S"}[mo]
        lines.append(f"| {name} | {B} | {g.dsc.mean():.2f} | {(g.dsc >= 0).mean():.3f} | {d.mean():+.2f} ({d.std(ddof=1) / np.sqrt(len(d)):.2f}) | "
                     f"{g.regret.mean():.3f} | {g.opt_rate.mean():.3f} | {g.greedy_rate.mean():.3f} | {g.ncand.mean():.1f} | {int(g.invalid.sum())} | "
                     + ("– | – | – |" if "pred_gain" not in g else f"{g.pred_gain.mean():+.3f} | {g.true_gain.mean():+.3f} | {(g.pred_gain - g.true_gain).mean():+.3f} |"))
    open(f"{OUT}/{tag}/tables.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
