"""L1 multi-step consistency (1/3): teacher-forced closed-loop chains in Diplomacy.

500 chains of 5 consecutive test transitions (data/processed/diplomacy/rollout_chains.parquet), Task J, K=0.
  open loop   : at step k the model gets the TRUE board and the true joint orders of step k (single-step accuracy on the chain's transitions).
  closed loop : at step k the model gets ITS OWN predicted board from step k-1 and the true joint orders of step k. The phase name is taken
                from the record (teacher-forced), so the true orders are always issued for the right phase type.
A prediction that cannot be parsed ends the chain: all later closed steps score 0 and are not queried.
Metrics per step k: EM (phase_exact) and unit F1 for both loops; drift ratio D_k = EM_closed(k) / EM_open(k); first divergence step
(first closed step with EM = 0); board validity of the closed board; void rate = share of the true orders of step k+1 that are not legal on
the model's predicted board (reference: the same share on the true board, which is ~0).
Task S (--task S): one power per chain (a power with units at the start, fixed by a hash of the chain id) is the actor; every step gives only
its own true orders (the other powers' orders are hidden), and the void rate counts only that power's next orders.
Usage: rollout_dip.py --model api:gpt-6-luna|<local dir> --tag <name> [--task J|S] [--n 500]
"""
import argparse, json, os, re, sys
import polars as pl, pandas as pd, numpy as np
from eventworld.dip_serialize import render_task_j, render_task_s, parse_state
import hashlib
from eventworld.metrics_dip import score_state, split_target
from eventworld.dip_engine import legal
from eventworld.rollout_predict import predict
D = "data/processed/diplomacy"


def derive_builds(st):
    """Adjustment-phase build/disband counts follow from the board by rule (centers minus units; builds only on unoccupied owned home centers),
    so they are derived here rather than taken from the model's optional BUILDS line."""
    occ = {u.split()[1].split("/")[0] for us in st["units"].values() for u in us}
    out = {}
    for p in set(st["units"]) | set(st["centers"]):
        n = len(st["centers"].get(p, [])) - len(st["units"].get(p, []))
        homes = [h for h in (st.get("homes") or {}).get(p, []) if h in st["centers"].get(p, []) and h not in occ] if n > 0 else []
        out[p] = {"count": n, "homes": homes}
    return out


def normalize_dislodged(txt):
    """Zero-shot models that never saw the dislodged-unit notation (*A LOC>opts) invent their own; map the two common variants,
    'A LOC*' and 'A LOC (dislodged)', to the standard '*A LOC' (no retreat options) so the board is not rejected for them."""
    txt = re.sub(r"\b([AF]) (\S+?)\s*\((?:dislodged|DISLODGED)[^)]*\)", r"*\1 \2", txt)
    return re.sub(r"(?<![*\w])([AF]) ([A-Z]{3}(?:/[A-Z]{2})?)\*", r"*\1 \2", txt)


def pred_to_state(text, name, homes):
    """Model output -> engine state dict. Retreat options come from the model's own dislodged-unit annotations; build counts are derived."""
    ps = parse_state(normalize_dislodged(split_target(text or "")[1]))
    if not any(ps["units"].values()):
        return None
    st = {"name": name, "units": {p: sorted(u) for p, u in ps["units"].items()}, "centers": {p: sorted(c) for p, c in ps["centers"].items()},
          "retreats": ps["retreats"], "builds": ps["builds"], "homes": homes}
    return st


def for_phase(st, phase_name):
    """The closed-loop board relabelled with the phase it is used in; adjustment phases get rule-derived build counts."""
    st = dict(st); st["name"] = phase_name
    if phase_name.endswith("A"):
        st["builds"] = derive_builds(st)
    return st


def void_rate(state, orders, map_name, rules):
    """Share of orders that are not legal on `state` (orders for units that do not exist count as void)."""
    flat = [o for ol in orders.values() for o in (ol or [])]
    if not flat:
        return np.nan
    try:
        L = {o for v in legal(state, map_name, rules).values() for o in v}
    except Exception:
        return 1.0
    norm = lambda o: " ".join(o.replace(" VIA", "").split())
    Ln = {norm(o) for o in L}
    return float(np.mean([norm(o) not in Ln for o in flat]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--concurrency", type=int, default=16); ap.add_argument("--task", default="J", choices=["J", "S"]); a = ap.parse_args()
    OUT = f"out/rollout/diplomacy_chains/{a.tag}"; os.makedirs(OUT, exist_ok=True); log = f"{OUT}/infer.log"
    chains = pd.read_parquet(f"{D}/rollout_chains.parquet").head(a.n)   # 500 chains of 5 consecutive test transitions (pipeline/dip_build_samples.py)
    chains["ids"] = chains.sample_ids.map(json.loads)
    ids = sorted({s for c in chains.ids for s in c})
    T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(ids)).collect().to_pandas().set_index("sample_id")
    rules_of = lambda r: ["NO_CHECK"] if r["rules_no_check"] else []
    cur = {c.chain_id: json.loads(T.loc[c.ids[0]]["state_json"]) for c in chains.itertuples()}
    def actor(cid):
        alive = sorted(p for p, u in cur0[cid]["units"].items() if u)
        return alive[int(hashlib.sha1(cid.encode()).hexdigest()[:8], 16) % len(alive)]
    cur0 = dict(cur); ACT = {c.chain_id: actor(c.chain_id) for c in chains.itertuples()}
    def prompt(r, st, orders, cid):
        if a.task == "J":
            return render_task_j(r["map"], rules_of(r), 7, st, orders)
        return render_task_s(r["map"], rules_of(r), 7, None, st, ACT[cid], orders.get(ACT[cid], []))
    mine = lambda o, cid: o if a.task == "J" else {ACT[cid]: o.get(ACT[cid], [])}   # closed-loop board per chain (None = dead)
    recs = []
    for k in range(5):
        rows_open, rows_closed = [], []
        for c in chains.itertuples():
            sid = c.ids[k]; r = T.loc[sid]; orders = json.loads(r["orders_json"])
            rows_open.append({"sample_id": f"{c.chain_id}|{k}|open", "task": a.task, "prompt": prompt(r, json.loads(r["state_json"]), orders, c.chain_id)})
            if k > 0 and cur[c.chain_id] is not None:
                st = for_phase(cur[c.chain_id], r["phase_name"])
                rows_closed.append({"sample_id": f"{c.chain_id}|{k}|closed", "task": a.task, "prompt": prompt(r, st, orders, c.chain_id)})
        P = predict(a.model, rows_open + rows_closed, f"{OUT}/diplomacy_step{k+1}", concurrency=a.concurrency, log=log)
        for c in chains.itertuples():
            sid = c.ids[k]; r = T.loc[sid]; true_ns = json.loads(r["next_state_json"]); homes = json.loads(r["state_json"]).get("homes")
            for mode in ("open", "closed"):
                key = f"{c.chain_id}|{k}|{'open' if (mode == 'open' or k == 0) else 'closed'}"   # step 1: closed input == open input
                alive = mode == "open" or k == 0 or cur[c.chain_id] is not None
                text = P.get(key, {}).get("text", "") if alive else ""
                s = score_state(split_target(text)[1], true_ns, r["map"]) if text else {"phase_exact": 0.0, "unit_f1": 0.0, "board_valid": 0.0, "power_exact_rate": 0.0}
                s.update(chain_id=c.chain_id, step=k + 1, mode=mode, alive=float(alive), phase_type=r["phase_type"])
                if mode == "closed":
                    nxt = pred_to_state(text, true_ns["name"], homes) if alive else None
                    if k + 1 < 5:   # void rate of the NEXT step's true orders on this predicted board, and on the true board as reference
                        r2 = T.loc[c.ids[k + 1]]; o2 = json.loads(r2["orders_json"])
                        if nxt is not None:
                            st2 = for_phase(nxt, r2["phase_name"]); s["void_next"] = void_rate(st2, mine(o2, c.chain_id), r2["map"], rules_of(r2))
                        else:
                            s["void_next"] = 1.0
                        s["void_next_true"] = void_rate(json.loads(r2["state_json"]), mine(o2, c.chain_id), r2["map"], rules_of(r2))
                    cur[c.chain_id] = nxt
                recs.append(s)
        print(f"step {k+1} done", flush=True)
    df = pd.DataFrame(recs); df.to_parquet(f"{OUT}/per_step.parquet", index=False)
    summarize(df, OUT, a.tag, a.task)


def summarize(df, OUT, tag, task="J"):
    op, cl = df[df["mode"] == "open"], df[df["mode"] == "closed"]
    g = pd.DataFrame({"EM_open": op.groupby("step").phase_exact.mean(), "EM_closed": cl.groupby("step").phase_exact.mean(),
                      "F1_open": op.groupby("step").unit_f1.mean(), "F1_closed": cl.groupby("step").unit_f1.mean(),
                      "valid_closed": cl.groupby("step").board_valid.mean(), "alive": cl.groupby("step").alive.mean(),
                      "void_next_closed": cl.groupby("step").void_next.mean(), "void_next_true": cl.groupby("step").void_next_true.mean()})
    g["D_k"] = g.EM_closed / g.EM_open.replace(0, np.nan)
    first = cl[cl.phase_exact == 0].groupby("chain_id").step.min()
    n = cl.chain_id.nunique()
    dist = first.value_counts().reindex(range(1, 6), fill_value=0) / n
    lines = [f"# Diplomacy teacher-forced chains: {tag} (Task {task}, K=0, {n} chains x 5 steps)", "", g.round(3).to_markdown(), "",
             "first divergence step (share of chains): " + ", ".join(f"{k}: {v:.3f}" for k, v in dist.items()) + f", never: {1 - len(first) / n:.3f}"]
    open(f"{OUT}/tables.md", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
