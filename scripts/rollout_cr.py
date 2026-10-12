"""L1 multi-step consistency (2/3): consistency ratio (CR) in Diplomacy, following Word2World's Real / WM / W2R protocol.

500 starting boards (movement phases of the 7p common test set, seed 0), one agent power per start (a power with units, chosen by seed).
Every power, agent and opponents, plays DipNet SL with greedy decoding. A trajectory is H = 10 phases.
  Real : the true engine, from the start board.
  WM   : everything inside the world model. At each phase DipNet orders all powers on the world model's board, and the world model (Task J, K=0)
         predicts the next board, including the next phase name. An unparseable or physically invalid board, or a board DipNet cannot play,
         ends the trajectory as a failure.
  W2R  : the agent's orders recorded in WM are replayed in the true engine (matched by phase name; a phase WM never reached gets no orders),
         while the opponents re-plan with DipNet on the true board.
Success: the agent's supply-center count after H phases minus at the start, dSC >= THRESH (default 0).
Metrics: success rates Real / WM / W2R; CR = W2R / Real; WM / Real (optimism); imagination gap = mean(dSC_WM - dSC_W2R) on trajectories WM
completed; mean dSC per setting; replay void rate = share of the agent's replayed orders that are illegal (or missing) on the true board.
Task S (--task S): inside WM the world model gets only the agent's orders (DipNet still chooses them); the opponents are no longer played by
DipNet but left to the world model, which is how an actor would use an event world model. Real and W2R are unchanged.
Usage: rollout_cr.py --real_only                      (engine only: base success rate; run first to check the threshold)
       rollout_cr.py --model api:gpt-6-sol|<dir> --tag <name>
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd, polars as pl
from eventworld.dipnet import dipnet_orders_batch
from eventworld.dip_engine import step, legal, _seed_game, is_valid_board
from eventworld.dip_serialize import render_task_j, render_task_s
from eventworld.rollout_predict import predict
from rollout_dip import pred_to_state, for_phase
D = "data/processed/diplomacy"; OUT = "out/rollout/diplomacy_cr"; H = 10


def starts(n=500):
    c = pd.read_parquet(f"{D}/eval_common_7p.parquet"); c = c[c.phase_type == "M"].sample(n=n, random_state=0)
    T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(c.sample_id.tolist())).select(
        "sample_id", "map", "rules_no_check", "state_json").collect().to_pandas().set_index("sample_id")
    out = []
    for sid in c.sample_id:
        st = json.loads(T.loc[sid]["state_json"])
        alive = sorted(p for p, u in st["units"].items() if u)
        agent = alive[int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) % len(alive)]
        out.append({"sid": sid, "map": T.loc[sid]["map"], "rules": ["NO_CHECK"] if T.loc[sid]["rules_no_check"] else [], "state": st, "agent": agent})
    return out


def sc(st, p):
    return len((st.get("centers") or {}).get(p, []) or [])


AGENT_MODEL = "sl"   # --agent_model rl: the agent's power plays DipNet RL (greedy) while the opponents keep DipNet SL (behaviour shift, W2W 5.2)


def dipnet(st, map_name, rules, agent=None):
    gs = _seed_game(st, map_name, rules).get_state()
    o = dipnet_orders_batch(gs, map_name=map_name, rules=rules, temperature=0.0)
    if agent is not None and AGENT_MODEL != "sl":
        o[agent] = dipnet_orders_batch(gs, powers=[agent], map_name=map_name, rules=rules, model=AGENT_MODEL, temperature=0.0).get(agent, [])
    return o


def run_engine(s, agent_orders=None):
    """Real (agent_orders None) or W2R (agent's orders replayed by phase name). Returns dSC, void count, replayed order count."""
    st, void, n = s["state"], 0, 0
    for _ in range(H):
        if st["name"] in ("COMPLETED", "") or not st["name"][-1] in "MRA":
            break
        orders = dipnet(st, s["map"], s["rules"], s["agent"])
        if agent_orders is not None:
            mine = agent_orders.get(st["name"], None)
            need = len(orders.get(s["agent"], []))
            L = {o for v in legal(st, s["map"], s["rules"]).values() for o in v}
            mine = mine or []
            void += sum(o not in L for o in mine) + max(0, need - len(mine)) if need or mine else 0
            n += max(need, len(mine))
            orders[s["agent"]] = [o for o in mine if o in L]
        st, _ = step(st, orders, s["map"], s["rules"])
    return sc(st, s["agent"]) - sc(s["state"], s["agent"]), void, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--real_only", action="store_true")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--thresh", type=int, default=0); ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--task", default="J", choices=["J", "S"])
    ap.add_argument("--agent_model", default="sl", choices=["sl", "rl"]); a = ap.parse_args()
    global AGENT_MODEL; AGENT_MODEL = a.agent_model
    S = starts(a.n); os.makedirs(OUT, exist_ok=True)
    real_path = f"{OUT}/real_n{a.n}.parquet" if a.agent_model == "sl" else f"{OUT}/real_n{a.n}_agent{a.agent_model}.parquet"
    if os.path.exists(real_path):
        real = pd.read_parquet(real_path)
    else:
        real = pd.DataFrame([{"sid": s["sid"], "agent": s["agent"], "dsc_real": run_engine(s)[0]} for s in S]); real.to_parquet(real_path, index=False)
    if a.real_only:
        d = real.dsc_real
        print(f"Real (engine, DipNet greedy, H={H}, n={len(d)}): mean dSC {d.mean():.2f}; success rate dSC>=0 {(d >= 0).mean():.3f}, >=1 {(d >= 1).mean():.3f}, "
              f">=-1 {(d >= -1).mean():.3f}; distribution {d.value_counts().sort_index().to_dict()}")
        return
    O = f"{OUT}/{a.tag}"; os.makedirs(O, exist_ok=True); log = f"{O}/infer.log"
    wm = {s["sid"]: {"state": s["state"], "alive": True, "fail": None, "agent_orders": {}} for s in S}
    for h in range(H):
        rows, ords = [], {}
        for s in S:
            w = wm[s["sid"]]
            if not w["alive"]:
                continue
            st = w["state"]
            if st["name"][-1:] not in ("M", "R", "A"):
                w["alive"] = False; continue   # game ended inside the world model
            try:
                o = dipnet(st, s["map"], s["rules"], s["agent"])
            except Exception:
                w["alive"] = False; w["fail"] = f"dipnet@{h}"; continue
            ords[s["sid"]] = o; w["agent_orders"][st["name"]] = o.get(s["agent"], [])
            p = render_task_j(s["map"], s["rules"], 7, st, o) if a.task == "J" else render_task_s(s["map"], s["rules"], 7, None, st, s["agent"], o.get(s["agent"], []))
            rows.append({"sample_id": f"{s['sid']}|{h}", "task": a.task, "prompt": p})
        if not rows:
            break
        P = predict(a.model, rows, f"{O}/diplomacy_h{h}", concurrency=a.concurrency, log=log)
        for s in S:
            w = wm[s["sid"]]
            if s["sid"] not in ords:
                continue
            text = P.get(f"{s['sid']}|{h}", {}).get("text", "")
            first = text.strip().splitlines()[0] if text.strip() else ""
            name = first.split()[1] if first.startswith("NEXT ") and len(first.split()) > 1 else None
            nxt = pred_to_state(text, name, w["state"].get("homes")) if name else None
            ok = nxt is not None
            if ok:
                try:
                    ok = is_valid_board(nxt, s["map"])
                except Exception:
                    ok = False
            if not ok:
                w["alive"] = False; w["fail"] = f"board@{h}"; continue
            w["state"] = for_phase(nxt, name)
        print(f"phase {h + 1}/{H}: {sum(w['alive'] for w in wm.values())} trajectories alive", flush=True)
    recs = []
    for s in S:
        w = wm[s["sid"]]
        done = w["fail"] is None
        dsc_wm = sc(w["state"], s["agent"]) - sc(s["state"], s["agent"]) if done else np.nan
        dsc_w2r, void, n = run_engine(s, w["agent_orders"])
        recs.append({"sid": s["sid"], "agent": s["agent"], "wm_done": done, "fail": w["fail"], "dsc_wm": dsc_wm, "dsc_w2r": dsc_w2r, "void": void, "n_replayed": n})
    df = pd.DataFrame(recs).merge(real, on=["sid", "agent"]); df.to_parquet(f"{O}/per_start.parquet", index=False)
    t = a.thresh
    real_s = (df.dsc_real >= t).mean(); wm_s = ((df.dsc_wm >= t) & df.wm_done).mean(); w2r_s = (df.dsc_w2r >= t).mean()
    done = df[df.wm_done]
    lines = [f"# Diplomacy CR: {a.tag} (Task {a.task}; agent DipNet {a.agent_model.upper()} greedy, opponents DipNet SL greedy, H={H} phases, n={len(df)}, success = dSC >= {t})", "",
             "| Real | WM | W2R | CR = W2R/Real | WM/Real | WM completed | imagination gap (dSC_WM - dSC_W2R) | mean dSC Real | mean dSC WM | mean dSC W2R | replay void rate |",
             "|---|---|---|---|---|---|---|---|---|---|---|",
             f"| {real_s:.3f} | {wm_s:.3f} | {w2r_s:.3f} | {w2r_s / real_s:.3f} | {wm_s / real_s:.3f} | {df.wm_done.mean():.3f} | {(done.dsc_wm - done.dsc_w2r).mean():.2f} | "
             f"{df.dsc_real.mean():.2f} | {done.dsc_wm.mean():.2f} | {df.dsc_w2r.mean():.2f} | {df.void.sum() / max(1, df.n_replayed.sum()):.3f} |", "",
             "WM failures by type: " + json.dumps(df.fail.fillna("none").str.split("@").str[0].value_counts().to_dict())]
    open(f"{O}/tables.md", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
