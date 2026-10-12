"""Agent usefulness (2/2): synthetic experience from the world model, following Word2World Section 7.2.

The behaviour policy is fixed (DipNet SL, temperature 0.1 for every power, the paper's benchmark setting); only the environment that
produces the trajectories changes. From the same 2,000 starting boards of the TRAIN split (7p movement phases, one agent power per start):
  real : the true engine plays H = 10 phases.
  wm   : the world model is the environment (Task S, K = 0): at each phase DipNet orders the agent's power on the world model's board and
         the world model predicts the next board, imagining the other powers. An unparseable or invalid board ends the trajectory.
A trajectory is successful if the agent's supply centers after H phases (as judged by its own environment) are at least those at the
start. Successful trajectories become behaviour-cloning data: every (board, agent's orders) pair with non-empty orders, as a policy
prompt (task P) whose target is the agent's order line.
Arms (Word2World's four): real 1k, syn 1k, 0.5k + 0.5k, 1k + 1k (counted in trajectories). Each trains Qwen3-1.7B the same way, and each
policy is evaluated in the true engine on the 200 test starts of the verifier experiment (agent = the policy, opponents = DipNet SL
greedy, H = 10): mean dSC, success rate (dSC >= 0), and legality of the policy's orders (illegal orders are dropped; units without a legal
order hold).

Usage: agent_syndata_dip.py gen --source real --tag real
       agent_syndata_dip.py gen --source wm --model <world model> --tag wm-<name>     (one GPU)
       agent_syndata_dip.py build --real real --syn wm-<name>                          (writes data/processed/diplomacy/sft_agent/)
       agent_syndata_dip.py eval --policy <checkpoint> --tag <arm>                     (one GPU)
Data-size control (real 2k): more real trajectories from NEW train starts, then a fifth arm of 2,000 successful real trajectories.
       agent_syndata_dip.py gen --source real --tag real2 --n 1300 --seed 2 --exclude real   (starts disjoint from traj_real; writes starts.json)
       agent_syndata_dip.py build --real real --syn wm-<name> --real2 real2 --arms real2k    (adds agent_real2k.json, leaves the other arms alone)
Imagined 2k (does the policy keep improving with more imagined games?): the same 1,300 new starts played inside the world model, then a sixth
arm of 2,000 successful imagined trajectories, built like real2k (every successful trajectory of --syn topped up from --syn2).
       agent_syndata_dip.py gen --source wm --model <world model> --tag wm2-<name> --n 1300 --seed 2 --exclude real
       agent_syndata_dip.py build --real real --syn wm-<name> --syn2 wm2-<name> --arms syn2k
"""
import argparse, hashlib, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, polars as pl
from eventworld.dip_engine import step, legal, is_valid_board, _seed_game
from eventworld.dip_serialize import render_header, render_state, render_task_s, _pw, UNABBR
from eventworld.prompts import SYS
from eventworld.rollout_predict import predict
from rollout_dip import pred_to_state, for_phase
from rollout_cr import starts as test_starts, sc
D = "data/processed/diplomacy"; OUT = "out/agent/diplomacy_syndata"; SFT = f"{D}/sft_agent"; H = 10; TEMP = 0.1


def train_starts(n=2000, seed=1, exclude=()):
    """n random 7p movement-phase starts of the TRAIN split. `exclude` (sample_ids) are dropped before sampling, so a second draw with another
    seed is disjoint from the first; with the defaults the result is the original 2,000 starts of traj_real."""
    t = (pl.scan_parquet(f"{D}/transitions.parquet").filter((pl.col("split") == "train") & (pl.col("pop") == "7p") & (pl.col("phase_type") == "M"))
         .select("sample_id", "map", "rules_no_check", "state_json").collect().to_pandas())
    if len(exclude):
        t = t[~t.sample_id.isin(set(exclude))]
    t = t.sample(n=n, random_state=seed)
    out = []
    for r in t.itertuples():
        st = json.loads(r.state_json)
        alive = sorted(p for p, u in st["units"].items() if u)
        agent = alive[int(hashlib.sha1(r.sample_id.encode()).hexdigest()[:8], 16) % len(alive)]
        out.append({"sid": r.sample_id, "map": r.map, "rules": ["NO_CHECK"] if r.rules_no_check else [], "state": st, "agent": agent})
    return out


_DN = None
def dn(st, m, rules, powers=None, temperature=0.0):
    global _DN
    if _DN is None:
        from eventworld.dipnet import dipnet_orders_batch
        _DN = dipnet_orders_batch
    return _DN(_seed_game(st, m, rules).get_state(), powers=powers, map_name=m, rules=rules, temperature=temperature)


def policy_prompt(m, rules, st, power):
    return "\n".join([render_header(m, 7, rules), f"PHASE {st['name']}", render_state(st), f"POWER {_pw(power)}"])


def policy_target(power, orders):
    return f"{_pw(power)}: " + " ; ".join(orders)


def parse_policy(text, power):
    for l in (text or "").strip().splitlines():
        l = l.strip()
        if len(l) > 4 and l[3] == ":" and UNABBR.get(l[:3], l[:3]) == power:
            return [o.strip() for o in l[4:].split(";") if o.strip()]
    return []


# ---------------- gen ----------------
def real_traj(s):
    st, steps = s["state"], []
    for _ in range(H):
        if st["name"][-1:] not in ("M", "R", "A"):
            break
        o = dn(st, s["map"], s["rules"], temperature=TEMP)
        if o.get(s["agent"]):
            steps.append({"state": st, "orders": o[s["agent"]]})
        st, _ = step(st, o, s["map"], s["rules"])
    return {"sid": s["sid"], "agent": s["agent"], "done": True, "dsc": sc(st, s["agent"]) - sc(s["state"], s["agent"]), "steps": steps}


def agent_orders(job):
    return dn(job["state"], job["map"], job["rules"], powers=[job["agent"]], temperature=TEMP).get(job["agent"], [])


def gen(a):
    excl = set()
    for tag in (a.exclude.split(",") if a.exclude else []):   # starts already used by these trajectory sets are not drawn again
        excl |= set(pd.read_parquet(f"{OUT}/traj_{tag}/trajectories.parquet").sid)
    S = train_starts(a.n, a.seed, excl); O = f"{OUT}/traj_{a.tag}"; os.makedirs(O, exist_ok=True)
    json.dump({"n": a.n, "seed": a.seed, "exclude": a.exclude, "n_excluded": len(excl),
               "starts": [{k: s[k] for k in ("sid", "map", "rules", "agent")} for s in S]}, open(f"{O}/starts.json", "w"))
    pool = Pool(a.workers)
    if a.source == "real":
        recs = pool.map(real_traj, S, chunksize=8)
    else:
        W = {s["sid"]: {"s": s, "state": s["state"], "alive": True, "fail": None, "steps": []} for s in S}
        for h in range(H):
            act = [w for w in W.values() if w["alive"] and w["state"]["name"][-1:] in ("M", "R", "A")]
            for w in W.values():
                if w["alive"] and w not in act:
                    w["alive"] = False   # game ended inside the world model
            if not act:
                break
            ords = pool.map(agent_orders, [{"state": w["state"], "map": w["s"]["map"], "rules": w["s"]["rules"], "agent": w["s"]["agent"]} for w in act], chunksize=8)
            rows = []
            for w, o in zip(act, ords):
                if o:
                    w["steps"].append({"state": w["state"], "orders": o})
                s = w["s"]
                rows.append({"sample_id": f"{s['sid']}|{h}", "task": "S", "prompt": render_task_s(s["map"], s["rules"], 7, None, w["state"], s["agent"], o)})
            P = predict(a.model, rows, f"{O}/diplomacy_h{h}", concurrency=16, log=f"{O}/infer.log")
            for w in act:
                s = w["s"]; text = P.get(f"{s['sid']}|{h}", {}).get("text", "")
                first = text.strip().splitlines()[0] if text.strip() else ""
                name = first.split()[1] if first.startswith("NEXT ") and len(first.split()) > 1 else None
                ns = pred_to_state(text, name, w["state"].get("homes")) if name else None
                ok = ns is not None
                if ok:
                    try:
                        ok = is_valid_board(ns, s["map"])
                    except Exception:
                        ok = False
                if not ok:
                    w["alive"] = False; w["fail"] = f"board@{h}"; continue
                w["state"] = for_phase(ns, name)
            print(f"phase {h + 1}/{H}: {sum(w['alive'] for w in W.values())} alive", flush=True)
        recs = [{"sid": w["s"]["sid"], "agent": w["s"]["agent"], "done": w["fail"] is None,
                 "dsc": (sc(w["state"], w["s"]["agent"]) - sc(w["s"]["state"], w["s"]["agent"])) if w["fail"] is None else np.nan, "steps": w["steps"]} for w in W.values()]
    pool.close()
    df = pd.DataFrame([{**{k: r[k] for k in ("sid", "agent", "done", "dsc")}, "n_steps": len(r["steps"]), "steps": json.dumps(r["steps"])} for r in recs])
    df["success"] = df.done & (df.dsc >= 0)
    df.to_parquet(f"{O}/trajectories.parquet", index=False)
    print(f"{a.tag}: {len(df)} trajectories, completed {df.done.mean():.3f}, success {df.success.mean():.3f}, mean dSC {df.dsc.mean():.2f}, "
          f"steps per successful trajectory {df[df.success].n_steps.mean():.1f}")


# ---------------- build ----------------
def samples(df, n, start_map):
    out = []
    for r in df[df.success].head(n).itertuples():
        s = start_map[r.sid]
        for stp in json.loads(r.steps):
            out.append({"system": SYS["dip_P"], "instruction": policy_prompt(s["map"], s["rules"], stp["state"], r.agent), "output": policy_target(r.agent, stp["orders"])})
    return out


def build(a):
    start_map = {s["sid"]: s for s in train_starts(a.n)}
    R = pd.read_parquet(f"{OUT}/traj_{a.real}/trajectories.parquet"); Sy = pd.read_parquet(f"{OUT}/traj_{a.syn}/trajectories.parquet")
    for name, df in (("real", R), ("syn", Sy)):
        print(f"{name}: {int(df.success.sum())} successful trajectories available")
    arms = {"real1k": samples(R, 1000, start_map), "syn1k": samples(Sy, 1000, start_map),
            "mix0.5k": samples(R, 500, start_map) + samples(Sy, 500, start_map), "mix1k1k": samples(R, 1000, start_map) + samples(Sy, 1000, start_map)}
    if a.real2:   # real 2k control: every successful trajectory of --real (a superset of real1k's 1,000) topped up to 2,000 from --real2
        R2 = pd.read_parquet(f"{OUT}/traj_{a.real2}/trajectories.parquet")
        start_map.update({s["sid"]: s for s in json.load(open(f"{OUT}/traj_{a.real2}/starts.json"))["starts"]})
        k = int(R.success.sum()); print(f"real2: {int(R2.success.sum())} successful trajectories available; real2k = {k} from {a.real} + {2000 - k} from {a.real2}")
        assert R2.success.sum() >= 2000 - k, "not enough successful trajectories in --real2"
        arms["real2k"] = samples(R, k, start_map) + samples(R2, 2000 - k, start_map)
    if a.syn2:    # imagined 2k: every successful trajectory of --syn (a superset of syn1k's 1,000) topped up to 2,000 from --syn2
        S2 = pd.read_parquet(f"{OUT}/traj_{a.syn2}/trajectories.parquet")
        start_map.update({s["sid"]: s for s in json.load(open(f"{OUT}/traj_{a.syn2}/starts.json"))["starts"]})
        k = int(Sy.success.sum()); print(f"syn2: {int(S2.success.sum())} successful trajectories available; syn2k = {k} from {a.syn} + {2000 - k} from {a.syn2}")
        assert S2.success.sum() >= 2000 - k, "not enough successful trajectories in --syn2"
        arms["syn2k"] = samples(Sy, k, start_map) + samples(S2, 2000 - k, start_map)
    sel = set(a.arms.split(",")) if a.arms else set(arms)
    os.makedirs(SFT, exist_ok=True)
    info = json.load(open(f"{SFT}/dataset_info.json")) if os.path.exists(f"{SFT}/dataset_info.json") else {}
    rng = np.random.default_rng(0)
    for k, v in arms.items():
        v = [v[i] for i in rng.permutation(len(v))]   # the permutation is drawn for every arm so a selected arm's order does not depend on --arms
        if k not in sel:
            continue
        json.dump(v, open(f"{SFT}/agent_{k}.json", "w"))
        info[f"agent_{k}"] = {"file_name": f"agent_{k}.json", "columns": {"prompt": "instruction", "response": "output", "system": "system"}}
        print(f"agent_{k}: {len(v)} samples")
    json.dump(info, open(f"{SFT}/dataset_info.json", "w"), indent=1)


# ---------------- eval ----------------
def play_opponents(job):
    st, m, rules, agent = job["state"], job["map"], job["rules"], job["agent"]
    base = dn(st, m, rules)   # the opponents' greedy orders; the agent's entry tells whether it has anything to order this phase
    L = legal(st, m, rules)
    return {"base": base, "legal": sorted({o for v in L.values() for o in v}), "need": bool(base.get(agent))}


def evaluate(a):
    S = test_starts(a.n); O = f"{OUT}/eval_{a.tag}"; os.makedirs(O, exist_ok=True)
    G = {s["sid"]: {"s": s, "state": s["state"], "n": 0, "legal": 0, "empty": 0} for s in S}
    pool = Pool(a.workers)
    for h in range(H):
        act = [g for g in G.values() if g["state"]["name"][-1:] in ("M", "R", "A")]
        if not act:
            break
        info = pool.map(play_opponents, [{"state": g["state"], "map": g["s"]["map"], "rules": g["s"]["rules"], "agent": g["s"]["agent"]} for g in act], chunksize=4)
        rows = [{"sample_id": f"{g['s']['sid']}|{h}", "task": "P", "prompt": policy_prompt(g["s"]["map"], g["s"]["rules"], g["state"], g["s"]["agent"])}
                for g, i in zip(act, info) if i["need"]]
        P = predict(a.policy, rows, f"{O}/diplomacy_h{h}", max_tokens=300, concurrency=16, log=f"{O}/infer.log") if rows else {}
        steps = []
        for g, i in zip(act, info):
            orders = dict(i["base"])
            if i["need"]:
                mine = parse_policy(P.get(f"{g['s']['sid']}|{h}", {}).get("text", ""), g["s"]["agent"])
                L = set(i["legal"]); ok = [o for o in mine if o in L]
                g["n"] += max(len(mine), len(i["base"][g["s"]["agent"]])); g["legal"] += len(ok); g["empty"] += not mine
                orders[g["s"]["agent"]] = ok
            steps.append((g, orders))
        nexts = pool.map(_step, [(g["state"], o, g["s"]["map"], g["s"]["rules"]) for g, o in steps], chunksize=4)
        for (g, _), ns in zip(steps, nexts):
            g["state"] = ns
        print(f"phase {h + 1}/{H}: {len(act)} games, {len(rows)} policy calls", flush=True)
    pool.close()
    df = pd.DataFrame([{"sid": g["s"]["sid"], "agent": g["s"]["agent"], "dsc": sc(g["state"], g["s"]["agent"]) - sc(g["s"]["state"], g["s"]["agent"]),
                        "n_orders": g["n"], "n_legal": g["legal"], "empty_phases": g["empty"]} for g in G.values()])
    df.to_parquet(f"{O}/per_start.parquet", index=False)
    line = (f"| {a.tag} | {df.dsc.mean():.2f} | {(df.dsc >= 0).mean():.3f} | {df.n_legal.sum() / max(1, df.n_orders.sum()):.3f} | {df.empty_phases.sum()} |")
    open(f"{O}/tables.md", "w").write("| Policy | mean dSC | success (dSC>=0) | legal-order rate | empty answers |\n|---|---|---|---|---|\n" + line + "\n")
    print(line)


def _step(args):
    st, o, m, rules = args
    return step(st, o, m, rules)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "build", "eval"])
    ap.add_argument("--source", choices=["real", "wm"]); ap.add_argument("--model"); ap.add_argument("--tag")
    ap.add_argument("--real"); ap.add_argument("--syn"); ap.add_argument("--policy")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1, help="gen: seed of the train-start draw (default 1 = the original 2,000 starts)")
    ap.add_argument("--exclude", help="gen: comma-separated trajectory tags whose starts are not drawn again")
    ap.add_argument("--real2", help="build: tag of a second real trajectory set; adds the real2k arm")
    ap.add_argument("--syn2", help="build: tag of a second world-model trajectory set; adds the syn2k arm")
    ap.add_argument("--arms", help="build: comma-separated arms to (re)write; default all")
    a = ap.parse_args()
    if a.n is None:
        a.n = 200 if a.cmd == "eval" else 2000
    {"gen": gen, "build": build, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
