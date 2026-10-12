"""D6: counterfactual actions for the 7p common eval set (movement phases): uniform / minimal / plausible / hold (empty action) / null, with engine references."""
import json, random, os, sys, time
import polars as pl, pandas as pd
from eventworld.dip_engine import step, legal
from eventworld.dip_serialize import render_task_j, render_task_j_target, render_task_s, render_task_s_direct_target

random.seed(0)
D = "data/processed/diplomacy"
t0 = time.time()
common = pd.read_parquet(f"{D}/eval_common_7p.parquet")
common = common[common.phase_type == "M"]
T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(common.sample_id.tolist())).collect().to_pandas().set_index("sample_id")
freq = pd.read_parquet(f"{D}/order_freq_train.parquet")
freq = freq[(freq["pop"] == "7p") & (freq.phase_type == "M")]
FREQ = {}
for r in freq.itertuples():
    FREQ.setdefault((r.unit_type, r.loc), []).append((r.template, r.count))
# history for Task S prompts (K=3)
test = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("split") == "test").select("game_id", "phase_idx", "state_json", "orders_json", "results_json").collect().to_pandas()
by_game = {g: d.sort_values("phase_idx") for g, d in test.groupby("game_id")}
def hist(game_id, phase_idx, K=3):
    d = by_game[game_id]; d = d[d.phase_idx < phase_idx].tail(K)
    return [(json.loads(r.state_json), json.loads(r.orders_json), json.loads(r.results_json)) for r in d.itertuples()]

def unit_locs(state, p):
    return [u.split(" ", 1) for u in state["units"].get(p, []) if not u.startswith("*")]

def sample_plausible(typ, loc, options):
    cands = FREQ.get((typ, loc), [])
    opts = set(options)
    weighted = [(t.replace("*", f"{typ} {loc}", 1), c) for t, c in cands if t.replace("*", f"{typ} {loc}", 1) in opts]
    if not weighted:
        return random.choice(options)
    tot = sum(c for _, c in weighted); r = random.uniform(0, tot)
    for o, c in weighted:
        r -= c
        if r <= 0:
            return o
    return weighted[-1][0]

def delta(a, b):
    out = []
    for p in set(a["units"]) | set(b["units"]):
        ua, ub = set(a["units"].get(p, [])), set(b["units"].get(p, []))
        out += [["+", p, u] for u in sorted(ub - ua)] + [["-", p, u] for u in sorted(ua - ub)]
    for p in set(a["centers"]) | set(b["centers"]):
        ca, cb = set(a["centers"].get(p, [])), set(b["centers"].get(p, []))
        out += [["+sc", p, c] for c in sorted(cb - ca)] + [["-sc", p, c] for c in sorted(ca - cb)]
    return out

rows = []; fj = open(f"{D}/eval_text/common_7p_CF_J.jsonl", "w"); fS = {K: open(f"{D}/eval_text/common_7p_CF_S_K{K}.jsonl", "w") for K in (0, 3)}  # protocol v3: K in {0,3}, S target = next state only
for r in common.itertuples():
    tr = T.loc[r.sample_id]
    st, orders, ns_fact = json.loads(tr["state_json"]), json.loads(tr["orders_json"]), json.loads(tr["next_state_json"])
    rules = ["NO_CHECK"] if tr["rules_no_check"] else []
    leg = legal(st, tr["map"], rules)
    for p in json.loads(r.designated):
        units = unit_locs(st, p)
        if not units:
            continue
        own = list(orders.get(p, []) or [])
        schemes = {"null": own, "hold": [f"{typ} {loc} H" for typ, loc in units]}
        schemes["uniform"] = [random.choice(leg[loc]) for typ, loc in units if leg.get(loc)]
        plaus = [sample_plausible(typ, loc, leg[loc]) for typ, loc in units if leg.get(loc)]
        schemes["plausible"] = plaus
        mini = list(own) if own else [f"{typ} {loc} H" for typ, loc in units]
        i = random.randrange(len(units)); typ, loc = units[i]
        cur = next((o for o in mini if o.split()[1] == loc), None)
        alts = [o for o in leg.get(loc, []) if o != cur]
        if alts:
            new = random.choice(alts)
            mini = [o for o in mini if o.split()[1] != loc] + [new]
        schemes["minimal"] = mini
        for scheme, cf_own in schemes.items():
            cf_orders = {q: list(o) for q, o in orders.items()}; cf_orders[p] = cf_own
            ns_cf, res_cf = step({k: st.get(k) for k in ("name", "units", "centers", "retreats", "builds", "homes")}, cf_orders, tr["map"], rules)
            ns_cf_min = {k: ns_cf.get(k) for k in ("name", "units", "centers", "retreats", "builds")}
            dl = delta(ns_fact, ns_cf_min)
            cf_id = f"{r.sample_id}:{p}:{scheme}"
            rows.append(dict(cf_id=cf_id, sample_id=r.sample_id, power=p, scheme=scheme, cf_orders_json=json.dumps(cf_own),
                             ref_next_state_json=json.dumps(ns_cf_min, separators=(",", ":")), ref_results_json=json.dumps({u: v for u, v in res_cf.items() if v}),
                             ref_delta_json=json.dumps(dl), ref_delta_size=len(dl),
                             n_orders_changed=sum(1 for o in cf_own if o not in own) + sum(1 for o in own if o not in cf_own),
                             touches_other_powers=any(x[1] != p for x in dl)))
            fj.write(json.dumps(dict(cf_id=cf_id, task="J", scheme=scheme, prompt=render_task_j(tr["map"], rules, 7, st, cf_orders), target=render_task_j_target(res_cf, st, ns_cf_min))) + "\n")
            for K, fs in fS.items():
                fs.write(json.dumps(dict(cf_id=cf_id, task="S", scheme=scheme, K=K, prompt=render_task_s(tr["map"], rules, 7, hist(tr["game_id"], tr["phase_idx"], K) if K else [], st, p, cf_own), target=render_task_s_direct_target(ns_cf_min))) + "\n")
fj.close(); [f.close() for f in fS.values()]
df = pd.DataFrame(rows); df.to_parquet(f"{D}/counterfactuals.parquet", index=False)
summ = df.groupby("scheme").agg(n=("cf_id", "size"), delta_size_mean=("ref_delta_size", "mean"), share_nonzero=("ref_delta_size", lambda x: (x > 0).mean()),
                                touches_others=("touches_other_powers", "mean"), orders_changed=("n_orders_changed", "mean")).round(3)
print(summ.to_string()); print(f"{len(df):,} counterfactuals in {time.time()-t0:.0f}s")
open("out/build/dip_counterfactuals.md", "w").write("# Diplomacy counterfactual set (7p common eval, movement phases)\n\n" + summ.to_markdown() + f"\n\n{len(df):,} rows\n")
