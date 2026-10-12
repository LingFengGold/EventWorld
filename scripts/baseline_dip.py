"""Floor baselines for Diplomacy, written as prediction JSONL in the shared contract, for the 7p/2p common eval sets.
  state_unchanged     (Task J): board stays; dislodged units disband; no results
  all_moves_succeed   (Task J): every move/retreat/build applied literally (orders as written, no adjudication); conflicts resolved by power order; centers by occupancy at winter
  most_frequent_order (Task S): other powers' orders = most frequent legal template per (unit type, location) from train; state via engine
  uniform_legal       (Task S): other powers' orders sampled uniformly from the legal set; state via engine
  own_literal         (Task S): the agent's orders applied as in all_moves_succeed, the other powers issue no orders
Scored with scripts/eval_dip.py (Task J: phase_exact; Task S: phase_exact of the board, or state_exact_factored for the engine-resolved ones).
"""
import json, os, random, sys
import polars as pl, pandas as pd
from eventworld.dip_serialize import render_task_j_target, render_task_s_target, FULL
from eventworld.dip_engine import step, legal
random.seed(0)
D = "data/processed/diplomacy"; OUT = "out/eval/baselines"; os.makedirs(OUT, exist_ok=True)
freq = pd.read_parquet(f"{D}/order_freq_train.parquet")
TOP = {}
for (pop, pt, ut, loc), d in freq.groupby(["pop", "phase_type", "unit_type", "loc"]):
    TOP[(pop, pt, ut, loc)] = list(d.sort_values("count", ascending=False)["template"])

_SC = {}
def _scs(map_name):
    if map_name not in _SC:
        from diplomacy import Map
        _SC[map_name] = {c.upper() for c in Map(map_name).scs}
    return _SC[map_name]

def strip_star(units):
    return {p: [u for u in us if not u.startswith("*")] for p, us in units.items()}

def state_unchanged(st, ns_name):
    return {"name": ns_name, "units": strip_star(st["units"]), "centers": st["centers"], "retreats": {}, "builds": {}}

def all_moves(st, orders, ns_name, map_name=None):
    units = {p: list(us) for p, us in strip_star(st["units"]).items()}
    occupied = {}
    for p in FULL:
        for u in units.get(p, []):
            occupied[u.split(" ", 1)[1].split("/")[0]] = (p, u)
    new_units = {p: [] for p in units}
    taken = set()
    for p in FULL:
        for u in units.get(p, []):
            typ, loc = u.split(" ", 1)
            order = next((o for o in (orders.get(p) or []) if o.split()[1] == loc), None)
            dest = None
            if order:
                t = order.split()
                if len(t) >= 4 and t[2] in ("-", "R"):
                    dest = t[3]
                elif len(t) >= 3 and t[2] == "D":
                    continue
            target = dest or loc
            if target.split("/")[0] in taken:
                target = loc
            if target.split("/")[0] in taken:
                continue
            taken.add(target.split("/")[0]); new_units[p].append(f"{typ} {target}")
    # retreat phase: dislodged units ('*A LOC') follow their retreat order; two units retreating to one province, or into an occupied one, disband
    dests = {}
    for p in FULL:
        for u in st["units"].get(p, []):
            if not u.startswith("*"):
                continue
            typ, loc = u[1:].split(" ", 1)
            order = next((o for o in (orders.get(p) or []) if len(o.split()) >= 4 and o.split()[1] == loc and o.split()[2] == "R"), None)
            if order:
                dests.setdefault(order.split()[3].split("/")[0], []).append((p, f"{typ} {order.split()[3]}"))
    for prov, us in dests.items():
        if len(us) == 1 and prov not in taken:
            taken.add(prov); new_units.setdefault(us[0][0], []).append(us[0][1])
    if st["name"].endswith("A"):
        for p in FULL:
            for o in orders.get(p) or []:
                t = o.split()
                if len(t) >= 3 and t[2] == "B":
                    new_units.setdefault(p, []).append(f"{t[0]} {t[1]}")
    centers = {p: list(c) for p, c in st["centers"].items()}
    if ns_name.endswith("A"):
        owner = {c: p for p, cs in centers.items() for c in cs}
        for p, us in new_units.items():
            for u in us:
                prov = u.split(" ", 1)[1].split("/")[0]
                if any(prov in cs for cs in centers.values()) or prov in owner or (map_name and prov in _scs(map_name)):
                    owner[prov] = p
        centers = {p: sorted(c for c, q in owner.items() if q == p) for p in FULL}
    return {"name": ns_name, "units": new_units, "centers": centers, "retreats": {}, "builds": {}}

def others_orders(st, power, pop, rules, mode, MAP="standard"):
    leg = legal(st, MAP, rules)
    out = {}
    for p, us in st["units"].items():
        if p == power or not us:
            continue
        ol = []
        for u in us:
            if u.startswith("*"):
                continue
            typ, loc = u.split(" ", 1)
            options = (leg or {}).get(loc)
            if not options:
                continue          # not orderable in this phase (e.g. non-dislodged unit in a retreat phase)
            if mode == "uniform":
                ol.append(random.choice(options)); continue
            pick = None
            for tmpl in TOP.get((pop, st["name"][-1], typ, loc), []):
                cand = tmpl.replace("*", f"{typ} {loc}", 1)
                if cand in options:
                    pick = cand; break
            ol.append(pick or (f"{typ} {loc} H" if f"{typ} {loc} H" in options else options[0]))
        if st["name"].endswith("A"):
            unit_locs = {u.split(" ", 1)[1] for u in us if not u.startswith("*")}
            for loc, options in (leg or {}).items():
                if loc in unit_locs or not options:
                    continue
                if any(o.split()[-1] == "B" and o.split()[1] == loc for o in options) and st.get("builds", {}).get(p, {}).get("count", 0) > 0 and loc in (st.get("builds", {}).get(p, {}).get("homes") or []):
                    ol.append(random.choice(options) if mode == "uniform" else options[0])
        out[p] = ol
    return out

for pop in ["7p", "2p"]:
    common = pd.read_parquet(f"{D}/eval_common_{pop}.parquet")
    T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("sample_id").is_in(common.sample_id.tolist())).collect().to_pandas().set_index("sample_id")
    fj1 = open(f"{OUT}/dip_{pop}_J_state_unchanged.jsonl", "w"); fj2 = open(f"{OUT}/dip_{pop}_J_all_moves_succeed.jsonl", "w")
    fs1 = open(f"{OUT}/dip_{pop}_S_most_frequent_order.jsonl", "w"); fs2 = open(f"{OUT}/dip_{pop}_S_uniform_legal.jsonl", "w")
    fs3 = open(f"{OUT}/dip_{pop}_S_own_literal.jsonl", "w")
    for r in common.itertuples():
        tr = T.loc[r.sample_id]
        st, orders, ns = json.loads(tr["state_json"]), json.loads(tr["orders_json"]), json.loads(tr["next_state_json"])
        rules = ["NO_CHECK"] if tr["rules_no_check"] else []
        for f, fn in [(fj1, state_unchanged), (fj2, lambda s, n: all_moves(s, orders, n, tr["map"]))]:
            pred = fn(st, ns["name"])
            f.write(json.dumps({"sample_id": r.sample_id, "task": "J", "pred": {"text": render_task_j_target({}, st, pred)}}) + "\n")
        for pw in json.loads(r.designated):
            for f, mode in [(fs1, "mfo"), (fs2, "uniform")]:
                texts = []
                for _ in range(3 if mode == "uniform" else 1):
                    oth = others_orders(st, pw, pop, rules, mode, tr["map"])
                    joint = {pw: orders.get(pw, [])} | oth
                    try:
                        eng_ns, _ = step({k: st.get(k) for k in ("name", "units", "centers", "retreats", "builds", "homes")}, joint, tr["map"], rules)
                    except Exception:
                        eng_ns = state_unchanged(st, ns["name"])
                    texts.append(render_task_s_target(joint, pw, {k: eng_ns.get(k) for k in ("name", "units", "centers", "retreats", "builds")}))
                f.write(json.dumps({"sample_id": f"{r.sample_id}:{pw}", "task": "S", "samples": [{"text": t} for t in texts]}) + "\n")
            own = {pw: orders.get(pw, [])}
            fs3.write(json.dumps({"sample_id": f"{r.sample_id}:{pw}", "task": "S", "samples": [{"text": render_task_s_target(own, pw, all_moves(st, own, ns["name"], tr["map"]))}]}) + "\n")
    for f in (fj1, fj2, fs1, fs2, fs3):
        f.close()
    print(pop, "done")
