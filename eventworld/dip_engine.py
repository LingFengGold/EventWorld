"""Thin wrapper around the diplomacy 1.1.2 engine.

step(state, orders, map_name, rules) -> (next_state, results)
legal(state, map_name, rules)        -> {loc: [order strings]}   (all possible orders per orderable location)
is_valid_board(state, map_name)      -> bool
adjacency(map_name)                  -> {loc: {"A": [...], "F": [...]}}

State dicts use the dataset's fields: units, centers, retreats, builds, homes, name.
"""
from functools import lru_cache
from diplomacy import Game

POWERS_ORDER = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def _seed_game(state, map_name="standard", rules=None):
    g = Game(map_name=map_name, rules=list(rules or []))
    st = {k: state[k] for k in ("name", "units", "centers", "homes") if k in state}
    if "retreats" in state:
        st["retreats"] = state["retreats"]
    if "builds" in state:
        st["builds"] = state["builds"]
    g.set_state(st)
    return g


def step(state, orders, map_name="standard", rules=None):
    """Adjudicate `orders` ({power: [order,...]}) from `state`; return (next_state, results)."""
    g = _seed_game(state, map_name, rules)
    for p, o in (orders or {}).items():
        if o:
            g.set_orders(p, o)
    g.process()
    ns = g.get_state()
    def tag(r):
        m = getattr(r, "message", None)
        return str(m) if m is not None else str(r).split(":")[-1]
    flat = {}
    for k, v in g.get_order_status().items():
        if isinstance(v, dict):
            for u, rs in v.items():
                flat[u] = [tag(r) for r in rs]
        else:
            flat[k] = [tag(r) for r in v]
    return ns, flat


def legal(state, map_name="standard", rules=None):
    g = _seed_game(state, map_name, rules)
    return {loc: list(v) for loc, v in g.get_all_possible_orders().items() if v}


def orderable(state, power, map_name="standard", rules=None):
    g = _seed_game(state, map_name, rules)
    return list(g.get_orderable_locations(power))


def is_valid_board(state, map_name="standard"):
    m = _map(map_name)
    seen = set()
    for p, units in state.get("units", {}).items():
        for u in units:
            dislodged = u.startswith("*")  # a dislodged unit legitimately shares its province with the attacker during the retreat phase
            u = u.lstrip("*")
            typ, loc = u.split(" ", 1)
            prov = loc.split("/")[0]
            if not dislodged:
                if prov in seen:
                    return False
                seen.add(prov)
            area = m.area_type(loc)
            if area is None:
                return False
            if typ == "A" and area == "WATER":
                return False
            if typ == "F" and area == "LAND":
                return False
    return True


@lru_cache(maxsize=8)
def _map(map_name="standard"):
    return Game(map_name=map_name).map


def adjacency(map_name="standard"):
    m = _map(map_name)
    out = {}
    for loc in m.locs:
        out[loc] = {
            "type": m.area_type(loc),
            "sc": loc.split("/")[0] in m.scs,
            "A": sorted(l for l in m.locs if l != loc and m.abuts("A", loc, "-", l)),
            "F": sorted(l for l in m.locs if l != loc and m.abuts("F", loc, "-", l)),
        }
    return out


def units_centers(state):
    """Canonical comparable form: {power: (sorted units, sorted centers)}."""
    return {p: (tuple(sorted(state["units"].get(p, []))), tuple(sorted(state["centers"].get(p, []))))
            for p in set(state.get("units", {})) | set(state.get("centers", {}))}
