"""D4: deterministic text renderer and parser for Diplomacy states, orders and results.

Format (one power per line, powers in fixed order, only powers with units or centers):
GAME standard 7P RULES NO_CHECK        <- RULES omitted when none of interest
PHASE S1902M
STATE
AUS: A BUD A GAL F GRE | SC BUD GRE TRI VIE
RUS: A MOS *A RUM>UKR,GAL F SEV | SC MOS SEV STP WAR    <- dislodged unit with retreat options (retreat phase)
BUILDS AUS +2 @BUD,TRI ; RUS -1                          <- adjustment phase only
ORDERS
AUS: A TRI - SER ; A BUD S A GAL - RUM
RESULTS
AUS: A GAL bounce ; TUR: A RUM cut dislodged
Parsing returns canonical structures (sets) so exact match is structural.
"""
import re

FULL = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
ABBR = {p: p[:3] for p in FULL}
UNABBR = {v: k for k, v in ABBR.items()}


def _pw(p):
    return ABBR.get(p, p[:3].upper())


def render_state(state, with_builds=True):
    lines = ["STATE"]
    units = state.get("units", {}) or {}
    centers = state.get("centers", {}) or {}
    retreats = state.get("retreats", {}) or {}
    for p in FULL:
        us = list(units.get(p, []) or []); cs = sorted(centers.get(p, []) or [])
        if not us and not cs:
            continue
        toks = []
        for u in sorted(us, key=lambda x: x.lstrip("*")):
            if u.startswith("*"):
                base = u[1:]
                opts = retreats.get(p, {}).get(base) or retreats.get(p, {}).get(u) or []
                toks.append(f"*{base}>" + (",".join(opts) if opts else "-"))
            else:
                toks.append(u)
        lines.append(f"{_pw(p)}: " + " ".join(toks) + " | SC " + (" ".join(cs) if cs else "-"))
    if with_builds and state.get("name", "").endswith("A"):
        b = state.get("builds", {}) or {}
        parts = []
        for p in FULL:
            info = b.get(p)
            if not info or not info.get("count"):
                continue
            c = info["count"]
            parts.append(f"{_pw(p)} {c:+d}" + (" @" + ",".join(sorted(info.get("homes", []))) if c > 0 and info.get("homes") else ""))
        if parts:
            lines.append("BUILDS " + " ; ".join(parts))
    return "\n".join(lines)


def render_orders(orders, header="ORDERS"):
    lines = [header]
    for p in FULL:
        ol = (orders or {}).get(p) or []
        if ol:
            lines.append(f"{_pw(p)}: " + " ; ".join(ol))
    return "\n".join(lines)


def render_results(results, state):
    """results: {unit: [tags]}; grouped by owning power using state units."""
    owner = {}
    for p, us in (state.get("units", {}) or {}).items():
        for u in us:
            owner[u.lstrip("*")] = p
    lines = ["RESULTS"]
    groups = {}
    for u, tags in (results or {}).items():
        tags = [str(t) for t in (tags or []) if str(t)]
        if not tags:
            continue
        p = owner.get(u.lstrip("*"), "?")
        groups.setdefault(p, []).append(f"{u.lstrip('*')} " + " ".join(t.replace(" ", "_") for t in sorted(tags)))
    for p in FULL:
        if p in groups:
            lines.append(f"{_pw(p)}: " + " ; ".join(sorted(groups[p])))
    if len(lines) == 1:
        lines.append("none")
    return "\n".join(lines)


def render_header(map_name, n_powers, rules):
    r = [x for x in (rules or []) if x in ("NO_CHECK", "BUILD_ANY")]
    h = f"GAME {map_name} {n_powers}P"
    if r:
        h += " RULES " + ",".join(r)
    return h


def render_history(history):
    """history: list of (state, orders, results) for the K previous phases (oldest first) -> HISTORY block lines (empty list if no history)."""
    if not history:
        return []
    out = ["HISTORY"]
    for hs, ho, hr in history:
        out += [f"PHASE {hs['name']}", render_state(hs), render_orders(ho)]
        if hs["name"].endswith("M"):
            out.append(render_results(hr, hs))
    return out


def render_task_j(map_name, rules, n_powers, state, orders, history=None):
    return "\n".join([render_header(map_name, n_powers, rules)] + render_history(history) + [f"PHASE {state['name']}", render_state(state), render_orders(orders)])


def render_task_j_target(results, state, next_state):
    """Task J target (protocol v3): only the next phase name and the resulting STATE, identical in form to the Task S target."""
    return "\n".join([f"NEXT {next_state['name']}", render_state(next_state)])


def render_task_j_target_with_results(results, state, next_state):
    """Legacy (split v1/v2 runs): RESULTS block for movement phases, then NEXT + STATE. Kept for reading old prediction files."""
    out = []
    if state["name"].endswith("M"):
        out.append(render_results(results, state))
    return "\n".join(out + [f"NEXT {next_state['name']}", render_state(next_state)])


def render_task_s(map_name, rules, n_powers, history, state, power, own_orders):
    """history: list of (state, orders, results) for the K previous phases (oldest first)."""
    out = [render_header(map_name, n_powers, rules)] + render_history(history)
    out += [f"PHASE {state['name']}", render_state(state), f"OWN {_pw(power)}: " + " ; ".join(own_orders or []) if own_orders else f"OWN {_pw(power)}: none"]
    return "\n".join(out)


def render_task_s_direct_target(next_state):
    """Task S main definition: the model predicts only the next state."""
    return "\n".join([f"NEXT {next_state['name']}", render_state(next_state)])


def render_task_s_target(orders, power, next_state):
    """S-factored diagnostic variant: OTHERS block, then the next state."""
    others = {p: o for p, o in (orders or {}).items() if p != power}
    return "\n".join([render_orders(others, header="OTHERS"), f"NEXT {next_state['name']}", render_state(next_state)])


# ---------------- parsing ----------------
_LINE = re.compile(r"^([A-Z]{3}):\s*(.*)$")


def parse_state(text):
    """Parse a STATE block (the text after the 'STATE' line, or containing it). Returns canonical dict."""
    units, centers, retreats, builds = {}, {}, {}, {}
    lines = [l.strip() for l in text.strip().splitlines()]
    if lines and lines[0] == "STATE":
        lines = lines[1:]
    for l in lines:
        if not l or l in ("ORDERS", "RESULTS", "OTHERS") or l.startswith("NEXT ") or l.startswith("PHASE "):
            break
        if l.startswith("BUILDS"):
            for part in l[len("BUILDS"):].split(";"):
                part = part.strip()
                if not part:
                    continue
                m = re.match(r"^([A-Z]{3})\s+([+-]\d+)(?:\s+@(\S+))?$", part)
                if m:
                    p = UNABBR.get(m.group(1), m.group(1))
                    builds[p] = {"count": int(m.group(2)), "homes": sorted(m.group(3).split(",")) if m.group(3) else []}
            continue
        m = _LINE.match(l)
        if not m:
            continue
        p = UNABBR.get(m.group(1), m.group(1))
        body = m.group(2)
        upart, _, cpart = body.partition("|")
        toks = upart.split()
        us = set()
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("*"):
                base_type = t[1:]; i += 1
                if i < len(toks):
                    locpart = toks[i]; i += 1
                    loc, _, opts = locpart.partition(">")
                    us.add(f"*{base_type} {loc}")
                    retreats.setdefault(p, {})[f"{base_type} {loc}"] = [] if opts in ("", "-") else opts.split(",")
                continue
            if t in ("A", "F") and i + 1 < len(toks):
                us.add(f"{t} {toks[i+1]}"); i += 2
            else:
                i += 1
        units[p] = us
        cs = cpart.replace("SC", "", 1).split()
        centers[p] = set() if cs == ["-"] else set(cs)
    return {"units": units, "centers": centers, "retreats": retreats, "builds": builds}


def parse_orders(text, header=("ORDERS", "OTHERS")):
    orders = {}
    lines = [l.strip() for l in text.strip().splitlines()]
    started = not any(l in header for l in lines)
    for l in lines:
        if l in header:
            started = True; continue
        if not started:
            continue
        if l in ("STATE", "RESULTS") or l.startswith("NEXT ") or l.startswith("PHASE "):
            break
        m = _LINE.match(l)
        if m:
            p = UNABBR.get(m.group(1), m.group(1))
            orders[p] = [o.strip() for o in m.group(2).split(";") if o.strip()]
    return orders


def parse_results(text):
    res = {}
    lines = [l.strip() for l in text.strip().splitlines()]
    started = "RESULTS" not in lines
    for l in lines:
        if l == "RESULTS":
            started = True; continue
        if not started:
            continue
        if l.startswith("NEXT ") or l in ("STATE", "ORDERS", "OTHERS", "none"):
            if l == "none":
                continue
            break
        m = _LINE.match(l)
        if not m:
            continue
        for part in m.group(2).split(";"):
            toks = part.strip().split()
            if len(toks) >= 3:
                res[f"{toks[0]} {toks[1]}"] = sorted(t.replace("_", " ") for t in toks[2:])
    return res


def canonical(state):
    """Canonical comparable form of a dataset state for exact match."""
    units = {p: {u for u in us} for p, us in (state.get("units", {}) or {}).items() if us}
    centers = {p: set(cs) for p, cs in (state.get("centers", {}) or {}).items() if cs}
    return units, centers


def parsed_canonical(parsed):
    return ({p: s for p, s in parsed["units"].items() if s}, {p: s for p, s in parsed["centers"].items() if s})
