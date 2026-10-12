"""Polymarket sample text serializer + parser (P6).

A *sample* is a plain dict (see `example_sample`):
  sample_id, task ("J" | "S" | "SF"), delta ("5m" | "1h" | "24h"), K (0 | 3, history hours), market {market_id, question, event_title, topic},
  t (epoch s), state {p_buy, p_sell, mid, spread, D, N1h, F1h, V1h, age_d, ttl_d (None ok), since_last_s},
  bars   [[close, vol, net, n], ...] hourly, oldest first; K=3 -> the last 3 hours, K=0 -> empty
  recent [[dt, wallet, d, usd, p], ...] history orders within the last K hours (cap RECENT_CAP most recent), oldest first (dt <= 0); K=0 -> empty
  actor  {wallet, pos, n_prev, flow} or None (control window, no designated actor)
  own    [[dt, d, q], ...]            Task S: the actor's orders in the window (empty list = no action)
  joint  [[dt, wallet, d, q], ...]    Task J: every order in the window, chronological
History length K (protocol v3): K=0 is the "lenient state-only" setting: the STATE line (including the scalar summaries D24h and
last-hour N/F/V) and the actor line are kept, but no BARS and no RECENT lines are printed; K=3 adds 3 hourly bars and the orders of the last
3 hours. Task S (main definition) predicts only the next market state given the actor's own orders; "SF" is the factored diagnostic variant
that additionally predicts what other wallets do (others N/F/V) and is not part of the main tables.
Wallets are pseudonymized per sample: actor -> "A", others -> w01, w02, ... in order of first appearance.
Prices print with 4 decimals, usd with 2, dt in whole seconds, bars close 4 / vol 2 / net 2 decimals.
`parse_prompt(render(s))` returns the canonical (rounded, pseudonymized) sample; see `roundtrip_ok`.
Model output contract: JSON {"p_buy","p_sell","p_zero","q10","q50","q90"} (+ "others":{"N","F","V"} for task SF only);
`parse_output` extracts and validates it.
"""
import json, re, datetime as dt

K_HOURS = {0: 0, 1: 1, 3: 3, 6: 6, 10: 10}   # history length in hours per K level (1, 6, 10 added 2026-09-25 for the L2 history sweep)
RECENT_CAP = 20                 # most-recent orders listed within the lookback window for K <= 3
K_DEFAULT = None                # render/canonical print every bar and RECENT order the sample carries (the sample builder applies the caps)
L_BARS = None


def recent_cap(K):
    """Cap on the RECENT list: 20 orders up to K = 3 (protocol v3, unchanged), 7 per hour of lookback beyond that (K = 6 -> 42, K = 10 -> 70)."""
    return RECENT_CAP if K <= 3 else 7 * K
JOINT_CAP = 128
DELTA_TXT = {"5m": "5 minutes", "1h": "1 hour", "24h": "24 hours"}
OUTPUT_SCHEMA_J = '{"p_buy": <last taker buy price at t+H>, "p_sell": <last taker sell price at t+H>, "p_zero": <P(mid unchanged)>, "q10": <10% quantile of mid change>, "q50": <median>, "q90": <90% quantile>}'
OUTPUT_SCHEMA_S = OUTPUT_SCHEMA_J  # main Task S: next state only
OUTPUT_SCHEMA_SF = OUTPUT_SCHEMA_J[:-1] + ', "others": {"N": <number of orders by other wallets>, "F": <their net flow in USD, buy minus sell>, "V": <their volume in USD>}}'


def _isna(x):
    return x is None or (isinstance(x, float) and x != x)


def _p(x):
    return "na" if _isna(x) else f"{x:.4f}"


def _usd(x):
    return f"${x:.2f}"


def _signed_usd(x):
    v = round(float(x), 2)  # sign decided after rounding so that -0.004 prints +$0.00 (parse -> render idempotent)
    return ("+" if v >= 0 else "-") + f"${abs(v):.2f}"


def _d(x):
    return "BUY" if x > 0 else "SELL"


class _Names(dict):
    """wallet id -> pseudonym; already-pseudonymous string ids pass through unchanged."""

    def __missing__(self, k):
        if isinstance(k, str):
            return k
        raise KeyError(k)


def _wid(w):
    return w if isinstance(w, str) else int(w)


def _pseudo(sample):
    names = _Names()
    if sample.get("actor") and not isinstance(sample["actor"]["wallet"], str):
        names[int(sample["actor"]["wallet"])] = "A"
    for o in (sample.get("recent") or []) + (sample.get("joint") or []):
        if not isinstance(o[1], str):
            names.setdefault(int(o[1]), f"w{len(names):02d}")
    return names


def render(sample, K=K_DEFAULT, L=L_BARS):
    s = sample["state"]; mk = sample["market"]; names = _pseudo(sample)
    task = sample["task"]; delta = sample["delta"]
    lines = []
    lines.append(f"MARKET {mk['market_id']} | topic {mk['topic']} | now {dt.datetime.fromtimestamp(sample['t'], dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | horizon {delta}")
    lines.append(f"Q: {mk['question']}")
    lines.append(f"EVENT: {mk.get('event_title') or 'na'}")
    ttl = "na" if _isna(s.get("ttl_d")) else f"{s['ttl_d']:.1f}d"
    lines.append(f"STATE p_buy {_p(s['p_buy'])} p_sell {_p(s['p_sell'])} mid {_p(s['mid'])} spread {_p(s['spread'])} | D24h {_usd(s['D'])} | last1h N {int(s['N1h'])} F {_signed_usd(s['F1h'])} V {_usd(s['V1h'])} | age {s['age_d']:.1f}d ttl {ttl} | last order {int(s['since_last_s'])}s ago")
    bars = sample.get("bars") or []
    if L is not None:
        bars = bars[-L:] if L else []
    if bars:
        lines.append("BARS 1h close/vol/net/n oldest first: " + " ".join(f"{b[0]:.4f}/{b[1]:.2f}/{b[2]:+.2f}/{int(b[3])}" for b in bars))
    recent = sample.get("recent") or []
    if K is not None:
        recent = recent[-K:] if K else []
    if recent:
        lines.append(f"RECENT {len(recent)} orders (dt wallet side usd @price): " + " | ".join(f"{int(o[0])}s {names[_wid(o[1])]} {_d(o[2])} {_usd(o[3])} @{_p(None if o[4] is None or o[4] != o[4] else o[4])}" for o in recent))
    if sample.get("actor"):
        a = sample["actor"]
        lines.append(f"AGENT A | position {a['pos']:+.2f} YES | {int(a['n_prev'])} prior orders | net flow {_signed_usd(a['flow'])}")
    else:
        lines.append("AGENT none (no designated actor)")
    if task in ("S", "SF"):
        own = sample.get("own") or []
        if own:
            lines.append(f"ACTION A ({len(own)} orders, dt side usd): " + " | ".join(f"{int(o[0])}s {_d(o[1])} {_usd(o[2])}" for o in own))
        else:
            lines.append("ACTION A: none")
        if task == "S":
            lines.append(f"TASK S: the agent's orders above are its only orders in the window. Predict the market state after {DELTA_TXT[delta]}.")
            lines.append("OUTPUT JSON " + OUTPUT_SCHEMA_S)
        else:
            lines.append(f"TASK SF: the agent's orders above are its only orders in the window. Predict the market state after {DELTA_TXT[delta]} and what other wallets do in the window.")
            lines.append("OUTPUT JSON " + OUTPUT_SCHEMA_SF)
    else:
        joint = sample.get("joint") or []
        shown = joint
        rest = None
        n_joint = len(joint) + (sample["joint_rest"][0] if sample.get("joint_rest") else 0)
        if sample.get("joint_rest"):
            rest = tuple(sample["joint_rest"])
        if len(joint) > JOINT_CAP:
            top = sorted(range(len(joint)), key=lambda i: -joint[i][3])[:JOINT_CAP]
            keep = sorted(top)
            shown = [joint[i] for i in keep]
            others = [joint[i] for i in range(len(joint)) if i not in set(keep)]
            rest = (len(others), sum(o[2] * o[3] for o in others), sum(o[3] for o in others))
        if shown:
            lines.append(f"JOINT {n_joint} orders in window (dt wallet side usd): " + " | ".join(f"{int(o[0])}s {names[_wid(o[1])]} {_d(o[2])} {_usd(o[3])}" for o in shown))
        else:
            lines.append("JOINT 0 orders in window")
        if rest:
            lines.append(f"JOINT_REST {rest[0]} smaller orders not listed: F {_signed_usd(rest[1])} V {_usd(rest[2])}")
        lines.append(f"TASK J: all orders in the window are given above (execution prices hidden). Predict the market state after {DELTA_TXT[delta]}.")
        lines.append("OUTPUT JSON " + OUTPUT_SCHEMA_J)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- parser -----------------------------------------
_num = r"[-+]?\d+(?:\.\d+)?"


def _parse_p(tok):
    return None if tok == "na" else float(tok)


def _parse_usd(tok):
    tok = tok.replace("$", "")
    return float(tok)


def parse_prompt(text):
    """Inverse of render (on the canonical sample); returns dict with the same keys, wallets as pseudonyms."""
    out = {"state": {}, "market": {}, "bars": [], "recent": [], "actor": None, "own": [], "joint": []}
    for line in text.splitlines():
        if line.startswith("MARKET "):
            m = re.match(r"MARKET (\S+) \| topic (\S+) \| now (.+) UTC \| horizon (\S+)", line)
            out["market"]["market_id"], out["market"]["topic"], now, out["delta"] = m.groups()
            out["t"] = int(dt.datetime.strptime(now, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
        elif line.startswith("Q: "):
            out["market"]["question"] = line[3:]
        elif line.startswith("EVENT: "):
            out["market"]["event_title"] = None if line[7:] == "na" else line[7:]
        elif line.startswith("STATE "):
            m = re.match(rf"STATE p_buy (\S+) p_sell (\S+) mid (\S+) spread (\S+) \| D24h \$({_num}) \| last1h N (\d+) F ([-+])\$({_num}) V \$({_num}) \| age ({_num})d ttl (\S+) \| last order (-?\d+)s ago", line)
            g = m.groups()
            out["state"] = {"p_buy": _parse_p(g[0]), "p_sell": _parse_p(g[1]), "mid": _parse_p(g[2]), "spread": _parse_p(g[3]), "D": float(g[4]),
                            "N1h": int(g[5]), "F1h": float(g[7]) * (1 if g[6] == "+" else -1), "V1h": float(g[8]), "age_d": float(g[9]),
                            "ttl_d": None if g[10] == "na" else float(g[10][:-1]), "since_last_s": int(g[11])}
        elif line.startswith("BARS "):
            body = line.split(": ", 1)[1]
            for tok in body.split():
                c, v, f, n = tok.split("/")
                out["bars"].append([float(c), float(v), float(f), int(n)])
        elif line.startswith("RECENT "):
            body = line.split(": ", 1)[1]
            for tok in body.split(" | "):
                m = re.match(rf"(-?\d+)s (\S+) (BUY|SELL) \$({_num}) @(na|{_num})", tok)
                out["recent"].append([int(m[1]), m[2], 1 if m[3] == "BUY" else -1, float(m[4]), _parse_p(m[5])])
        elif line.startswith("AGENT A"):
            m = re.match(rf"AGENT A \| position ([-+]{_num}) YES \| (\d+) prior orders \| net flow ([-+])\$({_num})", line)
            out["actor"] = {"wallet": "A", "pos": float(m[1]), "n_prev": int(m[2]), "flow": float(m[4]) * (1 if m[3] == "+" else -1)}
        elif line.startswith("ACTION A ("):
            body = line.split(": ", 1)[1]
            out.setdefault("task", "S")
            for tok in body.split(" | "):
                m = re.match(rf"(-?\d+)s (BUY|SELL) \$({_num})", tok)
                out["own"].append([int(m[1]), 1 if m[2] == "BUY" else -1, float(m[3])])
        elif line.startswith("ACTION A: none"):
            out.setdefault("task", "S")
        elif line.startswith("TASK SF:"):
            out["task"] = "SF"
        elif line.startswith("TASK S:"):
            out["task"] = "S"
        elif line.startswith("JOINT ") and not line.startswith("JOINT_REST"):
            out["task"] = "J"
            if ": " in line:
                body = line.split(": ", 1)[1]
                for tok in body.split(" | "):
                    m = re.match(rf"(-?\d+)s (\S+) (BUY|SELL) \$({_num})", tok)
                    out["joint"].append([int(m[1]), m[2], 1 if m[3] == "BUY" else -1, float(m[4])])
        elif line.startswith("JOINT_REST "):
            m = re.match(rf"JOINT_REST (\d+) smaller orders not listed: F ([-+])\$({_num}) V \$({_num})", line)
            out["joint_rest"] = [int(m[1]), float(m[3]) * (1 if m[2] == "+" else -1), float(m[4])]
    return out


def canonical(sample, K=K_DEFAULT, L=L_BARS):
    """The sample as the renderer/parser see it: rounded numbers, pseudonymous wallets, truncated lists."""
    names = _pseudo(sample); s = sample["state"]
    r4 = lambda x: None if x is None else round(float(x), 4)
    r2 = lambda x: round(float(x), 2)
    c = {"t": int(sample["t"]), "delta": sample["delta"], "task": sample["task"],
         "market": {"market_id": str(sample["market"]["market_id"]), "topic": sample["market"]["topic"], "question": sample["market"]["question"],
                    "event_title": sample["market"].get("event_title") or None},
         "state": {"p_buy": r4(s["p_buy"]), "p_sell": r4(s["p_sell"]), "mid": r4(s["mid"]), "spread": r4(s["spread"]), "D": r2(s["D"]), "N1h": int(s["N1h"]),
                   "F1h": r2(s["F1h"]), "V1h": r2(s["V1h"]), "age_d": round(float(s["age_d"]), 1), "ttl_d": None if _isna(s.get("ttl_d")) else round(float(s["ttl_d"]), 1),
                   "since_last_s": int(s["since_last_s"])},
         "bars": [[r4(b[0]), r2(b[1]), r2(b[2]), int(b[3])] for b in ((sample.get("bars") or []) if L is None else ((sample.get("bars") or [])[-L:] if L else []))],
         "recent": [[int(o[0]), names[_wid(o[1])], int(o[2]), r2(o[3]), r4(None if o[4] is None or o[4] != o[4] else o[4])] for o in ((sample.get("recent") or []) if K is None else ((sample.get("recent") or [])[-K:] if K else []))],
         "actor": None if not sample.get("actor") else {"wallet": "A", "pos": r2(sample["actor"]["pos"]), "n_prev": int(sample["actor"]["n_prev"]), "flow": r2(sample["actor"]["flow"])},
         "own": [[int(o[0]), int(o[1]), r2(o[2])] for o in (sample.get("own") or [])] if sample["task"] in ("S", "SF") else [],
         "joint": [[int(o[0]), names[_wid(o[1])], int(o[2]), r2(o[3])] for o in (sample.get("joint") or [])] if sample["task"] == "J" else []}
    if sample["task"] == "J" and len(c["joint"]) > JOINT_CAP:
        joint = c["joint"]
        keep = sorted(sorted(range(len(joint)), key=lambda i: -joint[i][3])[:JOINT_CAP])
        ks = set(keep)
        others = [joint[i] for i in range(len(joint)) if i not in ks]
        c["joint_rest"] = [len(others), r2(sum(o[2] * o[3] for o in others)), r2(sum(o[3] for o in others))]
        c["joint"] = [joint[i] for i in keep]
    return c


def _close(a, b, tol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
    return a == b


def roundtrip_ok(sample, K=K_DEFAULT, L=L_BARS):
    c = canonical(sample, K, L)
    p = parse_prompt(render(sample, K, L))
    # the "-0.00" sign of a zero net flow is not recoverable; normalise
    return _close(json.loads(json.dumps(c)), json.loads(json.dumps(p))), c, p


# ---------------------------------------------------------------- output ------------------------------------------
KEYS = ["p_buy", "p_sell", "p_zero", "q10", "q50", "q90"]


def parse_output(text, task="J"):
    """Extract the first JSON object; return dict with KEYS (+ others for task SF) or None if unparsable."""
    if text is None:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return _validate(obj, task)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        break
        start = text.find("{", start + 1)
    return None


def _validate(obj, task):
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in KEYS:
        v = obj.get(k)
        if v is None or isinstance(v, bool):
            return None
        out[k] = float(v)
    out["p_zero"] = min(1.0, max(0.0, out["p_zero"]))
    for k in ["p_buy", "p_sell"]:
        out[k] = min(1.0, max(0.0, out[k]))
    q = sorted([out["q10"], out["q50"], out["q90"]])
    out["q10"], out["q50"], out["q90"] = q
    if task == "SF":
        o = obj.get("others") or {}
        try:
            out["others"] = {"N": float(o.get("N", 0)), "F": float(o.get("F", 0)), "V": float(o.get("V", 0))}
        except (TypeError, ValueError):
            out["others"] = {"N": 0.0, "F": 0.0, "V": 0.0}
    return out


def example_sample():
    return {"sample_id": "x", "task": "S", "delta": "1h", "t": 1750000000,
            "market": {"market_id": "12345", "question": "Will X happen by June 30?", "event_title": "X event", "topic": "politics"},
            "state": {"p_buy": 0.32, "p_sell": 0.31, "mid": 0.315, "spread": 0.01, "D": 12400.5, "N1h": 12, "F1h": 340.25, "V1h": 1200.0, "age_d": 84.2, "ttl_d": 31.5, "since_last_s": 183},
            "bars": [[0.31, 2100.0, 400.0, 5], [0.31, 900.0, -100.0, 3]],
            "recent": [[-183, 77, 1, 120.0, 0.32], [-60, 91, -1, 40.0, 0.31]],
            "actor": {"wallet": 77, "pos": 1240.0, "n_prev": 7, "flow": 800.0},
            "own": [[0, 1, 8400.0], [120, 1, 1000.0]],
            "joint": [[0, 77, 1, 8400.0], [60, 5, -1, 40.0], [120, 77, 1, 1000.0]]}


if __name__ == "__main__":
    s = example_sample()
    print(render(s))
    ok, c, p = roundtrip_ok(s)
    print("roundtrip S:", ok)
    s["task"] = "J"
    print(render(s))
    ok, c, p = roundtrip_ok(s)
    print("roundtrip J:", ok)
    if not ok:
        print(c); print(p)
    print(parse_output('Sure! {"p_buy": 0.33, "p_sell": 0.32, "p_zero": 0.4, "q10": -0.01, "q50": 0.005, "q90": 0.02, "others": {"N": 5, "F": -100, "V": 300}}', "SF"))


# ---- few-shot as realised outcomes (protocol v3) -------------------------------------------------------------------------------
# Probabilistic targets have no per-sample truth, so demonstrations are NOT shown as model answers. The demos' input blocks are shown inside the
# user prompt together with what actually happened (end prices and realised mid change only, 4 decimals); the query keeps its TASK / OUTPUT lines.
def _split_task(prompt):
    i = prompt.index("\nTASK ")
    return prompt[:i], prompt[i + 1:]


def outcome_line(tf):
    return f"OUTCOME p_buy {float(tf['p_buy_end']):.4f} p_sell {float(tf['p_sell_end']):.4f} dmid {float(tf['dmid']):+.4f}"


def fewshot_outcomes_user(query_prompt, demos):
    """demos: iterable of demo records with 'prompt' and 'target_fields' (p_buy_end, p_sell_end, dmid). Returns the composed user prompt."""
    body, tail = _split_task(query_prompt)
    parts = [f"Here are {len(demos)} past windows from other markets and what actually happened afterwards."]
    for i, d in enumerate(demos, 1):
        db, _ = _split_task(d["prompt"])
        parts += [f"EXAMPLE {i}", db, outcome_line(d["target_fields"])]
    parts += ["Now the current window:", body, tail]
    return "\n".join(parts)


# ---- SFT output format: binned Delta-mid (protocol v3) --------------------------------------------------------------------------
# Probabilistic fields have no per-sample truth, so SFT targets contain only realised quantities: a bin letter for Delta-mid, the point Delta-mid,
# and the end prices. At inference the distribution comes from the model's probabilities over the 21 bin letters (single tokens for Qwen3):
# p_zero = P(zero bin), q10/q90 = quantiles of the bin mixture (zero bin = point mass, other bins uniform), q50 = generated point Delta-mid.
SFT_BIN_EDGES = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.1, 0.2]   # |Delta-mid| edges; 10 magnitude bins per sign + an exact-zero bin
SFT_BIN_MAX = 0.5
SFT_LETTERS = [chr(ord("A") + i) for i in range(21)]                        # A = most negative ... K = exactly zero ... U = most positive
SFT_OUTPUT_LINE = "OUTPUT four lines: BIN <letter A-U for the mid change, K = unchanged, earlier letters = larger falls, later letters = larger rises> / DMID <signed mid change> / P_BUY <price> / P_SELL <price>"


def sft_bin_index(x):
    """Delta-mid -> bin 0..20 (10 = exactly zero)."""
    x = float(x)
    if x == 0:
        return 10
    mag = sum(1 for e in SFT_BIN_EDGES if abs(x) > e) + 1
    return 10 + mag if x > 0 else 10 - mag


def sft_bin_bounds(b):
    """(lo, hi) of |Delta-mid| for bin b (signless magnitude interval), zero bin -> (0, 0)."""
    mag = abs(b - 10)
    if mag == 0:
        return 0.0, 0.0
    e = [0.0] + SFT_BIN_EDGES + [SFT_BIN_MAX]
    return e[mag - 1], e[mag]


SFT_OUTPUT_LINE_SF = ("OUTPUT five lines: OTHERS N <number of orders by other wallets in the window> F <their net flow in USD, buy minus sell> "
                      "V <their volume in USD> / " + SFT_OUTPUT_LINE[len("OUTPUT four lines: "):])


def sft_prompt(prompt):
    """Eval/SFT prompt with the JSON output line replaced by the binned output spec (the five-line SF spec for Task SF prompts)."""
    lines = prompt.rstrip("\n").split("\n")
    if lines and lines[-1].startswith("OUTPUT JSON"):
        lines[-1] = SFT_OUTPUT_LINE_SF if '"others"' in lines[-1] else SFT_OUTPUT_LINE
    return "\n".join(lines)


def sft_target(tf):
    """tf: target fields with p_buy_end, p_sell_end, dmid. All values realised; 4 decimals."""
    dm = float(tf["dmid"])
    return f"BIN {SFT_LETTERS[sft_bin_index(dm)]}\nDMID {dm:+.4f}\nP_BUY {float(tf['p_buy_end']):.4f}\nP_SELL {float(tf['p_sell_end']):.4f}"


def sft_target_sf(tf):
    """Task SF target: the other wallets' order count, net flow and volume in the window, then the four binned lines."""
    return f"OTHERS N {int(round(float(tf['N_o'])))} F {float(tf['F_o']):+.2f} V {max(0.0, float(tf['V_o'])):.2f}\n" + sft_target(tf)


def sft_decode_others(text):
    """'OTHERS N <n> F <f> V <v>' line of a Task SF output -> {"N","F","V"} (None if absent)."""
    m = re.search(r"OTHERS\s+N\s*([+-]?\d*\.?\d+)\s+F\s*([+-]?\d*\.?\d+)\s+V\s*([+-]?\d*\.?\d+)", text or "")
    return {"N": float(m.group(1)), "F": float(m.group(2)), "V": float(m.group(3))} if m else None


def sft_decode(text, bin_probs=None):
    """Generated text (starting after 'BIN') + optional probabilities over the 21 letters -> the six-key prediction dict (None if unparsable)."""
    vals = {}
    for key, pat in (("dmid", r"DMID\s*([+-]?\d*\.?\d+)"), ("p_buy", r"P_BUY\s*(\d*\.?\d+)"), ("p_sell", r"P_SELL\s*(\d*\.?\d+)"), ("bin", r"^\s*(?:BIN\s*)?([A-U])\b")):
        m = re.search(pat, text, re.M)
        if m:
            vals[key] = m.group(1)
    if not {"dmid", "p_buy", "p_sell"} <= vals.keys():
        return None
    dm = float(vals["dmid"])
    if bin_probs is None:  # no logprobs: degenerate distribution at the generated bin
        bin_probs = [0.0] * 21; bin_probs[SFT_LETTERS.index(vals["bin"]) if "bin" in vals else sft_bin_index(dm)] = 1.0
    tot = sum(bin_probs) or 1.0; p = [x / tot for x in bin_probs]
    # mixture CDF on a grid: zero bin = point mass at 0, other bins uniform over their signed interval
    import numpy as np
    grid = np.linspace(-SFT_BIN_MAX, SFT_BIN_MAX, 20001); F = np.zeros_like(grid)
    for b, pb in enumerate(p):
        if pb <= 0:
            continue
        if b == 10:
            F += pb * (grid >= 0)
            continue
        lo, hi = sft_bin_bounds(b)
        a, c = (lo, hi) if b > 10 else (-hi, -lo)
        F += pb * np.clip((grid - a) / (c - a), 0, 1)
    q = lambda u: float(grid[min(np.searchsorted(F, u, side="left"), len(grid) - 1)])
    q10, q90 = q(0.1), q(0.9)
    out = {"p_buy": min(1.0, max(0.0, float(vals["p_buy"]))), "p_sell": min(1.0, max(0.0, float(vals["p_sell"]))), "p_zero": float(p[10]),
           "q10": q10, "q50": min(max(dm, q10), q90), "q90": q90, "bin_probs": [round(x, 5) for x in p]}
    return out
