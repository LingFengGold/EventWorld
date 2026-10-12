"""Structural scoring for Diplomacy predictions.

A prediction is text in the serializer's target format. For Task J it contains RESULTS (M phases) and NEXT <phase> + STATE.
For Task S it contains OTHERS orders + NEXT + STATE. Scoring parses the text and compares sets; nothing is string-matched.
"""
import json, sys
from eventworld.dip_serialize import parse_state, parse_orders, parse_results, canonical, parsed_canonical
from eventworld.dip_engine import step, legal, is_valid_board


def split_target(text):
    """Return (results_text, others_text, next_state_text)."""
    t = text.strip()
    i = t.find("NEXT ")
    head, tail = (t[:i], t[i:]) if i >= 0 else (t, "")
    ns_txt = "\n".join(tail.splitlines()[1:]) if tail else ""
    return head, ns_txt


def score_state(pred_ns_txt, true_next, map_name="standard"):
    ps = parse_state(pred_ns_txt) if pred_ns_txt else {"units": {}, "centers": {}, "retreats": {}, "builds": {}}
    pu, pc = parsed_canonical(ps); tu, tc = canonical(true_next)
    P = {(p, u) for p, us in pu.items() for u in us}; Tt = {(p, u) for p, us in tu.items() for u in us}
    tp = len(P & Tt); prec = tp / len(P) if P else 0.0; rec = tp / len(Tt) if Tt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    powers = set(tu) | set(tc)
    per_power = {p: (pu.get(p, set()) == tu.get(p, set()) and pc.get(p, set()) == tc.get(p, set())) for p in powers}
    exact = all(per_power.values()) and set(pu) <= powers | set(pu) and pu.keys() <= (set(tu) | set(pc))
    exact = (pu == {p: s for p, s in tu.items() if s}) and (pc == {p: s for p, s in tc.items() if s})
    valid = False
    try:
        valid = is_valid_board({"units": {p: sorted(s) for p, s in pu.items()}}, map_name) if pu else False
    except Exception:
        valid = False
    return dict(unit_f1=f1, unit_precision=prec, unit_recall=rec, phase_exact=float(exact),
                power_exact_rate=sum(per_power.values()) / max(1, len(per_power)), board_valid=float(valid),
                unit_count_conserved=float(len(P) == len(Tt)), parsed=float(bool(pu)))


def score_results(pred_head_txt, true_results):
    pr = parse_results(pred_head_txt) if pred_head_txt else {}
    tr = {u: sorted(map(str, t)) for u, t in (true_results or {}).items() if t}
    if not tr:
        return dict(results_acc=1.0 if not pr else 0.0, results_fp=float(len(pr)))
    acc = sum(1 for u, t in tr.items() if pr.get(u) == t) / len(tr)
    fp = sum(1 for u in pr if u not in tr)
    return dict(results_acc=acc, results_fp=float(fp))


def score_task_j(pred_text, row):
    """row: dict with state, orders, results, next_state (parsed), phase_type, map."""
    head, ns_txt = split_target(pred_text)
    out = score_state(ns_txt, row["next_state"], row["map"])
    if row["phase_type"] == "M" and head.strip().startswith("RESULTS"):  # legacy J format only; protocol v3 targets carry no RESULTS block
        out.update(score_results(head, row["results"]))
    out["parsed"] = float(bool(ns_txt.strip()) and any(parse_state(ns_txt)["units"].values()))
    return out


def score_task_s_direct(pred_text, row):
    """Task S main definition: only the next state is predicted. Same state metrics as Task J."""
    _, ns_txt = split_target(pred_text or "")
    s = score_state(ns_txt, row["next_state"], row["map"])
    s["parsed"] = float(bool(ns_txt.strip()) and any(parse_state(ns_txt)["units"].values()))
    return s


def score_task_s(pred_texts, row, power, rules=None):
    """pred_texts: list of sampled target texts. Returns order top-1, legality, factored (engine) and direct state metrics."""
    st, orders, ns = row["state"], row["orders"], row["next_state"]
    others_true = {p: o for p, o in orders.items() if p != power and o}
    true_by_unit = {" ".join(o.split()[:2]): o for ol in others_true.values() for o in ol}
    leg = None
    top1 = []; illegal = []; extraneous = []; factored = []; direct = []; parsed = []
    for txt in pred_texts:
        head, ns_txt = split_target(txt)
        po = parse_orders(head, header=("OTHERS",))
        parsed.append(float(bool(po)))
        pred_by_unit = {" ".join(o.split()[:2]): o for p, ol in po.items() if p != power for o in ol}
        hits = [pred_by_unit.get(u) == o for u, o in true_by_unit.items()]
        top1.append(sum(hits) / len(hits) if hits else 1.0)
        if leg is None:
            try:
                leg = legal(st, row["map"], rules or [])
            except Exception:
                leg = {}
        def loc_of(o):
            t = o.split(); return t[1] if len(t) >= 2 else None
        located = {u: o for u, o in pred_by_unit.items() if loc_of(o)}
        orderable_preds = {u: o for u, o in located.items() if leg.get(loc_of(o))}
        n_ill = sum(1 for u, o in orderable_preds.items() if o not in leg[loc_of(o)])
        illegal.append(n_ill / max(1, len(orderable_preds)))
        extraneous.append((len(located) - len(orderable_preds)) / max(1, len(located)))
        try:
            joint = {p: list(o) for p, o in orders.items() if p == power}
            for p, ol in po.items():
                if p != power:
                    joint[p] = ol
            eng_ns, _ = step({k: st.get(k) for k in ("name", "units", "centers", "retreats", "builds", "homes")}, joint, row["map"], rules or [])
            factored.append(float(canonical(eng_ns) == canonical(ns)))
        except Exception:
            factored.append(0.0)
        direct.append(score_state(ns_txt, ns, row["map"])["phase_exact"])
    m = lambda a: sum(a) / len(a) if a else 0.0
    return dict(others_top1=m(top1), others_illegal_rate=m(illegal), others_extraneous_rate=m(extraneous), state_exact_factored=m(factored),
                state_exact_direct=m(direct), state_exact_factored_best=max(factored) if factored else 0.0, parsed=m(parsed), n_samples=len(pred_texts))


def score_counterfactual(pred_fact_text, pred_cf_text, true_next_fact, ref_delta):
    """Change-set precision/recall/F1 between the model's predicted delta and the engine's delta."""
    _, a = split_target(pred_fact_text); _, b = split_target(pred_cf_text)
    pa, pb = parse_state(a), parse_state(b)
    def units(ps):
        return {(p, u) for p, s in ps["units"].items() for u in s}
    def cents(ps):
        return {(p, c) for p, s in ps["centers"].items() for c in s}
    pred = {("+", p, u) for p, u in units(pb) - units(pa)} | {("-", p, u) for p, u in units(pa) - units(pb)} \
         | {("+sc", p, c) for p, c in cents(pb) - cents(pa)} | {("-sc", p, c) for p, c in cents(pa) - cents(pb)}
    ref = {tuple(x) for x in ref_delta}
    tp = len(pred & ref); prec = tp / len(pred) if pred else (1.0 if not ref else 0.0); rec = tp / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(cf_precision=prec, cf_recall=rec, cf_f1=f1, cf_exact=float(pred == ref),
                insensitive=float(bool(ref) and not pred), spurious=float(not ref and bool(pred)),
                pred_delta_size=len(pred), ref_delta_size=len(ref))
