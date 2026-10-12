"""Deep per-file statistics over the diplomacy/research dataset (streaming, multiprocess).
Writes out/dip_deep/<file>.games.csv (one row per game) and out/dip_deep/<file>.agg.json."""
import json, sys, os, re, collections, csv
from multiprocessing import Pool
import numpy as np

DATA = "data/diplomacy"
OUT = "out/dip_deep"; os.makedirs(OUT, exist_ok=True)
FILES = ["standard_no_press","standard_public_press","standard_press_with_msgs","other_maps","standard_press_without_msgs"]
NW = 12  # workers per file

ORDER_RE = re.compile(r"^[AF] [A-Z]{3}(?:/[A-Z]{2})?(?: (.*))?$")

def order_type(o):
    # returns H, -, VIA, S, C, R, B, D, or ?
    if o.endswith(" VIA"): return "VIA"
    parts = o.split()
    if len(parts) < 3: return "?"
    t = parts[2]
    return t if t in ("H","-","S","C","R","B","D") else "?"

def game_row(g):
    ph = g["phases"]
    r = collections.OrderedDict()
    r["id"] = g["id"]; r["map"] = g["map"]; r["rules"] = "|".join(sorted(g["rules"]))
    r["n_phases"] = len(ph)
    types = collections.Counter(p["name"][-1] for p in ph)
    r["n_M"], r["n_R"], r["n_A"] = types.get("M",0), types.get("R",0), types.get("A",0)
    last = ph[-1]["name"]; r["last_phase"] = last
    r["last_year"] = int(last[1:5]) if last[1:5].isdigit() else -1
    first = ph[0]["name"]; r["first_year"] = int(first[1:5]) if first[1:5].isdigit() else -1
    ts = [p["state"].get("timestamp") for p in ph]
    ts = [t for t in ts if t]
    r["ts_first"] = ts[0] if ts else 0; r["ts_last"] = ts[-1] if ts else 0
    r["ts_span_s"] = (ts[-1]-ts[0])/1e6 if len(ts) > 1 else 0
    powers = list(ph[0]["state"]["units"].keys()); r["n_powers"] = len(powers)
    st_end = ph[-1]["state"]
    cen = {pw: len(st_end["centers"].get(pw, [])) for pw in powers}
    r["max_centers"] = max(cen.values()); r["winner"] = max(cen, key=cen.get)
    r["n_alive_end"] = sum(1 for pw in powers if cen[pw] > 0 or st_end["units"].get(pw))
    r["n_eliminated"] = r["n_powers"] - r["n_alive_end"]
    r["final_centers"] = "|".join(f"{pw}:{cen[pw]}" for pw in powers)
    # civil disorder: any phase where a power flagged
    cd = set()
    for p in ph:
        for pw, v in (p["state"].get("civil_disorder") or {}).items():
            if v: cd.add(pw)
    r["n_cd_powers"] = len(cd)
    # orders
    oc = collections.Counter(); res = collections.Counter()
    n_units_M = 0; n_orders_M = 0; n_missing_M = 0; n_powers_noorders = 0; n_powerphases_M = 0
    for p in ph:
        orders = p.get("orders") or {}
        for pw, ords in orders.items():
            for o in (ords or []): oc[order_type(o)] += 1
        if p["name"].endswith("M"):
            for pw in powers:
                units = p["state"]["units"].get(pw, [])
                nu = sum(1 for u in units if not u.startswith("*"))
                if nu == 0: continue
                n_powerphases_M += 1
                no = len(orders.get(pw) or [])
                n_units_M += nu; n_orders_M += no
                if no == 0: n_powers_noorders += 1
                n_missing_M += max(0, nu - no)
        for u, tags in (p.get("results") or {}).items():
            for t in tags:
                if t: res[t] += 1
    for k in ("H","-","VIA","S","C","R","B","D","?"): r["ord_"+k] = oc.get(k,0)
    r["units_M"] = n_units_M; r["orders_M"] = n_orders_M; r["missing_orders_M"] = n_missing_M
    r["powerphases_M"] = n_powerphases_M; r["powerphases_M_noorders"] = n_powers_noorders
    for k in ("bounce","cut","void","dislodged","disband","no convoy"): r["res_"+k] = res.get(k,0)
    r["res_other"] = sum(v for k,v in res.items() if k not in ("bounce","cut","void","dislodged","disband","no convoy"))
    # messages
    msgs = [m for p in ph for m in (p.get("messages") or [])]
    r["n_msgs"] = len(msgs)
    r["n_msgs_global"] = sum(1 for m in msgs if m.get("recipient") == "GLOBAL")
    r["msg_chars"] = sum(len(m.get("message") or "") for m in msgs)
    r["n_msg_unique"] = len(set(m.get("message") for m in msgs))
    r["n_msg_senders"] = len(set(m.get("sender") for m in msgs))
    mt = [m.get("time_sent") for m in msgs if m.get("time_sent")]
    r["msg_t_first"] = min(mt) if mt else 0; r["msg_t_last"] = max(mt) if mt else 0
    r["n_phases_with_msgs"] = sum(1 for p in ph if p.get("messages"))
    return r

def worker(args):
    fn, rank = args
    rows = []; n = 0
    with open(f"{DATA}/{fn}.jsonl", "rb") as f:
        for i, line in enumerate(f):
            if i % NW != rank: continue
            try:
                rows.append(game_row(json.loads(line)))
            except Exception as e:
                rows.append({"id": f"ERR-{i}", "err": repr(e)[:200]})
            n += 1
    return rows

def pct(a, q): return float(np.percentile(a, q)) if len(a) else None

def run(fn):
    with Pool(NW) as pool:
        parts = pool.map(worker, [(fn, k) for k in range(NW)])
    rows = [r for p in parts for r in p]
    errs = [r for r in rows if "err" in r]; rows = [r for r in rows if "err" not in r]
    keys = list(rows[0].keys())
    with open(f"{OUT}/{fn}.games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    A = lambda k: np.array([r[k] for r in rows])
    nph = A("n_phases"); ly = A("last_year"); nm = A("n_msgs")
    agg = {"file": fn, "games": len(rows), "errors": len(errs), "err_samples": [e["err"] for e in errs[:5]],
           "unique_ids": len(set(r["id"] for r in rows)),
           "maps": collections.Counter(r["map"] for r in rows).most_common(30),
           "rules": collections.Counter(r["rules"] for r in rows).most_common(30),
           "n_powers": collections.Counter(r["n_powers"] for r in rows).most_common(),
           "phases": {"p10": pct(nph,10), "p50": pct(nph,50), "p90": pct(nph,90), "max": int(nph.max()), "mean": float(nph.mean()), "sum": int(nph.sum()),
                      "ge20": int((nph>=20).sum()), "ge40": int((nph>=40).sum()), "ge60": int((nph>=60).sum())},
           "M_phases_total": int(A("n_M").sum()), "R_phases_total": int(A("n_R").sum()), "A_phases_total": int(A("n_A").sum()),
           "last_year": {"<=1902": int((ly<=1902).sum()), "1903-1905": int(((ly>=1903)&(ly<=1905)).sum()), "1906-1910": int(((ly>=1906)&(ly<=1910)).sum()), ">1910": int((ly>1910).sum()), "p50": pct(ly,50), "p90": pct(ly,90), "max": int(ly.max())},
           "first_year_not_1901": int((A("first_year")!=1901).sum()),
           "solo_18plus": int((A("max_centers")>=18).sum()), "max_centers_dist": collections.Counter(int(x) for x in A("max_centers")).most_common(),
           "n_eliminated_dist": collections.Counter(int(x) for x in A("n_eliminated")).most_common(),
           "cd_powers_dist": collections.Counter(int(x) for x in A("n_cd_powers")).most_common(),
           "orders": {k: int(A("ord_"+k).sum()) for k in ("H","-","VIA","S","C","R","B","D","?")},
           "results": {k: int(A("res_"+k).sum()) for k in ("bounce","cut","void","dislodged","disband","no convoy","other")},
           "units_M": int(A("units_M").sum()), "orders_M": int(A("orders_M").sum()), "missing_orders_M": int(A("missing_orders_M").sum()),
           "powerphases_M": int(A("powerphases_M").sum()), "powerphases_M_noorders": int(A("powerphases_M_noorders").sum()),
           "ts": {"span_s_p50": pct(A("ts_span_s"),50), "span_s_p90": pct(A("ts_span_s"),90), "span_s_max": float(A("ts_span_s").max()),
                  "first_min": int(A("ts_first").min()), "first_max": int(A("ts_first").max())},
           "msgs": {"games_with_msgs": int((nm>0).sum()), "total": int(nm.sum()), "global": int(A("n_msgs_global").sum()), "chars": int(A("msg_chars").sum()),
                    "unique": int(A("n_msg_unique").sum()),
                    "per_game_p50": pct(nm[nm>0],50) if (nm>0).any() else None, "per_game_p90": pct(nm[nm>0],90) if (nm>0).any() else None, "per_game_max": int(nm.max()),
                    "t_min": int(A("msg_t_first")[A("msg_t_first")>0].min()) if (A("msg_t_first")>0).any() else None,
                    "t_max": int(A("msg_t_last").max()),
                    "senders_dist": collections.Counter(int(x) for x in A("n_msg_senders")).most_common()},
           }
    with open(f"{OUT}/{fn}.agg.json", "w") as f: json.dump(agg, f, indent=1, default=str)
    print("DONE", fn, len(rows), flush=True)

if __name__ == "__main__":
    for fn in (sys.argv[1:] or FILES): run(fn)
