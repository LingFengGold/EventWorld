"""D2: one row per non-terminal phase of every kept no-press game, with engine check and difficulty features."""
import json, re, sys, time
from multiprocessing import Pool
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from eventworld.dip_engine import step

IDX = pd.read_parquet("data/processed/diplomacy/games_index.parquet").set_index("game_id")
OUT = "data/processed/diplomacy/transitions.parquet"
MOVE = re.compile(r"^([AF]) (\S+) - (\S+)( VIA)?$")

def parse_orders(orders):
    """Return dict loc->(unit_type, kind, dest) for movement-phase orders."""
    out = {}
    for p, ol in (orders or {}).items():
        for o in ol or []:
            t = o.split()
            if len(t) < 2 or t[0] not in ("A", "F"):
                continue
            typ, loc = t[0], t[1]
            if len(t) == 2 or t[2] == "H":
                kind, dest = "H", None
            elif t[2] == "-":
                kind, dest = "M", t[3] if len(t) > 3 else None
                if "VIA" in t:
                    kind = "MV"
            elif t[2] == "S":
                kind, dest = "S", None
            elif t[2] == "C":
                kind, dest = "C", None
            elif t[2] in ("R", "D", "B"):
                kind, dest = t[2], t[3] if len(t) > 3 else None
            else:
                kind, dest = "?", None
            out[loc] = (typ, kind, dest, p)
    return out

def prov(loc):
    return loc.split("/")[0]

def difficulty(state, orders, results, next_state):
    po = parse_orders(orders)
    occupied = {prov(u.split(" ", 1)[1].lstrip("*")): p for p, us in state["units"].items() for u in us}
    moves = {l: prov(v[2]) for l, v in po.items() if v[1] in ("M", "MV") and v[2]}
    dest_count = {}
    for l, d in moves.items():
        dest_count[d] = dest_count.get(d, 0) + 1
    contested = set()
    for l, d in moves.items():
        if dest_count[d] >= 2:
            contested.add(d)
        elif d in occupied:
            occ_loc = next((ll for ll in po if prov(ll) == d), None)
            if occ_loc is None or po[occ_loc][1] not in ("M", "MV"):
                contested.add(d)           # moving into a unit that stays
            elif moves.get(occ_loc) == prov(l):
                contested.add(d); contested.add(prov(l))   # head-to-head
    in_conflict = sum(1 for l, d in moves.items() if d in contested) + sum(1 for l in po if prov(l) in contested and po[l][1] not in ("M", "MV"))
    tags = {}
    for u, rs in (results or {}).items():
        for r in rs or []:
            tags[r] = tags.get(r, 0) + 1
    def units(s):
        return {u.lstrip("*") for us in s["units"].values() for u in us}
    n_changed = len(units(state) ^ units(next_state))
    kinds = [v[1] for v in po.values()]
    n_convoy = sum(1 for k in kinds if k in ("C", "MV"))
    n_via = kinds.count("MV")
    n_dis = tags.get("dislodged", 0)
    n_nc = tags.get("no convoy", 0)
    if n_dis >= 2 or n_nc > 0 or (n_via > 0 and n_dis > 0):
        tier = 3          # multiple dislodgements or convoy-relevant adjudication
    elif tags.get("cut", 0) > 0 or n_dis > 0:
        tier = 2          # a support cut or a single dislodgement
    elif contested:
        tier = 1          # contested provinces resolved by simple strength comparison
    else:
        tier = 0          # no contact
    return dict(n_move=sum(1 for k in kinds if k in ("M", "MV")), n_support=kinds.count("S"), n_convoy_orders=n_convoy, n_via=n_via,
                n_hold=kinds.count("H"), n_contested=len(contested), n_units_in_conflict=in_conflict,
                n_bounce=tags.get("bounce", 0), n_cut=tags.get("cut", 0), n_dislodged=n_dis, n_void=tags.get("void", 0),
                n_noconvoy=tags.get("no convoy", 0), n_changed_units=n_changed, difficulty_tier=tier)

def process_game(line):
    g = json.loads(line)
    gid = g["id"]
    if gid not in IDX.index:
        return []
    meta = IDX.loc[gid]
    rules = g.get("rules", [])
    rows = []
    ph = g["phases"]
    for k in range(len(ph) - 1):
        p, nxt = ph[k], ph[k + 1]
        name = p["name"]
        if name == "COMPLETED" or not name[-1] in "MRA":
            continue
        st = p["state"]; ns = nxt["state"]
        st_min = {kk: st.get(kk) for kk in ("name", "units", "centers", "retreats", "builds", "homes")}
        ns_min = {kk: ns.get(kk) for kk in ("name", "units", "centers", "retreats", "builds")}
        try:
            eng_ns, eng_res = step(st_min | {"homes": st.get("homes")}, p["orders"], g["map"], rules)
            eng_ok = all(sorted(eng_ns["units"].get(pw, [])) == sorted(ns["units"].get(pw, [])) for pw in ns["units"]) and \
                     all(sorted(eng_ns["centers"].get(pw, [])) == sorted(ns["centers"].get(pw, [])) for pw in ns["centers"])
            res_match = None
            if name.endswith("M"):
                rec = {u: sorted(r) for u, r in (p["results"] or {}).items() if r}
                eng = {u: sorted(r) for u, r in eng_res.items() if r}
                res_match = rec == eng
        except Exception as e:
            eng_ok, res_match = False, None
        alive = [pw for pw, us in st["units"].items() if us]
        n_noorders = sum(1 for pw in alive if not (p["orders"] or {}).get(pw))
        row = dict(sample_id=f"{gid}:{k}", game_id=gid, pop=meta["pop"], split=meta["split"], scale_bucket=meta["scale_bucket"],
                   map=g["map"], rules_no_check=bool(meta["no_check"]), phase_idx=k, phase_name=name, phase_type=name[-1],
                   year=int(name[1:5]), season=name[0], is_primary=name.endswith("M"),
                   state_json=json.dumps(st_min, separators=(",", ":")), orders_json=json.dumps(p["orders"] or {}, separators=(",", ":")),
                   results_json=json.dumps(p["results"] or {}, separators=(",", ":")), next_state_json=json.dumps(ns_min, separators=(",", ":")),
                   engine_ok=bool(eng_ok), results_match_M=res_match,
                   n_units_total=sum(len(us) for us in st["units"].values()), n_powers_alive=len(alive), n_powers_no_orders=n_noorders)
        if name.endswith("M"):
            row.update(difficulty(st, p["orders"], p["results"], ns))
        rows.append(row)
    return rows

def main():
    t0 = time.time()
    writer = None; n = 0
    with open("data/diplomacy/standard_no_press.jsonl") as f, Pool(32) as pool:
        batch = []
        for rows in pool.imap_unordered(process_game, f, chunksize=8):
            batch.extend(rows)
            if len(batch) >= 20000:
                tbl = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(OUT, tbl.schema, compression="zstd")
                writer.write_table(tbl); n += len(batch); batch = []
                print(f"  {n:,} rows  {time.time()-t0:.0f}s", flush=True)
        if batch:
            tbl = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT, tbl.schema, compression="zstd")
            writer.write_table(tbl); n += len(batch)
    writer.close()
    print(f"wrote {n:,} rows to {OUT} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
