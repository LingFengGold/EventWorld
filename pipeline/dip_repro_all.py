# Full engine-reproduction check over ALL games in the given jsonl files, in parallel.
# Writes out/dip_deep/repro_all.csv (one row per game) and prints a per-file summary.
# Usage: python pipeline/dip_repro_all.py standard_no_press standard_press_with_msgs ...
import json, sys, csv, os, collections
from multiprocessing import Pool
from diplomacy import Game

def check(args):
    fn, line = args
    g = json.loads(line); ph = g["phases"]; n = max(0, len(ph) - 1)
    try:
        game = Game(map_name=g["map"], rules=g["rules"])
        for k in range(n):
            p = ph[k]
            if game.get_current_phase() != p["name"]:
                return (fn, g["id"], g["map"], "|".join(g["rules"]), n, k, "phase-name", p["name"])
            for pw, ords in (p["orders"] or {}).items():
                if ords: game.set_orders(pw, ords)
            game.process(); st = game.get_state(); nxt = ph[k + 1]["state"]
            same = all(sorted(st["units"].get(pw, [])) == sorted(nxt["units"].get(pw, [])) for pw in nxt["units"]) and \
                   all(sorted(st["centers"].get(pw, [])) == sorted(nxt["centers"].get(pw, [])) for pw in nxt["centers"])
            if not same:
                return (fn, g["id"], g["map"], "|".join(g["rules"]), n, k, "state", p["name"])
        return (fn, g["id"], g["map"], "|".join(g["rules"]), n, n, "ok", "")
    except Exception as e:
        return (fn, g["id"], g.get("map"), "|".join(g.get("rules", [])), n, -1, "exception", repr(e)[:80])

def lines(fn):
    with open(f"data/diplomacy/{fn}.jsonl") as f:
        for line in f: yield (fn, line)

if __name__ == "__main__":
    files = sys.argv[1:]
    os.makedirs("out/dip_deep", exist_ok=True)
    with open("out/dip_deep/repro_all.csv", "w", newline="") as out:
        w = csv.writer(out); w.writerow(["file", "id", "map", "rules", "n_transitions", "ok_transitions", "status", "fail_phase"])
        with Pool(48) as pool:
            for fn in files:
                summ = collections.Counter(); tr_ok = tr_tot = 0; games = 0
                for row in pool.imap_unordered(check, lines(fn), chunksize=16):
                    w.writerow(row); games += 1; summ[row[6]] += 1; tr_tot += row[4]; tr_ok += max(0, row[5])
                out.flush()
                print(f"{fn}: games {summ['ok']}/{games} fully reproduced ({100*summ['ok']/max(1,games):.2f}%); "
                      f"transitions matched before first failure {tr_ok}/{tr_tot}; failures {dict((k,v) for k,v in summ.items() if k!='ok')}", flush=True)
