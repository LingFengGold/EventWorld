"""DipNet SL (Paquette et al. 2019) as the non-LLM opponent-model baseline for Task S on the 7p common eval set.
For each (transition, designated power): other powers' orders = DipNet SL given the board (+ previous movement phase as history);
next state = engine adjudication of own recorded orders + DipNet others. Writes Task S prediction JSONL (greedy sample + 4 T=0.1 samples).
"""
import json, os, sys, time
import polars as pl, pandas as pd
from eventworld.dipnet import dipnet_orders_batch
from eventworld.dip_engine import step
from eventworld.dip_serialize import render_task_s_target
D = "data/processed/diplomacy"; OUT = "out/eval/baselines"; os.makedirs(OUT, exist_ok=True)
common = pd.read_parquet(f"{D}/eval_common_7p.parquet")
T = pl.scan_parquet(f"{D}/transitions.parquet").filter(pl.col("game_id").is_in(common.game_id.unique().tolist())).select(
    "sample_id", "game_id", "phase_idx", "phase_type", "map", "rules_no_check", "state_json", "orders_json", "next_state_json").collect().to_pandas()
by_game = {g: d.sort_values("phase_idx") for g, d in T.groupby("game_id")}
rows = T.set_index("sample_id")
t0 = time.time(); n = 0
with open(f"{OUT}/dip_7p_S_dipnet_sl.jsonl", "w") as f:
    for r in common.itertuples():
        tr = rows.loc[r.sample_id]; st = json.loads(tr["state_json"]); orders = json.loads(tr["orders_json"]); ns = json.loads(tr["next_state_json"])
        st_full = {k: st.get(k) for k in ("name", "units", "centers", "retreats", "builds", "homes")}
        rules = ["NO_CHECK"] if tr["rules_no_check"] else []
        prev = by_game[tr["game_id"]]; prev = prev[(prev.phase_idx < tr["phase_idx"]) & (prev.phase_type == "M")].tail(1)
        kw = {}
        if len(prev):
            kw = dict(prev_state={k: json.loads(prev.iloc[0].state_json).get(k) for k in ("name", "units", "centers", "retreats", "builds", "homes")}, prev_orders=json.loads(prev.iloc[0].orders_json))
        try:
            greedy = dipnet_orders_batch(st_full, temperature=0.0, **kw)
            samples = [dipnet_orders_batch(st_full, temperature=0.1, **kw) for _ in range(4)]
        except Exception as e:
            greedy, samples = {}, []
        for pw in json.loads(r.designated):
            texts = []
            for oth in [greedy] + samples:
                joint = {pw: orders.get(pw, [])} | {p: o for p, o in oth.items() if p != pw}
                try:
                    eng_ns, _ = step(st_full, joint, tr["map"], rules)
                except Exception:
                    eng_ns = ns
                texts.append(render_task_s_target(joint, pw, {k: eng_ns.get(k) for k in ("name", "units", "centers", "retreats", "builds")}))
            f.write(json.dumps({"sample_id": f"{r.sample_id}:{pw}", "task": "S", "variant": "dipnet_sl", "samples": [{"text": t} for t in texts], "pred": {"text": texts[0]}}) + "\n"); n += 1
        if n % 500 == 0:
            print(f"{n} samples {time.time()-t0:.0f}s", flush=True)
print(f"wrote {n} Task S predictions in {time.time()-t0:.0f}s")
