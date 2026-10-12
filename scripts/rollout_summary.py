"""Summary tables of the L1 rollout experiments (Task J vs Task S, all models) from out/rollout/*/<tag>/per_step.parquet / per_start.parquet.
Writes out/build/L1_rollout_summary.md. Missing runs are skipped."""
import os, numpy as np, pandas as pd
R = "out/rollout"
MODELS = [("Qwen3-1.7B SFT", "sft1.7b"), ("Qwen3-8B SFT", "sft8b"), ("GPT-6-Luna", "gpt-6-luna"), ("GPT-6-Sol", "gpt-6-sol")]
# Diplomacy: plain tag = Task J, "_S" = Task S. Polymarket: plain tag = Task S, "_J" = Task J.
tag = lambda dom, base, task: base + ("" if (dom == "dip") == (task == "J") else f"_{task}")
out = ["# L1 rollout summary: Task J vs Task S (K=0)", ""]

out += ["## Diplomacy teacher-forced 5-step chains (500 chains)", "",
        "| Task | Model | EM closed @1 / @5 | D_5 | unit F1 closed @1 → @5 | unit F1 open (mean) | board valid closed (mean) | next-order void rate closed (mean) |", "|---|---|---|---|---|---|---|---|"]
for task in "JS":
    for name, base in MODELS:
        p = f"{R}/diplomacy_chains/{tag('dip', base, task)}/per_step.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p); op, cl = d[d["mode"] == "open"], d[d["mode"] == "closed"]
        em = cl.groupby("step").phase_exact.mean(); emo = op.groupby("step").phase_exact.mean(); f1 = cl.groupby("step").unit_f1.mean()
        d5 = em[5] / emo[5] if emo[5] > 0 else np.nan
        out.append(f"| {task} | {name} | {em[1]:.3f} / {em[5]:.3f} | {d5:.3f} | {f1[1]:.3f} → {f1[5]:.3f} | {op.unit_f1.mean():.3f} | {cl.board_valid.mean():.3f} | {cl.void_next.mean():.3f} |")

out += ["", "## Diplomacy CR (500 starts x 10 phases, DipNet SL greedy, success = dSC >= 0; Real = 0.650)", "",
        "In Task J the world model is given every power's orders; in Task S only the agent's, and it must imagine the opponents.", "",
        "| Task | Model | WM | W2R | CR = W2R/Real | WM/Real | WM completed | imagination gap (SC) | mean dSC WM / W2R | replay void rate |", "|---|---|---|---|---|---|---|---|---|---|"]
for task in "JS":
    for name, base in MODELS:
        p = f"{R}/diplomacy_cr/{tag('dip', base, task)}/per_start.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p); real = (d.dsc_real >= 0).mean(); wm = ((d.dsc_wm >= 0) & d.wm_done).mean(); w2r = (d.dsc_w2r >= 0).mean(); dn = d[d.wm_done]
        out.append(f"| {task} | {name} | {wm:.3f} | {w2r:.3f} | {w2r / real:.3f} | {wm / real:.3f} | {d.wm_done.mean():.3f} | {(dn.dsc_wm - dn.dsc_w2r).mean():+.2f} | "
                   f"{dn.dsc_wm.mean():+.2f} / {d.dsc_w2r.mean():+.2f} | {d.void.sum() / max(1, d.n_replayed.sum()):.3f} |")

out += ["", "## Polymarket 24-step hourly rollouts (500 chains)", "",
        "No-change forecast from t0: MAE 0.0040 at 1 h, 0.0099 at 24 h.", "",
        "| Task | Model | MAE closed @1 / @24 | MAE open (steps 2-24) | closed/open @24 | CRPS closed @24 | cov80 closed @1 → @24 | cov80 open (steps 2-24) | direct 24 h MAE | direct / closed @24 | direct cov80 |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for task in "JS":
    for name, base in MODELS:
        p = f"{R}/polymarket_rollout/{tag('pm', base, task)}/per_step.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p); cl = d[(d["mode"] == "closed") & d.ok]; op = d[(d["mode"] == "open") & d.ok]; dr = d[d["mode"] == "direct24"]
        mc = cl.groupby("step").ae.mean(); cc = cl.groupby("step").cover.mean(); mo = op.groupby("step").ae.mean()
        out.append(f"| {task} | {name} | {mc[1]:.4f} / {mc[24]:.4f} | {op[op.step > 1].ae.mean():.4f} | {mc[24] / mo[24]:.1f} | {cl[cl.step == 24].crps.mean():.4f} | "
                   f"{cc[1]:.2f} → {cc[24]:.2f} | {op[op.step > 1].cover.mean():.2f} | {dr.ae.mean():.4f} | {dr.ae.mean() / mc[24]:.2f} | {dr.cover.mean():.2f} |")
open("out/build/L1_rollout_summary.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
