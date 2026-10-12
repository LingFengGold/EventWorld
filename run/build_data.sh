#!/bin/bash
# Build both benchmarks from the raw records (Section 4, Appendix A). The processed files are also released (see README), so this step is
# only needed to rebuild them. Raw inputs (see README): data/diplomacy/standard_no_press.jsonl and standard_press_with_msgs.jsonl
# (webDiplomacy games of Paquette et al., 2019), data/polymarket/trades.parquet and markets.parquet (Hugging Face SII-WANGZJ/Polymarket_data).
source "$(dirname "$0")/common.sh"
mkdir -p out/build

# ---- Diplomacy: replay every game in the engine, keep the exactly reproduced no-press games, split by game (test = last 4% of game ids)
$PY pipeline/dip_deep.py standard_no_press standard_press_with_msgs       # per-game statistics -> out/dip_deep/*.games.csv
$PY pipeline/dip_repro_all.py standard_no_press standard_press_with_msgs  # engine replay of every game -> out/dip_deep/repro_all.csv
$PY pipeline/dip_build_index.py              # games_index.parquet (reproduced games, populations, splits)
$PY pipeline/dip_extract_transitions.py      # transitions.parquet (one row per phase, engine check, interaction features)
$PY pipeline/dip_order_stats.py              # order_freq_train.parquet (most-frequent-order baseline)
$PY pipeline/dip_build_samples.py            # decision points: seven- and two-player test sets (Task J / S, K = 0 and 3), demonstrations,
                                             # rollout chains, SFT id lists
$PY pipeline/dip_build_counterfactuals.py    # actions never taken: counterfactual orders with engine references (Task S and J form)
for N in 5 20 80 320; do                     # SFT sets: half Task J, half Task S, K mixed over {0, 3}
  $PY pipeline/dip_build_sft.py --n_j $((N * 500)) --n_s $((N * 500)) --name dip7p_${N}k; done
$PY pipeline/dip_build_sft.py --sf --name dip7p_20k_sf          # factored Task S (the other powers' orders, then the board)
$PY pipeline/dip_build_sft.py --K -2 --name dip7p_20k_ksweep    # history length uniform over {0, 1, 3, 6, 10}
# press games (negotiation messages; Appendix, Table press)
for stage in index transitions eval sft; do $PY_GPU pipeline/press_build.py $stage; done

# ---- Polymarket: long-lived markets, taker orders merged per trader, decision points (large orders + random control times), targets
$PY pipeline/pm_extract_orders.py            # raw trades -> taker orders of the 4,176 markets traded >= 30 days with >= 5,000 trades
$PY pipeline/pm_build_store.py               # block-level order store per market (orders_by_market/, markets.parquet, wallets.parquet)
$PY pipeline/pm_build_points.py              # decision points: anchors (large orders, the trader is the agent) and control windows
$PY pipeline/pm_build_splits.py              # event-grouped splits: train / val / test_id (training period), test_ood (events from 2026)
$PY pipeline/pm_build_targets.py             # targets at 5 min, 1 h and 24 h
$PY pipeline/pm_response_curve.py            # train-only response curve (reference for the Polymarket interventions)
$PY pipeline/pm_build_eval.py                # evaluation sets (Task J / S, K = 0 and 3), interventions, rollout chains, demonstrations
$PY pipeline/pm_build_rollout_j.py           # Task J version of the rollout chains
for N in 5 20 80 320; do $PY pipeline/pm_build_sft.py --n $((N * 1000)) --name pm_${N}k; done
$PY pipeline/pm_build_sft.py --n 20000 --sf --name pm_20k_sf
$PY pipeline/pm_build_sft.py --n 20000 --ks 0,1,3,6,10 --name pm_20k_ksweep

# ---- shared subsets and derived evaluation sets
$PY pipeline/build_api_subset.py             # 1,000 Diplomacy transitions and 1,333 Polymarket points shared by all models
$PY pipeline/build_api_subset_ood.py         # 1,333 points of the later markets (outcome-knowledge check)
$PY pipeline/build_k_sweep_dip.py            # history K in {1, 6, 10}
$PY pipeline/build_k_sweep_pm.py
$PY pipeline/build_sfactored.py              # factored Task S evaluation sets
