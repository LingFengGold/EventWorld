#!/bin/bash
# Q1 / Q2 (Sections 6.1, 6.2) and the related appendix experiments: single-step prediction with every actor's action (Task J) or only the
# agent's (Task S), baselines, scaling, history length, factored Task S, horizons and idle windows, robustness, from-scratch control,
# negotiation messages, outcome knowledge. Prompted models are evaluated on the shared subsets; fine-tuned models on the full test sets.
# Needs: the processed data, the checkpoints of run/train.sh, one GPU (CUDA_VISIBLE_DEVICES), EW_API_BASE / EW_API_KEY for GPT-6-Luna / Sol,
# and the DipNet service for the DipNet baseline (third_party/dipnet/README.md).
# Usage: bash run/q1_q2.sh [all | one or more of: baselines prompted finetuned factored history horizons press leakage]
source "$(dirname "$0")/common.sh"
CK=out/sft

baselines() {   # Diplomacy literal / frequency / DipNet baselines (CPU), Polymarket no-change and LightGBM
  $PY scripts/baseline_dip.py
  for pop in 7p 2p; do score J $B/dip_${pop}_J_state_unchanged.jsonl; score J $B/dip_${pop}_J_all_moves_succeed.jsonl
    for b in most_frequent_order uniform_legal own_literal; do score S $B/dip_${pop}_S_$b.jsonl; done; done
  $PY scripts/baseline_dipnet.py && score S $B/dip_7p_S_dipnet_sl.jsonl
  $PY scripts/baseline_pm.py; }

prompted_run() {   # <local model|api:deployment> <out dir>: Task J / S, K in {0, 3}, zero-shot and 3-shot, on the shared subsets
  local m=$1 o=$2; mkdir -p "$o"
  for K in 0 3; do for SH in 0 3; do sfx=$([ $SH = 0 ] && echo zs || echo 3s)
    for T in J S; do
      if [[ $m == api:* ]]; then
        api "${m#api:}" --input $E/sub1k_7p_${T}_K$K.jsonl --out $o/dip_${T}_K${K}_${sfx}.jsonl --max_tokens 8000 --shots $SH --shots_from $E/demos_dip_${T}_K$K.jsonl
        api "${m#api:}" --input $P/sub1k_test_id_K$K.jsonl --where task=$T --out $o/pm_${T}_K${K}_${sfx}.jsonl --max_tokens 2000 --shots $SH --shots_from $P/demos_${T}_K$K.jsonl
      else
        vllm "$m" --input $E/sub1k_7p_${T}_K$K.jsonl --out $o/dip_${T}_K${K}_${sfx}.jsonl --max_tokens 700 --shots $SH --shots_from $E/demos_dip_${T}_K$K.jsonl
        vllm "$m" --input $P/sub1k_test_id_K$K.jsonl --where task=$T --out $o/pm_${T}_K${K}_${sfx}.jsonl --max_tokens 300 --shots $SH --shots_from $P/demos_${T}_K$K.jsonl
      fi
      score $T $o/dip_${T}_K${K}_${sfx}.jsonl; done; done; done; }

prompted() { prompted_run Qwen/Qwen3-8B $L/qwen3-8b; prompted_run api:$LUNA $A/gpt-6-luna; prompted_run api:$SOL $A/gpt-6-sol_sub1k; }

finetuned() {   # standard evaluation of every fine-tuned world model (Table 2, scaling, robustness, from-scratch control)
  eval_finetuned $CK/qwen3-1.7b-dip20k-v4 sft1.7b-20k-v4 dip;  eval_finetuned $CK/qwen3-1.7b-pm20k sft1.7b-pm20k pm
  eval_finetuned $CK/qwen3-8b-dip20k qwen3-8b-dip20k dip;      eval_finetuned $CK/qwen3-8b-pm20k qwen3-8b-pm20k pm
  eval_finetuned $CK/qwen3-8b-dip320k qwen3-8b-dip320k dip;    eval_finetuned $CK/qwen3-8b-pm320k qwen3-8b-pm320k pm
  for N in 5 80 320; do eval_finetuned $CK/qwen3-1.7b-dip${N}k sft1.7b-dip${N}k dip; eval_finetuned $CK/qwen3-1.7b-pm${N}k sft1.7b-pm${N}k pm; done
  for S in 0.6b 4b; do eval_finetuned $CK/qwen3-$S-dip20k sft$S-dip20k dip; eval_finetuned $CK/qwen3-$S-pm20k sft$S-pm20k pm; done
  for N in 20 320; do eval_finetuned $CK/qwen3-1.7b-dip${N}k-scratch sft1.7b-dip${N}k-scratch dip; eval_finetuned $CK/qwen3-1.7b-pm${N}k-scratch sft1.7b-pm${N}k-scratch pm; done
  for S in 43 44; do o=$L/qwen3-1.7b-dip20k-v4-seed$S   # training seeds (Appendix, Table seeds): Task J / S on the shared subset
    for T in J S; do vllm $CK/qwen3-1.7b-dip20k-v4-seed$S --input $E/sub1k_7p_${T}_K0.jsonl --out $o/dip_${T}_K0_zs.jsonl --max_tokens 700; score $T $o/dip_${T}_K0_zs.jsonl; done; done; }

factored() {   # factored Task S: predict the other actors first (Appendix, Tables l2-gap-full, seeds, opp)
  eval_sf Qwen/Qwen3-8B $L/qwen3-8b dip; eval_sf Qwen/Qwen3-8B $L/qwen3-8b pm
  for M in "api:$LUNA $A/gpt-6-luna" "api:$SOL $A/gpt-6-sol_sub1k"; do set -- $M; eval_sf $1 $2 dip; eval_sf $1 $2 pm; done
  SF_KS="0 3"; eval_sf $CK/qwen3-1.7b-dip20k-sf $L/sft1.7b-dip20k-sf dip
  for S in 43 44; do eval_sf $CK/qwen3-1.7b-dip20k-sf-seed$S $L/qwen3-1.7b-dip20k-sf-seed$S dip; done
  SF_KS="0"; eval_sf $CK/qwen3-8b-dip20k-sf $L/qwen3-8b-dip20k-sf dip
  eval_sf $CK/qwen3-1.7b-pm20k-sf $L/sft1.7b-pm20k-sf pm binned; }

history() {   # history length K in {0, 1, 3, 6, 10} (Figure 5, Appendix)
  for K in 1 6 10; do for M in "$LUNA $A/gpt-6-luna" "$SOL $A/gpt-6-sol_sub1k"; do set -- $M; for T in J S; do
    api $1 --input $E/sub1k_7p_${T}_K$K.jsonl --out $2/dip_${T}_K${K}_zs.jsonl --max_tokens 8000; score $T $2/dip_${T}_K${K}_zs.jsonl
    api $1 --input $P/sub1k_test_id_K$K.jsonl --where task=$T --out $2/pm_${T}_K${K}_zs.jsonl --max_tokens 8000; done; done; done
  for K in 0 1 3 6 10; do for T in J S; do
    o=$L/sft1.7b-dip20k-ksweep; vllm $CK/qwen3-1.7b-dip20k-ksweep --input $E/sub1k_7p_${T}_K$K.jsonl --out $o/dip_${T}_K${K}_zs.jsonl --max_tokens 700; score $T $o/dip_${T}_K${K}_zs.jsonl
    o=$L/sft1.7b-pm20k-ksweep; vllm $CK/qwen3-1.7b-pm20k-ksweep --pm_binned --input $P/sub1k_test_id_K$K.jsonl --where task=$T --out $o/pm_test_id_${T}_K${K}_zs.jsonl --max_tokens 60
  done; done; }

horizons_all() {   # Polymarket horizons 5 min / 24 h (Figure 4a); the idle-window comparison (Figure 4b) reuses the K = 0 runs
  horizons $CK/qwen3-1.7b-pm20k $L/sft1.7b-pm20k binned; horizons api:$LUNA $A/gpt-6-luna; horizons api:$SOL $A/gpt-6-sol_sub1k; }

press() {   # negotiation messages (Appendix, Table press): Task S with and without the agent's messages, and a blanked-messages control
  run_press() { local m=$1 o=$2 mt=$3; mkdir -p $o
    for f in J_K0 S_K0 SM_K0 SM_K0_nomsg SF_K0 SFM_K0 SFM_K0_nomsg; do local x=$([[ $f == SF* ]] && echo 1500 || echo 700)
      if [[ $m == api:* ]]; then api "${m#api:}" --input $EP/press_test_$f.jsonl --out $o/press_${f}_zs.jsonl --max_tokens 8000
      else vllm "$m" --input $EP/press_test_$f.jsonl --out $o/press_${f}_zs.jsonl --max_tokens $x; fi; done
    ev() { $PY scripts/press_eval.py "$@" > /dev/null; }
    ev --task J --pred $o/press_J_K0_zs.jsonl --out $o/press_J_K0_zs; ev --task S --pred $o/press_S_K0_zs.jsonl --out $o/press_S_K0_zs
    ev --task SM --pred $o/press_SM_K0_zs.jsonl --out $o/press_SM_K0_zs --paired_with $o/press_S_K0_zs
    ev --task SM --pred $o/press_SM_K0_nomsg_zs.jsonl --out $o/press_SM_K0_nomsg_zs --paired_with $o/press_SM_K0_zs
    ev --task SF --pred $o/press_SF_K0_zs.jsonl --out $o/press_SF_K0_zs; ev --task SFM --pred $o/press_SFM_K0_zs.jsonl --out $o/press_SFM_K0_zs --paired_with $o/press_SF_K0_zs
    ev --task SFM --pred $o/press_SFM_K0_nomsg_zs.jsonl --out $o/press_SFM_K0_nomsg_zs --paired_with $o/press_SFM_K0_zs; }
  run_press api:$LUNA $A/gpt-6-luna_press; run_press api:$SOL $A/gpt-6-sol_press
  # fine-tuned: one checkpoint per condition, trained on the press games (plain: J + S; msg: J + SM; sf: J + SF; sfmsg: J + SFM)
  local o=$L/sft1.7b-press20k
  for C in plain msg sf sfmsg; do mkdir -p $o-$C; done
  vllm $CK/qwen3-1.7b-press20k-plain --input $EP/press_test_S_K0.jsonl --out $o-plain/press_S_K0_zs.jsonl --max_tokens 700
  vllm $CK/qwen3-1.7b-press20k-plain --input $EP/press_test_J_K0.jsonl --out $o-plain/press_J_K0_zs.jsonl --max_tokens 700
  for f in SM_K0 SM_K0_nomsg; do vllm $CK/qwen3-1.7b-press20k-msg --input $EP/press_test_$f.jsonl --out $o-msg/press_${f}_zs.jsonl --max_tokens 700; done
  vllm $CK/qwen3-1.7b-press20k-sf --input $EP/press_test_SF_K0.jsonl --out $o-sf/press_SF_K0_zs.jsonl --max_tokens 1500
  for f in SFM_K0 SFM_K0_nomsg; do vllm $CK/qwen3-1.7b-press20k-sfmsg --input $EP/press_test_$f.jsonl --out $o-sfmsg/press_${f}_zs.jsonl --max_tokens 1500; done
  ev() { $PY scripts/press_eval.py "$@"; }
  ev --task S --pred $o-plain/press_S_K0_zs.jsonl --out $o-plain/press_S_K0_zs; ev --task J --pred $o-plain/press_J_K0_zs.jsonl --out $o-plain/press_J_K0_zs
  ev --task SM --pred $o-msg/press_SM_K0_zs.jsonl --out $o-msg/press_SM_K0_zs --paired_with $o-plain/press_S_K0_zs
  ev --task SM --pred $o-msg/press_SM_K0_nomsg_zs.jsonl --out $o-msg/press_SM_K0_nomsg_zs --paired_with $o-msg/press_SM_K0_zs
  ev --task SF --pred $o-sf/press_SF_K0_zs.jsonl --out $o-sf/press_SF_K0_zs
  ev --task SFM --pred $o-sfmsg/press_SFM_K0_zs.jsonl --out $o-sfmsg/press_SFM_K0_zs --paired_with $o-sf/press_SF_K0_zs
  ev --task SFM --pred $o-sfmsg/press_SFM_K0_nomsg_zs.jsonl --out $o-sfmsg/press_SFM_K0_nomsg_zs --paired_with $o-sfmsg/press_SFM_K0_zs; }

leakage() {   # outcome knowledge of the prompted models (Appendix, Table leak): later markets, and a direct probe of the outcomes
  for M in "$LUNA $A/gpt-6-luna" "$SOL $A/gpt-6-sol_sub1k"; do set -- $M; for T in J S; do
    api $1 --input $P/sub1k_test_ood_K0.jsonl --where task=$T --out $2/pm_sub1k_test_ood_${T}_K0_zs.jsonl --max_tokens 2000; done
    $PY scripts/leakage_recall_probe.py run $1 $CONC; done
  $PY scripts/leakage_recall_probe.py score $LUNA $SOL; }

for stage in "${@:-all}"; do
  case $stage in
  all) baselines; prompted; finetuned; factored; history; horizons_all; press; leakage ;;
  horizons) horizons_all ;;
  *) "$stage" ;;
esac
done
