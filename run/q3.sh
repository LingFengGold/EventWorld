#!/bin/bash
# Q3 (Section 6.3): situations not in the event log. Long-horizon rollouts (Diplomacy five-step chains and consistency with the agent's plans,
# Polymarket 24-hour rollouts) and actions never taken (Diplomacy counterfactuals in Task S and Task J form; Polymarket interventions).
# Needs one GPU, EW_API_BASE / EW_API_KEY, and the DipNet service for the consistency runs (third_party/dipnet/README.md).
# Usage: bash run/q3.sh [all|rollouts|counterfactuals|interventions]
source "$(dirname "$0")/common.sh"
CK=out/sft
DIP_M=("sft1.7b $CK/qwen3-1.7b-dip20k-v4" "sft8b $CK/qwen3-8b-dip20k" "gpt-6-luna api:$LUNA" "gpt-6-sol api:$SOL")
PM_M=("sft1.7b $CK/qwen3-1.7b-pm20k" "sft8b $CK/qwen3-8b-pm20k" "gpt-6-luna api:$LUNA" "gpt-6-sol api:$SOL")

rollouts() {   # tags: Diplomacy <model> = Task J, <model>_S = Task S; Polymarket <model> = Task S, <model>_J = Task J
  $PY scripts/rollout_cr.py --real_only                                 # engine reference (Real) for the 500 starts
  $PY scripts/rollout_cr.py --real_only --agent_model rl
  for x in "${DIP_M[@]}"; do set -- $x
    $PY scripts/rollout_dip.py --model $2 --tag $1 --task J; $PY scripts/rollout_dip.py --model $2 --tag ${1}_S --task S
    $PY scripts/rollout_cr.py --model $2 --tag $1 --task J;  $PY scripts/rollout_cr.py --model $2 --tag ${1}_S --task S; done
  # consistency when the agent is the RL-trained DipNet (Appendix, Table cr-agent)
  $PY scripts/rollout_cr.py --model $CK/qwen3-1.7b-dip20k-v4 --tag sft1.7b_agentrl --task J --agent_model rl
  $PY scripts/rollout_cr.py --model $CK/qwen3-1.7b-dip20k-v4 --tag sft1.7b_S_agentrl --task S --agent_model rl
  for x in "${PM_M[@]}"; do set -- $x; local b=$([[ $2 == api:* ]] && echo "" || echo --binned)
    $PY scripts/rollout_pm.py --model $2 --tag $1 --task S $b; $PY scripts/rollout_pm.py --model $2 --tag ${1}_J --task J $b; done
  $PY scripts/rollout_pm_acc24.py
  $PY scripts/rollout_summary.py; }

counterfactuals() {
  # Task S form on the shared subset for the prompted models (factual = the model's Task S, K = 0 run of run/q1_q2.sh); the fine-tuned models
  # are covered by eval_finetuned (run/q1_q2.sh finetuned)
  vllm Qwen/Qwen3-8B --input $E/sub1k_7p_CF_S_K0.jsonl --out $L/qwen3-8b/dip_CF_S_K0_zs.jsonl --max_tokens 700
  score_cf $L/qwen3-8b/dip_S_K0_zs.jsonl $L/qwen3-8b/dip_CF_S_K0_zs.jsonl $L/qwen3-8b/dip_CF_K0_zs
  for M in "$LUNA $A/gpt-6-luna" "$SOL $A/gpt-6-sol_sub1k"; do set -- $M
    api $1 --input $E/sub1k_7p_CF_S_K0.jsonl --out $2/dip_CF_S_K0_zs.jsonl --max_tokens 8000; score_cf $2/dip_S_K0_zs.jsonl $2/dip_CF_S_K0_zs.jsonl $2/dip_CF_K0_zs
    # Task J form: the other players' recorded orders are given (Appendix, Table l3-taskj; Table 4)
    api $1 --input $E/sub1k_7p_CF_J_K0.jsonl --out $2/dip_CF_J_K0_zs.jsonl --max_tokens 8000; score_cf $2/dip_J_K0_zs.jsonl $2/dip_CF_J_K0_zs.jsonl $2/dip_CF_J_K0_zs; done
  local o=$L/sft1.7b-20k-v4
  vllm $CK/qwen3-1.7b-dip20k-v4 --input $E/sub1k_7p_CF_J_K0.jsonl --out $o/dip_CF_J_K0_zs.jsonl --max_tokens 700
  score_cf $o/dip_J_K0_zs.jsonl $o/dip_CF_J_K0_zs.jsonl $o/dip_CF_J_K0_zs
  o=$L/qwen3-8b-dip20k   # factual Task J predictions regenerated together with the counterfactuals
  vllm $CK/qwen3-8b-dip20k --input $E/sub1k_7p_CF_J_K0.jsonl --out $o/dip_CF_J_K0_zs.jsonl --max_tokens 700
  vllm $CK/qwen3-8b-dip20k --input $E/sub1k_7p_J_K0.jsonl --out $o/dip_J_K0_zs.regen_sub1k.jsonl --max_tokens 700
  score_cf $o/dip_J_K0_zs.regen_sub1k.jsonl $o/dip_CF_J_K0_zs.jsonl $o/dip_CF_J_K0_zs; }

interventions_all() {   # Polymarket: the agent's order removed, reversed or scaled (Appendix, Table l3-full), scored against the response curve
  interventions Qwen/Qwen3-8B $L/qwen3-8b; interventions api:$LUNA $A/gpt-6-luna; interventions api:$SOL $A/gpt-6-sol_sub1k
  interventions $CK/qwen3-1.7b-pm20k $L/sft1.7b-pm20k binned; interventions $CK/qwen3-8b-pm20k $L/qwen3-8b-pm20k binned; }

for stage in "${@:-all}"; do
  case $stage in
  all) rollouts; counterfactuals; interventions_all ;;
  interventions) interventions_all ;;
  *) "$stage" ;;
esac
done
