# Shared settings and helpers for the run scripts (source it; run everything from the repository root).
#   PY      Python of the CPU environment (data pipeline, scoring, API calls)            [default: python]
#   PY_GPU  Python of the GPU environment (vLLM, LLaMA-Factory)                          [default: python]
#   GPU_MEM vLLM memory share per card                                                   [default: 0.8]
#   LUNA / SOL  deployment names of the two prompted models behind EW_API_BASE (see README)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-python}; PY_GPU=${PY_GPU:-python}; GPU_MEM=${GPU_MEM:-0.8}; CONC=${CONC:-16}
LUNA=${LUNA:-gpt-6-luna}; SOL=${SOL:-gpt-6-sol}
E=data/processed/diplomacy/eval_text; P=data/processed/polymarket/eval; EP=data/processed/diplomacy_press/eval_text
L=out/eval/local; A=out/eval/api; B=out/eval/baselines
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

_out() { local prev=""; for a in "$@"; do [ "$prev" = "--out" ] && { echo "$a"; return; }; prev=$a; done; }

# vllm <model> <infer_vllm.py args...>: local inference (greedy); skipped if the output exists. Writes <out>.part first, renames on success.
vllm() { local m=$1; shift; local out; out=$(_out "$@"); [ -s "$out" ] && return 0; mkdir -p "$(dirname "$out")"
  local args=(); local prev=""; for a in "$@"; do if [ "$prev" = "--out" ]; then args+=("$out.part"); else args+=("$a"); fi; prev=$a; done
  $PY_GPU scripts/infer_vllm.py --model "$m" --gpu_mem "$GPU_MEM" --max_model_len 16384 "${args[@]}" && mv "$out.part" "$out"; }

# api <deployment> <infer_api.py args...>: prompted inference through the OpenAI-compatible endpoint; skipped if the output exists.
api() { local m=$1; shift; local out; out=$(_out "$@"); [ -s "$out" ] && return 0; mkdir -p "$(dirname "$out")"
  $PY scripts/infer_api.py --model "$m" --concurrency "$CONC" "$@"; }

# score <task J|S|SF> <pred.jsonl>: Diplomacy scorer, writes <pred>/ (metrics.json, per_sample.parquet).
score() { $PY scripts/eval_dip.py --task "$1" --pred "$2" --out "${2%.jsonl}" > /dev/null; }
# score_cf <factual pred.jsonl> <counterfactual pred.jsonl> <out dir>
score_cf() { $PY scripts/eval_dip.py --task CF --pred_fact "$1" --pred_cf "$2" --out "$3" > /dev/null; }

# eval_finetuned <checkpoint> <tag> <dip|pm>: the standard zero-shot evaluation of a fine-tuned world model.
#   dip: Task J / S at K = 0 and 3 on the seven-player test set, counterfactuals (Task S form, K = 0), two-player variant (K = 0).
#   pm : Task J / S (1 h, binned output) at K = 0 and 3 on test_id, K = 0 on test_ood.
eval_finetuned() { local ck=$1 tag=$2 dom=$3 o=$L/$2; mkdir -p "$o"
  if [ "$dom" = dip ]; then
    for K in 0 3; do for T in J S; do
      vllm "$ck" --input $E/common_7p_${T}_K$K.jsonl --out $o/dip_${T}_K${K}_zs.jsonl --max_tokens 700; score $T $o/dip_${T}_K${K}_zs.jsonl; done; done
    vllm "$ck" --input $E/common_7p_CF_S_K0.jsonl --out $o/dip_CF_S_K0_zs.jsonl --max_tokens 700
    score_cf $o/dip_S_K0_zs.jsonl $o/dip_CF_S_K0_zs.jsonl $o/dip_CF_K0_zs
    for T in J S; do vllm "$ck" --input $E/common_2p_${T}_K0.jsonl --out $o/dip2p_${T}_K0_zs.jsonl --max_tokens 700; score $T $o/dip2p_${T}_K0_zs.jsonl; done
  else
    for K in 0 3; do for T in J S; do
      vllm "$ck" --pm_binned --input $P/eval_test_id_K$K.jsonl --where delta=1h --where task=$T --out $o/pm_test_id_${T}_K${K}_zs.jsonl --max_tokens 60; done; done
    for T in J S; do vllm "$ck" --pm_binned --input $P/eval_test_ood_K0.jsonl --where delta=1h --where task=$T --out $o/pm_test_ood_${T}_K0_zs.jsonl --max_tokens 60; done
  fi; }

# eval_sf <checkpoint|api:deployment|qwen> <out dir> <dip|pm> [binned]: factored Task S (the other actors first) on the shared subsets, K = 0 (and K = 3 in Diplomacy).
eval_sf() { local m=$1 o=$2 dom=$3 binned=${4:-}; mkdir -p "$o"
  if [ "$dom" = dip ]; then
    for K in ${SF_KS:-0}; do
      if [[ $m == api:* ]]; then api "${m#api:}" --input $E/sub1k_7p_SF_K$K.jsonl --out $o/dip_SF_K${K}_zs.jsonl --max_tokens 8000
      else vllm "$m" --input $E/sub1k_7p_SF_K$K.jsonl --out $o/dip_SF_K${K}_zs.jsonl --max_tokens 1500; fi
      score SF $o/dip_SF_K${K}_zs.jsonl; done
  elif [[ $m == api:* ]]; then api "${m#api:}" --input $P/sub1k_test_id_SF_K0.jsonl --out $o/pm_SF_K0_zs.jsonl --max_tokens 8000
  elif [ "$binned" = binned ]; then vllm "$m" --pm_binned --input $P/sub1k_test_id_SF_K0.jsonl --out $o/pm_SF_K0_zs.jsonl --max_tokens 120
  else vllm "$m" --input $P/sub1k_test_id_SF_K0.jsonl --out $o/pm_SF_K0_zs.jsonl --max_tokens 700; fi; }

# interventions <checkpoint|api:deployment> <out dir> [binned]: Polymarket actions never taken (Appendix), test_id anchors.
interventions() { local m=$1 o=$2 binned=${3:-}
  if [[ $m == api:* ]]; then api "${m#api:}" --input $P/interventions_K0.jsonl --where split=test_id --out $o/pm_interventions_K0.jsonl --max_tokens 8000
  elif [ "$binned" = binned ]; then vllm "$m" --pm_binned --input $P/interventions_K0.jsonl --where split=test_id --out $o/pm_interventions_K0.jsonl --max_tokens 60
  else vllm "$m" --input $P/interventions_K0.jsonl --where split=test_id --out $o/pm_interventions_K0.jsonl --max_tokens 700; fi; }

# horizons <checkpoint|api:deployment> <out dir> [binned]: Polymarket at 5 min and 24 h on the shared points (1 h comes from the main runs).
horizons() { local m=$1 o=$2 binned=${3:-}
  for H in 5m 24h; do for T in J S; do
    if [[ $m == api:* ]]; then api "${m#api:}" --input $P/sub1k_test_id_K0_h5m24h.jsonl --where delta=$H --where task=$T --out $o/pm_${T}_K0_${H}_zs.jsonl --max_tokens 8000
    else vllm "$m" --pm_binned --input $P/sub1k_test_id_K0_h5m24h.jsonl --where delta=$H --where task=$T --out $o/pm_${T}_K0_${H}_zs.jsonl --max_tokens 60; fi; done; done; }
