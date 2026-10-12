#!/bin/bash
# Q4 (Section 6.4): can event world models help an agent act? Diplomacy only (the engine executes the agent's actions), 200 test starts,
# 10 phases, opponents = DipNet (Paquette et al., 2019). Plans made inside the world model (Figure 6) come from the consistency runs of
# run/q3.sh. Needs the DipNet service (third_party/dipnet/README.md), one GPU, and the world models of run/train.sh.
# Usage: bash run/q4.sh [all|verifier|bc]
source "$(dirname "$0")/common.sh"
CK=out/sft; LF=${LF:-llamafactory-cli}

verifier() {   # pre-execution verification: the agent samples B candidate orders, the world model predicts each outcome, the best is played
  $PY scripts/agent_verifier_dip.py --scorer engine --tag engine                     # engine upper bound and the B = 1 baseline (CPU)
  $PY scripts/agent_verifier_dip.py --scorer $CK/qwen3-1.7b-dip20k-v4 --tag sft1.7b-20k --modes S,J
  $PY scripts/agent_verifier_dip.py --scorer $CK/qwen3-8b-dip20k --tag sft8b-20k --modes S,J
  $PY scripts/agent_verifier_dip.py --scorer $CK/qwen3-1.7b-dip320k --tag sft1.7b-320k --modes S,J
  # optimization pressure: predicted vs realized gain of the chosen candidate as B grows (Figure 7b)
  $PY scripts/agent_verifier_dip.py --scorer engine --tag engine_pressure --budgets 2,4,10,32
  $PY scripts/agent_verifier_dip.py --scorer $CK/qwen3-1.7b-dip320k --tag sft1.7b-320k_pressure --modes S,J --budgets 2,4,10,32; }

bc() {   # behavior cloning on real vs imagined games, and world-model initialization (Table 5)
  local S=scripts/agent_syndata_dip.py WM=$CK/qwen3-1.7b-dip320k
  $PY $S gen --source real --tag real                                                 # 2,000 train starts played in the engine
  $PY $S gen --source wm --model $WM --tag wm-sft1.7b-320k                            # the same starts played inside the 320k world model
  $PY $S build --real real --syn wm-sft1.7b-320k                                      # arms real1k, syn1k, mix1k1k (+ mix0.5k)
  $PY $S gen --source real --tag real2 --n 1300 --seed 2 --exclude real               # 1,300 new starts: Real 2k and Imagined 2k
  $PY $S build --real real --syn wm-sft1.7b-320k --real2 real2 --arms real2k
  $PY $S gen --source wm --model $WM --tag wm2-sft1.7b-320k --n 1300 --seed 2 --exclude real
  $PY $S build --real real --syn wm-sft1.7b-320k --syn2 wm2-sft1.7b-320k --arms syn2k
  # one GPU per policy: train (configs/sft_agent.yaml), then play the 200 test starts in the engine
  for arm in real1k syn1k mix1k1k real2k syn2k "real1k_wminit20k real1k $CK/qwen3-1.7b-dip20k-v4" "real1k_wminit320k real1k $WM"; do
    set -- $arm; local tag=$1 ds=${2:-$1} init=${3:-Qwen/Qwen3-1.7B}
    ls $CK/qwen3-1.7b-agent-$tag/*.safetensors > /dev/null 2>&1 || \
      $LF train configs/sft_agent.yaml dataset=agent_$ds output_dir=$CK/qwen3-1.7b-agent-$tag model_name_or_path=$init
    EW_GPU_MEM=0.3 $PY $S eval --policy $CK/qwen3-1.7b-agent-$tag --tag $tag; done; }

for stage in "${@:-all}"; do
  case $stage in
  all) verifier; bc ;;
  *) "$stage" ;;
esac
done
