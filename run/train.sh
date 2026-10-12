#!/bin/bash
# Fine-tune every world model of the paper with LLaMA-Factory (full fine-tuning, 3 epochs, effective batch 32; Appendix C.2).
# configs/sft.yaml expects 2 GPUs (CUDA_VISIBLE_DEVICES=0,1), configs/sft_8b.yaml 4 GPUs. A finished model (weights in out/sft/<name>) is skipped.
# Usage: bash run/train.sh [all|main|scaling|seeds|factored|history|scratch|press|8b]      (default: all)
source "$(dirname "$0")/common.sh"
LF=${LF:-llamafactory-cli}
DIP="dataset_dir=data/processed/diplomacy/sft"; PM="dataset_dir=data/processed/polymarket/sft"; PRESS="dataset_dir=data/processed/diplomacy_press/sft"

# train <config> <output name> <dataset> [extra key=value overrides]
train() { local cfg=$1 name=$2 ds=$3; shift 3; ls out/sft/$name/*.safetensors > /dev/null 2>&1 && return 0
  FORCE_TORCHRUN=1 $LF train $cfg dataset=$ds eval_dataset=${ds}_val output_dir=out/sft/$name "$@"; }

main()     { train configs/sft.yaml qwen3-1.7b-dip20k-v4 dip7p_20k $DIP
             train configs/sft.yaml qwen3-1.7b-pm20k pm_20k $PM; }
scaling()  { for N in 5 80 320; do train configs/sft.yaml qwen3-1.7b-dip${N}k dip7p_${N}k $DIP; train configs/sft.yaml qwen3-1.7b-pm${N}k pm_${N}k $PM; done
             train configs/sft.yaml qwen3-0.6b-dip20k dip7p_20k $DIP model_name_or_path=Qwen/Qwen3-0.6B
             train configs/sft.yaml qwen3-0.6b-pm20k pm_20k $PM model_name_or_path=Qwen/Qwen3-0.6B
             train configs/sft.yaml qwen3-4b-dip20k dip7p_20k $DIP model_name_or_path=Qwen/Qwen3-4B per_device_train_batch_size=2 gradient_accumulation_steps=8
             train configs/sft.yaml qwen3-4b-pm20k pm_20k $PM model_name_or_path=Qwen/Qwen3-4B per_device_train_batch_size=2 gradient_accumulation_steps=8; }
seeds()    { for S in 43 44; do train configs/sft.yaml qwen3-1.7b-dip20k-v4-seed$S dip7p_20k $DIP seed=$S
                                train configs/sft.yaml qwen3-1.7b-dip20k-sf-seed$S dip7p_20k_sf $DIP seed=$S; done; }
factored() { train configs/sft.yaml qwen3-1.7b-dip20k-sf dip7p_20k_sf $DIP
             train configs/sft.yaml qwen3-1.7b-pm20k-sf pm_20k_sf $PM; }
history()  { train configs/sft.yaml qwen3-1.7b-dip20k-ksweep dip7p_20k_ksweep $DIP cutoff_len=6144
             train configs/sft.yaml qwen3-1.7b-pm20k-ksweep pm_20k_ksweep $PM; }
scratch()  { for N in 20 320; do   # same architecture, random initialization; lr 1e-4 chosen by a probe on dip7p_20k_val (Appendix, From-Scratch Control)
               train configs/sft.yaml qwen3-1.7b-dip${N}k-scratch dip7p_${N}k $DIP train_from_scratch=true learning_rate=1.0e-4
               train configs/sft.yaml qwen3-1.7b-pm${N}k-scratch pm_${N}k $PM train_from_scratch=true learning_rate=1.0e-4; done; }
press()    { for C in plain msg sf sfmsg; do train configs/sft.yaml qwen3-1.7b-press20k-$C press_20k_$C $PRESS cutoff_len=2560; done; }
big()      { train configs/sft_8b.yaml qwen3-8b-dip20k dip7p_20k $DIP
             train configs/sft_8b.yaml qwen3-8b-pm20k pm_20k $PM
             train configs/sft_8b.yaml qwen3-8b-dip320k dip7p_320k $DIP
             train configs/sft_8b.yaml qwen3-8b-pm320k pm_320k $PM cutoff_len=3600
             train configs/sft_8b.yaml qwen3-8b-dip20k-sf dip7p_20k_sf $DIP; }

for stage in "${@:-all}"; do
  case $stage in
  all) main; scaling; seeds; factored; history; scratch; press; big ;;
  8b) big ;;
  *) "$stage" ;;
esac
done
