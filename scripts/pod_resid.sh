#!/usr/bin/env bash
# Resid-stream patching ("where is the bias injected"): combined BBQ+CrowS, then WinoQueer segmented.
# Run AFTER the head/ablation/MLP runs finish (shares the one GPU):
#   cd ~/bias_llm && git pull && nohup bash scripts/pod_resid.sh > resid_run.log 2>&1 &
set -e
M=meta-llama/Llama-3.1-8B
mkdir -p combined_resid wq_seg_resid

echo "=== resid patching: combined BBQ+CrowS ==="
python -u scripts/run_winoqueer_resid_patching.py \
  --pairs_csv data/combined/results/_all/segmented/cohort.csv \
  --out_dir ./combined_resid --no_resort \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16

echo "=== resid patching: WinoQueer segmented cohort ==="
python -u scripts/run_winoqueer_resid_patching.py \
  --pairs_csv data/winoqueer/results/segmented/cohort.csv \
  --out_dir ./wq_seg_resid --no_resort \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16

echo "=== DONE: combined_resid + wq_seg_resid ==="
