#!/usr/bin/env bash
# Combined BBQ+CrowS patching — the remaining stages (head_patching already done).
# Run:  cd ~/bias_llm && git pull && nohup bash scripts/pod_combined_remaining.sh > combined_remaining.log 2>&1 &
set -e
COH=data/combined/results/_all/segmented/cohort.csv
M=meta-llama/Llama-3.1-8B
mkdir -p combined_head_ablation combined_mlp

echo "=== head ablation ==="
python -u scripts/run_winoqueer_head_ablation.py \
  --pairs_csv "$COH" --out_dir ./combined_head_ablation --no_resort \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16

echo "=== MLP neuron attribution ==="
python -u scripts/run_winoqueer_mlp_neuron_attribution.py \
  --pairs_csv "$COH" --out_dir ./combined_mlp --no_resort --cohort_csv "$COH" \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16 \
  --verify_topk 96 --verify_pairs 150

echo "=== DONE: head_ablation + mlp ==="
