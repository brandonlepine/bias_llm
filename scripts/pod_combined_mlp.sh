#!/usr/bin/env bash
# Re-run ONLY the MLP neuron attribution WITH the per-group fix (build_group_map now reads Group_x/
# axis/block). head_patching + head_ablation already completed and are fine — don't re-run them.
# Run:  cd ~/bias_llm && git pull && nohup bash scripts/pod_combined_mlp.sh > combined_mlp.log 2>&1 &
set -e
COH=data/combined/results/_all/segmented/cohort.csv
M=meta-llama/Llama-3.1-8B
mkdir -p combined_mlp

python -u scripts/run_winoqueer_mlp_neuron_attribution.py \
  --pairs_csv "$COH" --out_dir ./combined_mlp --no_resort --cohort_csv "$COH" \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16 \
  --verify_topk 96 --verify_pairs 150

echo "=== DONE: combined_mlp (per-group accumulation ON) ==="
