#!/usr/bin/env bash
# Resid-stream patching ("where in depth is the bias injected") for ALL THREE datasets SEPARATELY,
# so BBQ vs CrowS vs WinoQueer can be compared (the by-source split the pooled combined run can't give).
# resid raw is per-pair, so the old "combined BBQ+CrowS" == bbq_resid + crows_resid concatenated.
#
# Run on the pod, AFTER the head/ablation/MLP runs (shares the one GPU):
#   cd ~/bias_llm && git pull && nohup bash scripts/pod_resid_by_source.sh > resid_by_source.log 2>&1 &
set -e
M=meta-llama/Llama-3.1-8B
COMMON=(--no_resort --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16)
mkdir -p bbq_resid crows_resid

echo "=== [1/2] resid patching: BBQ only (3481 pairs) ==="
python -u scripts/run_winoqueer_resid_patching.py \
  --pairs_csv data/combined/results/bbq/segmented/cohort.csv \
  --out_dir ./bbq_resid "${COMMON[@]}"

echo "=== [2/2] resid patching: CrowS-Pairs only (579 pairs) ==="
python -u scripts/run_winoqueer_resid_patching.py \
  --pairs_csv data/combined/results/crows/segmented/cohort.csv \
  --out_dir ./crows_resid "${COMMON[@]}"

# WinoQueer: reusing the EXISTING pooled resid run (pod_results/winoqueer/pooled/resid) — not re-run
# here. If you later want it on the segmented cohort for a same-flavour by-axis comparison, add:
#   python -u scripts/run_winoqueer_resid_patching.py \
#     --pairs_csv data/winoqueer/results/segmented/cohort.csv --out_dir ./wq_seg_resid "${COMMON[@]}"

echo "=== DONE: bbq_resid + crows_resid ==="
