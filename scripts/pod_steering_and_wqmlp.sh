#!/usr/bin/env bash
# Bundled GPU job: the two open threads from the pipeline review / steering plan.
#   1. WinoQueer MLP-neuron attribution RE-RUN with the fixed pooled-ratio code
#      (Σnum/Σdenom). The headline cross-dataset MLP Jaccard currently compares a
#      PRE-fix WQ ranking against a POST-fix combined ranking — not apples-to-apples
#      until this re-runs. Output overwrites the WQ MLP attribution dir.
#   2. WinoQueer steering sweeps with --save_vectors, one per construct-ALIGNED axis
#      (sexual_orientation, gender_identity), persisting per-layer bias vectors (.pt).
#   3. OOD steering transfer: inject each saved vector into the BBQ MCQ forward and
#      measure Δbias. Matched (SO vec -> SO BBQ, gender_id vec -> Gender BBQ) AND
#      cross-construct specificity (each vec -> the other axis's BBQ file).
#
# Run:  cd ~/bias_llm && git pull && nohup bash scripts/pod_steering_and_wqmlp.sh > steering_wqmlp.log 2>&1 &
#       tail -f steering_wqmlp.log
set -euo pipefail

M=meta-llama/Llama-3.1-8B
WQ=data/winoqueer/results/segmented/cohort.csv          # full WQ cohort (has `axis` col)
OUT=pod_results/steering_transfer
VEC=$OUT/vectors
mkdir -p "$OUT" "$VEC" wq_mlp_refix

# --- Step 0: split the WQ cohort by construct-aligned axis (umbrella LGBTQ/Queer dropped) ---
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/winoqueer/results/segmented/cohort.csv")
for ax in ("sexual_orientation", "gender_identity"):
    sub = df[df.axis == ax]
    p = f"pod_results/steering_transfer/wq_{ax}.csv"
    sub.to_csv(p, index=False)
    print(f"{ax}: {len(sub)} pairs -> {p}")
PY

# ============================================================================
# 1) WinoQueer MLP neuron attribution — RE-RUN with fixed pooled ratio
# ============================================================================
echo "=== [1/3] WinoQueer MLP attribution (POST-fix pooled ratio) ==="
python -u scripts/run_winoqueer_mlp_neuron_attribution.py \
  --pairs_csv "$WQ" --out_dir ./wq_mlp_refix --no_resort --cohort_csv "$WQ" \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16 \
  --verify_topk 96 --verify_pairs 150

# ============================================================================
# 2) Steering sweeps with --save_vectors (per aligned axis)
# ============================================================================
for AX in sexual_orientation gender_identity; do
  echo "=== [2/3] Steering sweep: $AX ==="
  python -u scripts/run_winoqueer_steering_sweep.py \
    --pairs_csv "$OUT/wq_${AX}.csv" --out_dir "$OUT/sweep_${AX}" \
    --save_vectors "$VEC/wq_${AX}.pt" \
    --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16 \
    --max_pairs 600 --vector_position readout --n_random_seeds 5
done

# ============================================================================
# 3) OOD steering transfer onto BBQ MCQ — all layers, ambiguous context.
# Run table: vec_axis | bbq_file | subcategory | kind. The trans/binary split (via
# stereotyped_groups) keeps the gender_identity-aligned trans cell separate from the
# cis-binary role cell, so the matched test isn't diluted AND the binary cell is there
# for the within-gender construct comparison.
# ============================================================================
SO=data/bbq/data/Sexual_orientation.jsonl
GEN=data/bbq/data/Gender_identity.jsonl
RUNS=(
  "sexual_orientation|$SO|all|MATCHED"            # SO vector  -> SO QA
  "gender_identity|$GEN|trans|MATCHED"            # trans vec  -> trans QA (the key causal test)
  "gender_identity|$GEN|binary|WITHIN-GENDER"     # trans vec  -> cis-binary QA (the comparison)
  "sexual_orientation|$GEN|trans|CROSS"           # SO vec     -> trans QA (specificity)
  "gender_identity|$SO|all|CROSS"                 # trans vec  -> SO QA   (specificity)
)
for R in "${RUNS[@]}"; do
  IFS='|' read -r VEC_AX BBQ_FILE SUB KIND <<< "$R"
  TAG="vec-${VEC_AX}__$(basename "$BBQ_FILE" .jsonl)-${SUB}"
  echo "=== [3/3] Transfer ($KIND): $TAG ==="
  python -u scripts/run_bbq_steering_transfer.py \
    --vectors "$VEC/wq_${VEC_AX}.pt" --bbq_path "$BBQ_FILE" \
    --out_dir "$OUT/transfer_${TAG}" \
    --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16 \
    --context_condition ambig --subcategory "$SUB" --positions last --n_random_seeds 5
done

echo "=== DONE: wq_mlp_refix + steering vectors + 5 transfer runs ==="
