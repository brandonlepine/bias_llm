#!/usr/bin/env bash
# Build identity-only direction vectors (v_identity) from the identity-mention dataset.
# One forward pass over the prompts -> per-identity centroids (all axes) + contrast vectors for the
# WinoQueer-aligned axes (sexual_orientation, gender_identity), read PRIMARILY at the identity-token
# span (with last-token + mean-pool variants saved alongside for position-matched residualization).
#
# Run:  cd ~/bias_llm && git pull && nohup bash scripts/pod_identity_vectors.sh > identity_vectors.log 2>&1 &
#       tail -f identity_vectors.log
set -euo pipefail

M=meta-llama/Llama-3.1-8B
OUT=pod_results/identity_vectors

echo "=== v_identity: centroids (all axes) + contrasts (SO, gender_identity) ==="
python -u scripts/build_identity_vectors.py \
  --identity_csv data/mi_identity_prompts.csv --out_dir "$OUT" \
  --axes all --per_identity \
  --model_path "$M" --tl_model_name "$M" --device cuda --dtype float16 --batch_size 16

echo "=== DONE: $OUT (v_identity_*.pt + identity_centroids.pt) ==="
echo "Next (CPU, after v_bias .pt lands from the steering job):"
echo "  python scripts/residualize_vectors.py --bias <v_bias.pt> \\"
echo "    --identity $OUT/v_identity_sexual_orientation.pt --out $OUT/v_stereotype_so.pt --mode project"
