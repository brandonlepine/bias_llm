#!/usr/bin/env bash
# ===========================================================================================
# Unified, MODEL-PARAMETERIZED mech-interp pipeline runner.
#
# One entry point for the whole battery so a fresh pod with a DIFFERENT model just runs:
#     cd ~/bias_llm && git pull && nohup bash scripts/run_pipeline.sh --model <HF_ID> all \
#         > run_<tag>.log 2>&1 &
# All outputs are namespaced by model tag under pod_results/<tag>/, so multiple models coexist
# and never overwrite each other. The stimulus cohorts are held FIXED across models (same inputs)
# so cross-model comparisons are apples-to-apples.
#
# Run everything ......... bash scripts/run_pipeline.sh --model <id> all
# Run a dataset battery .. bash scripts/run_pipeline.sh --model <id> wq combined
# Run one probe .......... bash scripts/run_pipeline.sh --model <id> wq:mlp combined:resid
# Run a cross step ....... bash scripts/run_pipeline.sh --model <id> identity transfer residualize
# List steps ............. bash scripts/run_pipeline.sh list
#
# Flags (or env vars): --model MODEL  --tl TL_MODEL(=MODEL)  --tag TAG(=slug(MODEL))
#                      --device cuda  --dtype float16
# ===========================================================================================
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
TL_MODEL="${TL_MODEL:-}"
TAG="${TAG:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

ACTION=run
STEPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL="$2"; shift 2;;
    --tl)     TL_MODEL="$2"; shift 2;;
    --tag)    TAG="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --dtype)  DTYPE="$2"; shift 2;;
    -h|--help) ACTION=help; shift;;
    list)      ACTION=list; shift;;
    -*) echo "unknown flag: $1" >&2; exit 1;;
    *) STEPS+=("$1"); shift;;
  esac
done

TL_MODEL="${TL_MODEL:-$MODEL}"
[[ -z "$TAG" ]] && TAG="$(printf '%s' "$MODEL" | sed -E 's#.*/##; s/[^A-Za-z0-9._-]/-/g' | tr 'A-Z' 'a-z')"
ROOT="pod_results/$TAG"
COMMON=(--model_path "$MODEL" --tl_model_name "$TL_MODEL" --device "$DEVICE" --dtype "$DTYPE")

cohort_for() {
  case "$1" in
    wq)       echo data/winoqueer/results/segmented/cohort.csv;;
    combined) echo data/combined/results/_all/segmented/cohort.csv;;
    bbq)      echo data/combined/results/bbq/segmented/cohort.csv;;
    crows)    echo data/combined/results/crows/segmented/cohort.csv;;
    *) return 1;;
  esac
}

banner() { echo; echo "=================================================================="; \
           echo ">>> [$TAG] $*"; echo "=================================================================="; }

# ---------------------------------------------------------------------------- GPU probes
p_resid()    { local d=$1 c; c=$(cohort_for "$d"); banner "$d : resid patching"
  python -u scripts/run_winoqueer_resid_patching.py --pairs_csv "$c" \
    --out_dir "$ROOT/$d/resid" --no_resort "${COMMON[@]}"; }

p_head()     { local d=$1 c; c=$(cohort_for "$d"); banner "$d : head patching"
  python -u scripts/run_winoqueer_head_patching.py --pairs_csv "$c" \
    --out_dir "$ROOT/$d/head" --no_resort "${COMMON[@]}"; }

p_ablation() { local d=$1 c; c=$(cohort_for "$d"); banner "$d : head ablation"
  python -u scripts/run_winoqueer_head_ablation.py --pairs_csv "$c" \
    --out_dir "$ROOT/$d/ablation" --no_resort "${COMMON[@]}"; }

p_greedy()   { local d=$1 c A; c=$(cohort_for "$d"); A="$ROOT/$d/ablation"; banner "$d : greedy knockout"
  if [[ ! -f "$A/head_ablation_ranking.csv" ]]; then
    echo "SKIP greedy[$d]: missing $A/head_ablation_ranking.csv — run $d:ablation first." >&2; return 0; fi
  python -u scripts/run_winoqueer_greedy_knockout.py --pairs_csv "$c" --out_dir "$ROOT/$d/greedy" \
    --ablation_ranking_csv "$A/head_ablation_ranking.csv" \
    --marginal_curve_csv "$A/head_knockout_curve.csv" --no_resort "${COMMON[@]}"; }

p_mlp()      { local d=$1 c; c=$(cohort_for "$d"); banner "$d : MLP neuron attribution"
  python -u scripts/run_winoqueer_mlp_neuron_attribution.py --pairs_csv "$c" --out_dir "$ROOT/$d/mlp" \
    --cohort_csv "$c" --no_resort --verify_topk 96 --verify_pairs 150 "${COMMON[@]}"; }

p_steering() { local d=$1 c; c=$(cohort_for "$d"); banner "$d : steering sweep"
  python -u scripts/run_winoqueer_steering_sweep.py --pairs_csv "$c" --out_dir "$ROOT/$d/steering" \
    --save_vectors "$ROOT/$d/steering/vectors.pt" \
    --max_pairs 600 --vector_position readout --n_random_seeds 5 "${COMMON[@]}"; }

# ---------------------------------------------------------------------------- CPU analysis
p_seg()      { local d=$1 c src=(); c=$(cohort_for "$d"); banner "$d : segmented analysis (CPU)"
  [[ "$d" == combined ]] && src=(--source_col source)
  python -u scripts/segmented_circuit_analysis.py \
    --patching_raw "$ROOT/$d/head/head_patching_raw.csv" \
    --ablation_raw "$ROOT/$d/ablation/head_ablation_raw.csv" \
    --cohort "$c" --out_dir "$ROOT/$d/seg_head" --label "$d" "${src[@]}"
  python -u scripts/segmented_mlp_circuit_analysis.py \
    --in_dir "$ROOT/$d/mlp" --cohort "$c" --out_dir "$ROOT/$d/seg_mlp" --label "$d"
  python -u scripts/segmented_resid_analysis.py \
    --resid_raw "$ROOT/$d/resid/resid_pre_patching_raw.csv" \
    --cohort "$c" --out_dir "$ROOT/$d/seg_resid" --label "$d" "${src[@]}"; }

# A battery ALWAYS ends with the segmented analysis — the per-axis / per-identity (and, for the
# combined cohort, per-source) disentangling. The raw probes keep every pair (joined on row_id), so
# seg is what turns them into the by-group profiles we actually compare; pooling all identities into
# one undifferentiated run would defeat the comparison, so it is never the default.
battery() { local d=$1; p_resid "$d"; p_head "$d"; p_ablation "$d"; p_greedy "$d"; p_mlp "$d"; p_steering "$d"; p_seg "$d"; }

# ---------------------------------------------------------------------------- cross / OOD steps
s_identity() { banner "identity-only vectors (v_identity, all axes)"
  python -u scripts/build_identity_vectors.py --identity_csv data/mi_identity_prompts.csv \
    --out_dir "$ROOT/identity" --axes all --batch_size 16 "${COMMON[@]}"; }

s_transfer() { local T="$ROOT/transfer" V="$ROOT/transfer/vectors" WQ; WQ=$(cohort_for wq)
  banner "OOD steering transfer (WinoQueer vectors -> BBQ QA)"
  mkdir -p "$T" "$V"
  python - "$WQ" "$T" <<'PY'
import sys, pandas as pd
wq, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(wq)
for ax in ("sexual_orientation", "gender_identity"):
    sub = df[df.axis == ax]; sub.to_csv(f"{out}/wq_{ax}.csv", index=False)
    print(f"{ax}: {len(sub)} pairs")
PY
  for AX in sexual_orientation gender_identity; do
    python -u scripts/run_winoqueer_steering_sweep.py --pairs_csv "$T/wq_${AX}.csv" \
      --out_dir "$T/sweep_${AX}" --save_vectors "$V/wq_${AX}.pt" \
      --max_pairs 600 --vector_position readout --n_random_seeds 5 "${COMMON[@]}"
    python -u scripts/run_winoqueer_steering_sweep.py --pairs_csv "$T/wq_${AX}.csv" \
      --out_dir "$T/sweep_${AX}_identitypos" --save_vectors "$V/wq_${AX}_identitypos.pt" \
      --max_pairs 600 --vector_position identity --n_random_seeds 5 "${COMMON[@]}"
  done
  local SO=data/bbq/data/Sexual_orientation.jsonl GEN=data/bbq/data/Gender_identity.jsonl
  local RUNS=( "sexual_orientation|$SO|all" "gender_identity|$GEN|trans" "gender_identity|$GEN|binary" \
               "sexual_orientation|$GEN|trans" "gender_identity|$SO|all" )
  for R in "${RUNS[@]}"; do IFS='|' read -r va bf sub <<< "$R"
    python -u scripts/run_bbq_steering_transfer.py --vectors "$V/wq_${va}.pt" --bbq_path "$bf" \
      --out_dir "$T/transfer_vec-${va}__$(basename "$bf" .jsonl)-${sub}" \
      --context_condition ambig --subcategory "$sub" --positions last --n_random_seeds 5 "${COMMON[@]}"
  done; }

s_residualize() { banner "residualize: v_stereotype = v_bias - v_identity (CPU)"
  for AX in sexual_orientation gender_identity; do
    local B="$ROOT/transfer/vectors/wq_${AX}_identitypos.pt" I="$ROOT/identity/v_identity_${AX}.pt"
    if [[ ! -f "$B" || ! -f "$I" ]]; then
      echo "SKIP residualize[$AX]: need $B and $I (run transfer + identity first)." >&2; continue; fi
    python scripts/residualize_vectors.py --bias "$B" --identity "$I" \
      --out "$ROOT/identity/v_stereotype_${AX}.pt" --mode project --identity_variant identity
  done; }

s_compare() { banner "cross-dataset comparison (wq vs combined, CPU)"
  mkdir -p "$ROOT/compare"
  python -u scripts/compare_segmented_runs.py --out_dir "$ROOT/compare/head" \
    --run wq "$ROOT/wq/seg_head" --run combined "$ROOT/combined/seg_head"
  python -u scripts/compare_segmented_mlp_runs.py --out_dir "$ROOT/compare/mlp" \
    --run wq "$ROOT/wq/mlp" --run combined "$ROOT/combined/mlp"; }

# ---------------------------------------------------------------------------- dispatch
run_one() {
  local s=$1
  if [[ "$s" == *:* ]]; then
    local d=${s%%:*} probe=${s##*:}
    cohort_for "$d" >/dev/null || { echo "unknown dataset: $d" >&2; exit 1; }
    case "$probe" in
      resid) p_resid "$d";; head) p_head "$d";; ablation) p_ablation "$d";;
      greedy) p_greedy "$d";; mlp) p_mlp "$d";; steering) p_steering "$d";; seg) p_seg "$d";;
      *) echo "unknown probe: $probe (resid|head|ablation|greedy|mlp|steering|seg)" >&2; exit 1;;
    esac; return
  fi
  case "$s" in
    wq|combined|bbq|crows) battery "$s";;
    identity)    s_identity;;
    transfer)    s_transfer;;
    residualize) s_residualize;;
    compare)     s_compare;;
    all)
      # battery (incl. seg) on WinoQueer + the combined BBQ+CrowS cohort. The combined cohort keeps
      # its `source` column, so combined's seg splits by axis AND by source (bbq vs crows) — no need
      # to re-run the battery on the bbq/crows cohorts separately (they're available as datasets if
      # you want a fully isolated by-source battery: `... bbq crows`).
      battery wq; battery combined
      s_identity; s_transfer; s_residualize; s_compare;;
    *) echo "unknown step: $s (try: list)" >&2; exit 1;;
  esac
}

usage() {
  cat <<EOF
Unified mech-interp pipeline runner (model: $MODEL  tag: $TAG  out: $ROOT)

  bash scripts/run_pipeline.sh [--model ID --tl ID --tag T --device cuda --dtype float16] STEP...

DATASET BATTERIES (resid head ablation greedy mlp steering + seg):  wq  combined  bbq  crows
  every battery ENDS with segmented (by-axis/identity; combined also by-source) analysis by default
SINGLE PROBE:                                                  <dataset>:<probe>   e.g. wq:mlp
  probes: resid head ablation greedy mlp steering seg
CROSS / OOD STEPS:   identity  transfer  residualize  compare
ALL:                 all   (= battery wq + battery combined [each incl. seg] + identity + transfer
                            + residualize + compare; combined's seg disentangles bbq vs crows)

Outputs -> pod_results/<tag>/...   (per-model namespaced; cohorts held fixed across models)
EOF
}

case "$ACTION" in
  help|list) usage; exit 0;;
esac
if [[ ${#STEPS[@]} -eq 0 ]]; then usage; exit 0; fi

echo "model=$MODEL  tl=$TL_MODEL  tag=$TAG  device=$DEVICE  dtype=$DTYPE"
echo "steps: ${STEPS[*]}"
for s in "${STEPS[@]}"; do run_one "$s"; done
banner "PIPELINE COMPLETE — outputs in $ROOT"
