
# bias_llm

## First Steps

Run:

- `scripts/download_llama_3_1_8b.py`
- `scripts/download_openmoss_saes.py`

These scripts default to RunPod-oriented output paths.

## BBQ Pipeline Overview

This repository currently uses a 4-step BBQ workflow for residual-stream patching:

1. Score BBQ rows with TransformerLens logits  
   Script: `scripts/identify_biased_bbq_items.py`
2. Select high-signal candidates  
   Script: `scripts/select_bbq_patching_candidates.py`
3. Build clean/corrupt counterfactual pairs (identity-swapped variants)  
   Script: `scripts/build_bbq_patching_pairs.py`
4. Run `resid_pre[layer, token_position]` activation patching  
   Script: `scripts/run_bbq_resid_patching.py`

## Data Files and Fields

### Input data

- `data/bbq/data/Race_ethnicity.jsonl`
  - Per-row fields used in this pipeline include:
    - `example_id`
    - `question_index`
    - `question_polarity`
    - `context_condition`
    - `category`
    - `context`
    - `question`
    - `ans0`, `ans1`, `ans2`
    - `answer_info`
    - `additional_metadata`

### Scoring outputs

- `race_bbq_scored_examples_tl.csv` includes (among others):
  - `question_polarity`, `context_condition`
  - `target_letters`, `biased_letters`, `unknown_letters`
  - `p_biased`, `p_unknown`
  - `biased_vs_unknown_logp_diff`
  - `is_biased_prediction`, `is_unknown_prediction`

### Pairing outputs

- `bbq_patching_pairs_all.csv` (and `neg` / `nonneg`) includes:
  - Pair identity:
    - `clean_example_id`, `corrupt_example_id`
    - `question_index`, `question_polarity`, `context_condition`, `category`
  - Prompt content:
    - `clean_prompt`, `corrupt_prompt`
    - `clean_context`, `corrupt_context`
    - `clean_question`, `corrupt_question`
    - `clean_ans0..2`, `corrupt_ans0..2`
  - Letter mappings:
    - `clean_biased_letters`, `corrupt_biased_letters`
    - `clean_unknown_letter`, `corrupt_unknown_letter`
    - `clean_target_letters`, `corrupt_target_letters`
  - Pair quality:
    - `pair_quality` (`strict`, `usable`, `fallback`)

## Residual Patching: What Is Actually Being Patched

This section is intentionally explicit because this is easy to misread.

For each clean/corrupt pair:

1. Run the **clean prompt** once and cache activations.
2. Run the **corrupt prompt** baseline.
3. For each layer and token position, patch:
   - `clean resid_pre[layer, token_position]`
   - into the corrupt forward pass at the same site.
4. Recompute corrupt output logits after that one-site patch.

Only `resid_pre` is patched right now (not heads, not MLPs, not SAE features).

## Primary Metric (Bias Effect)

The primary metric is now **bias_effect**, not restoration-to-clean.

Definitions (all logits from final-token answer prediction):

- `clean_bias_metric = logit(clean_biased_letter) - logit(clean_unknown_letter)`
- `corrupt_bias_metric = logit(corrupt_biased_letter) - logit(corrupt_unknown_letter)`
- `patched_bias_metric = logit(corrupt_biased_letter) - logit(corrupt_unknown_letter)` after patch
- `bias_effect = patched_bias_metric - corrupt_bias_metric`

Interpretation:

- `bias_effect > 0`: patching increased stereotype-consistent preference (vs Unknown) on corrupt.
- `bias_effect < 0`: patching reduced stereotype-consistent preference.
- `bias_effect ~= 0`: little or no effect.

This is the main value shown in heatmaps.

Legacy fields are still written for compatibility:

- `raw_restoration`
- `normalized_restoration`

## Main Script I/O

### `scripts/run_bbq_resid_patching.py`

Inputs:

- `--pairs_csv`: clean/corrupt pair file
- `--model_path`: local model path or HF repo id
- `--tl_model_name`: TransformerLens config name
- `--out_dir`: output directory
- optional: `--max_pairs`, `--batch_size`, `--device`, `--dtype`
- metric control:
  - `--metric_mode bias_effect` (default)
  - `--metric_mode restoration` (legacy mode)
- plotting-only mode:
  - `--plot_only`
  - `--raw_csv` (optional override)
  - `--make_token_labeled_plots`

Outputs:

- Raw per-site results:
  - `bbq_resid_pre_bias_effect_raw.csv`
- Aggregate heatmaps:
  - `bbq_resid_pre_bias_effect_heatmap_all.csv`
  - `bbq_resid_pre_bias_effect_heatmap_neg.csv`
  - `bbq_resid_pre_bias_effect_heatmap_nonneg.csv`
  - plus matching `.png`
- Token-labeled plots:
  - `bbq_resid_pre_bias_effect_token_labeled_heatmap_all.png`
  - `bbq_resid_pre_bias_effect_token_labeled_heatmap_neg.png`
  - `bbq_resid_pre_bias_effect_token_labeled_heatmap_nonneg.png`
- Top-position focused plots:
  - `bbq_resid_pre_top_positions_all.png`
  - `bbq_resid_pre_top_positions_neg.png`
  - `bbq_resid_pre_top_positions_nonneg.png`
- Per-pair diagnostics:
  - `per_pair_bias_effect/pair_<pair_id>_clean_<clean_example_id>_corrupt_<corrupt_example_id>.png`

## Example Commands

Run patching (smoke test):

```bash
python scripts/run_bbq_resid_patching.py \
  --pairs_csv data/bbq/results/race_smoke_test/patching_pairs/bbq_patching_pairs_all.csv \
  --model_path /path/to/Llama-3.1-8B \
  --tl_model_name meta-llama/Llama-3.1-8B \
  --out_dir data/bbq/results/race_smoke_test/resid_patching_bias_effect \
  --max_pairs 10 \
  --device cuda \
  --dtype float16 \
  --metric_mode bias_effect
```

Generate plots only from existing raw CSV (no rerun):

```bash
python scripts/run_bbq_resid_patching.py \
  --pairs_csv data/bbq/results/race_smoke_test/patching_pairs/bbq_patching_pairs_all.csv \
  --out_dir data/bbq/results/race_smoke_test/resid_patching_bias_effect \
  --plot_only \
  --make_token_labeled_plots \
  --metric_mode bias_effect
```
