
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
3. Build **token-aligned minimal-swap** clean/corrupt counterfactual pairs  
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
  - Identity metadata:
    - `biased_identity`, `biased_identity_group`
    - `other_identity`, `other_identity_group`
    - `swap_identities` (e.g. `Hispanic<->African`)
  - Alignment diagnostics:
    - `n_diff_tokens` (number of token positions that differ — all identity tokens)
    - `prompt_token_len`
  - Pair quality:
    - `pair_type` (`minimal_swap`), `pair_quality` (`strict`)

## How Pairs Are Built (token-aligned minimal swap)

The corrupt prompt is **synthesized from the same source example** as the clean
prompt by swapping the two named identity surface forms in place (in both the
context and the answer options), keeping the unknown option, the answer order, and
every other token identical. The clean and corrupt prompts are therefore
token-for-token aligned everywhere except at the identity token positions.

This replaces an earlier approach that paired two *different* BBQ examples with
deliberately swapped answer positions. That approach destroyed token alignment
(empirically 0 / 1032 pairs were fully aligned; divergence began exactly at the
identity tokens), which made the per-token-position heatmap axis uninterpretable.

**Alignment gate.** A swap preserves token alignment only when the two surface
forms tokenize to the same number of tokens *as they appear in context* (with a
leading space). The builder requires `len(tok(" " + a)) == len(tok(" " + b))`, then
re-tokenizes both full prompts and verifies (a) equal length and (b) every
differing position decodes to an identity token. Pairs failing either check are
dropped and counted in the summary (`dropped_surface_token_len_mismatch`, etc.).
On the race candidate set this yields ~25% of candidates (315 pairs), with every
surviving pair verified to differ only at identity tokens.

## Residual Patching: What Is Actually Being Patched

This section is intentionally explicit because this is easy to misread.

For each clean/corrupt pair:

1. Run the **clean prompt** once and cache activations.
2. Run the **corrupt prompt** baseline.
3. For each layer and token position, patch:
   - `clean resid_pre[layer, token_position]`
   - into the corrupt forward pass at the same site.
4. Recompute corrupt output logits after that one-site patch.

Because the pairs are token-aligned, position `i` in the clean run corresponds to
position `i` in the corrupt run, so a single-site patch is well defined and the
token-position heatmap axis is interpretable *within* a pair.

**Cross-pair aggregation.** Different pairs still have different prompt lengths and
content, so averaging `bias_effect` by *raw* token position across pairs is only
valid for the fixed instruction prefix and the final answer token. For a
cross-pair-valid view, the run script also aggregates by **semantic span**
(`instruction`, `context`, `question`, `option_A/B/C`, `answer`) and by whether each
position is a swapped **identity token** — these align across all pairs and
templates, so the `question_index`-grouping workaround is unnecessary. Each patched
row in the raw CSV carries `span` and `is_identity_token` columns.

Only `resid_pre` is patched right now (not heads, not MLPs, not SAE features).

## Primary Metric (Bias Effect)

The primary metric is now **bias_effect**, not restoration-to-clean.

Definitions (all logits from final-token answer prediction). The minimal-swap
construction reads clean and corrupt at the **same fixed letters**: `biased_letter`
is the slot the stereotyped identity occupies in the *clean* prompt, and
`unknown_letter` is the unknown slot (unchanged by the swap):

- `clean_bias_metric = logit(biased_letter) - logit(unknown_letter)` on clean
- `corrupt_bias_metric = logit(biased_letter) - logit(unknown_letter)` on corrupt
- `patched_bias_metric = logit(biased_letter) - logit(unknown_letter)` after patch
- `bias_effect = patched_bias_metric - corrupt_bias_metric`

In the clean prompt the stereotyped identity sits in `biased_letter`; the swap moves
it out of that slot, so the corrupt baseline is de-biased
(`corrupt_bias_metric < clean_bias_metric`). The patch tests where the
"prefer the stereotyped slot" signal is stored.

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
- Aggregate heatmaps (by raw token position):
  - `bbq_resid_pre_bias_effect_heatmap_all.csv`
  - `bbq_resid_pre_bias_effect_heatmap_neg.csv`
  - `bbq_resid_pre_bias_effect_heatmap_nonneg.csv`
  - plus matching `.png`
  - NOTE: raw token position only aligns across pairs for the fixed instruction
    prefix and the final answer token; for content positions, use the span
    aggregation below.
- Span-aggregated heatmaps (cross-pair valid):
  - `bbq_resid_pre_bias_effect_span_heatmap_all.csv` (+ `neg` / `nonneg`)
  - plus matching `.png` — `layer × {instruction, context, question, option_A,
    option_B, option_C, answer}`
  - `bbq_resid_pre_bias_effect_by_span_identity.csv` — same, split by
    `is_identity_token` (1 = a swapped identity token, 0 = a shared token), the
    most direct bias-localization readout
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

Build token-aligned minimal-swap pairs:

```bash
python scripts/build_bbq_patching_pairs.py \
  --candidates_csv data/bbq/results/race_smoke_test/patching_candidates/race_bbq_patching_candidates_all.csv \
  --out_dir data/bbq/results/race_smoke_test/patching_pairs \
  --bbq_jsonl data/bbq/data/Race_ethnicity.jsonl \
  --model_path meta-llama/Llama-3.1-8B
```

The `--model_path` tokenizer must match the patching model so the alignment gate is exact.

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
