
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
   Works across all single-identity BBQ categories (Race_ethnicity, Gender_identity,
   Physical_appearance, Disability_status, Religion, SES, Sexual_orientation) — target
   matching resolves 100% of ambiguous items in each. Pass `--bbq_path` for the split
   and `--out_prefix` to name outputs (defaults to `<file_stem>_bbq`, e.g.
   `gender_identity_bbq`). The race smoke-test artifacts use prefix `race_bbq`.
2. Select high-signal candidates  
   Script: `scripts/select_bbq_patching_candidates.py`  
   Output names follow `--out_prefix`, which defaults to the `--scored_csv` stem with
   `_scored_examples_tl` stripped — so it auto-matches the prefix from step 1 (e.g.
   `gender_identity_bbq`) and the steps chain without extra flags.
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

**Reference-distractor filter (`--distractor_mode reference`, default).** Each BBQ
instance pairs *two* identities. If both are marked/stereotyped groups (e.g.
Black↔Asian), `bias_effect` is a contrast between two stereotypes, not
stereotype-vs-neutral — confounded. So by default the builder keeps only pairs whose
**distractor is the dominant/unmarked reference group** for the category, defined by
an explicit per-category set `DOMINANT_REFERENCE_GROUPS` (race: White/Caucasian/
European; gender: man/male/cisgender; disability: nonDisabled; SES: highSES; physical:
nonObese/notPregnant/tall/…; religion: Christian/Protestant/Catholic; sexual
orientation: straight). Use `--distractor_mode all` to keep every aligned pair. Each
pair records `target_identity`/`target_identity_group`, `distractor_identity`/
`distractor_identity_group`, and `distractor_is_reference`. Clean-pair counts: race
108, gender 703, SES 522, disability 179, physical 173, religion 70, sexual_orientation 56.

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
- performance / robustness:
  - `--patch_batch_size N` (default 16): patches N token positions per forward pass
    (batched over positions within a layer) → ~N× fewer forwards on CUDA. Numerically
    equivalent to N=1 up to float batching nondeterminism.
  - **Resume by default**: the raw CSV is written per-pair and flushed, so re-running the
    same command continues where it left off (skips completed pairs, redoes the last
    possibly-partial one). Pass `--overwrite` to start fresh.
- metric control:
  - `--metric_mode bias_effect` (default)
  - `--metric_mode restoration` (legacy mode)
- plotting-only mode:
  - `--plot_only`
  - `--raw_csv` (optional override)
  - `--make_token_labeled_plots`

Outputs:

- Raw per-site results (source of truth, one row per pair×layer×position):
  - `bbq_resid_pre_bias_effect_raw.csv` — includes `span` and `is_identity_token`
- **Canonical cross-pair aggregate — by semantic span** (`layer × {instruction,
  context, question, option_A, option_B, option_C, answer}`):
  - `bbq_resid_pre_bias_effect_span_heatmap_{all,neg,nonneg}.{csv,png}`
- **Span × identity** (two panels: swapped-identity tokens vs shared scaffold tokens):
  - `bbq_resid_pre_bias_effect_span_identity_{all,neg,nonneg}.png`
  - `bbq_resid_pre_bias_effect_by_span_identity.csv`
  - This is the most direct localization view: where the swapped identity tokens
    inject the effect vs where it propagates into shared tokens.
- Per-pair token-labeled heatmaps (valid: positions align within a single pair):
  - `per_pair_bias_effect/pair_<pair_id>_clean_<id>_corrupt_<id>.png`
  - Dead leading columns (everything before the first swapped identity token, which
    is exactly 0 by construction) are trimmed; swapped identity tokens are red.
  - **Both** swapped identities are outlined so it's never ambiguous which tokens are
    which: the stereotyped **target** (green; from `target_letters`, polarity-independent)
    and the **distractor** it is swapped against (orange), each in the context and as
    its answer option, with the group names in the legend. The title shows the metric
    `readout` slots and the `target` slot (they coincide for `neg`, differ for `nonneg`).
- Diagnostics:
  - `bbq_resid_pre_bias_effect_token_position_label_diagnostics.csv`

Why there is **no** raw-token-position aggregate heatmap: averaging `bias_effect`
by integer position across pairs is invalid because pairs have different content at
each index (only the fixed instruction prefix and the final answer token align).
That aggregation mislabels and mixes tokens, so the span aggregates above replace it.

## Example Commands

Build token-aligned minimal-swap pairs:

```bash
python scripts/build_bbq_patching_pairs.py \
  --candidates_csv data/bbq/results/race_smoke_test/patching_candidates/race_bbq_patching_candidates_all.csv \
  --out_dir data/bbq/results/race_smoke_test/patching_pairs \
  --bbq_jsonl data/bbq/data/Race_ethnicity.jsonl \
  --model_path meta-llama/Llama-3.1-8B \
  --distractor_mode reference
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
