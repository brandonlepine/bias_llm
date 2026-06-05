# Mechanistic-Interpretability Pipeline — Run Order & Figure Guide

**Model:** Llama-3.1-8B (base) via TransformerLens. SAEs: OpenMoss `Llama3_1-8B-Base-L24R-32x`.

**Research arc:** the circuit analyses were developed on **WinoQueer** (LGBTQ+ bias).
We are now replicating them on **out-of-distribution** datasets — **BBQ** and
**CrowS-Pairs** — to test whether the same heads / MLP neurons / layers carry bias
across social axes and across datasets. The cross-dataset comparison scripts
(Stage 7) are where that validation conclusion is actually drawn.

There are **two distinct mechanistic probes** in this repo, used on different prompt
formats. Keep them separate when reading results:

| Probe | Prompt format | Metric | Primary scripts |
|---|---|---|---|
| **Continuation scoring** | `prefix + shared predicate continuation`; minimal identity swap in the prefix | `bias_score = logP(cont \| queer/target prefix) − logP(cont \| control/dominant prefix)` | WinoQueer + CrowS + BBQ-"winoqueer-style" |
| **MCQ residual patching** | full BBQ multiple-choice prompt | `bias_effect = logit(biased_letter) − logit(unknown_letter)`, patched − corrupt | BBQ-MCQ only |

The head/MLP **circuit** analyses (patching, ablation, attribution, steering) all run
on the **continuation** probe and share one metric family: a per-pair
`bias_effect` (sufficiency: inject the stereotyped state) normalized to
`normalized_restoration = bias_effect / (queer_avg − control_avg)` so axes/datasets
with different baseline gaps are comparable.

---

## Stage 0 — Setup (run once)

| # | Script | Produces | Plots |
|---|---|---|---|
| 0a | `download_llama_3_1_8b.py` | local model under `models/Llama-3.1-8B` | none |
| 0b | `download_openmoss_saes.py` | SAEs under `saes/` | none |

---

## Stage 1 — WinoQueer data prep (continuation probe)

Order: 1 → 2 → 3 → (human Prodigy labeling) → 4 → 5 → 6. `winoqueer_identity_taxonomy.py`
is an imported library (taxonomy + stats helpers + plot styling); it runs nothing itself.

| # | Script | Consumes | Produces | Plots |
|---|---|---|---|---|
| 1 | `parse_winoqueer_predicates.py` | raw WinoQueer CSV | `--out_csv` (prefix/predicate/continuation split) + `--summary_csv` | none |
| 2 | `score_winoqueer_bias.py` | parsed CSV | `winoqueer_scored.csv`, `winoqueer_most_positive_bias.csv`, `winoqueer_most_negative_bias.csv`, `winoqueer_bias_summary.csv` | **`bias_score_histogram.png`** (distribution of `bias_score`, mean & zero lines); **`bias_score_by_identity.png`** (mean `bias_score` per `Gender_ID_x`, bar, red>0/blue<0) |
| 3 | `prepare_winoqueer_predicate_labeling.py` | `winoqueer_scored.csv` | `--out_csv` + `--out_jsonl` (one row per unique predicate for Prodigy) | none |
| — | *(human labels predicates in Prodigy → JSONL)* | | | |
| 4 | `audit_winoqueer_predicate_labels.py` | scored CSV + Prodigy JSONL | `winoqueer_scored_with_predicate_labels.csv`, `winoqueer_predicate_label_audit.csv`, `winoqueer_predicate_label_summary.csv` | none |
| 5 | `select_winoqueer_patching_candidates.py` | scored-with-labels CSV | `winoqueer_patching_candidates_all.csv` + 2 summary CSVs | none |
| 6 | `build_winoqueer_segmented_cohort.py` | candidates CSV | `cohort.csv`, `cohort_coverage.csv` | none |

`bias_score` sign convention (consistent across all stages): **positive = the stereotype
favors the marked/queer identity.**

---

## Stage 2 — WinoQueer mechanistic probes (the circuit)

All consume a **pairs/cohort CSV** (`sent_x`, `sent_y`, `prefix_y`, `predicate`,
`predicate_label_provisional`, `Gender_ID_x/y`, `bias_score`, `row_id`). The shared
metric helpers (`align_pair`, `continuation_logp`, the bias metric) live in
`run_winoqueer_resid_patching.py` and are imported by the others.

| Script | What it does | Key outputs | Plots (axes) |
|---|---|---|---|
| `run_winoqueer_resid_patching.py` | inject queer `resid_pre` into control run per (layer, position); localize bias in depth | `resid_pre_patching_raw.csv`, `resid_pre_span_heatmap.{csv,png}`, `resid_pre_identity_span_by_layer.csv` | **span heatmap** (y=layer, x=span {shared_pre/identity/shared_post/continuation}, color=mean `bias_effect`); **per-label span heatmaps**; **per-pair** heatmaps (y=layer, x=every control token, identity tokens boxed) |
| `run_winoqueer_head_patching.py` | head **sufficiency**: patch each head's `hook_z` (queer→control); also readout→identity attention | `head_patching_raw.csv`, `head_patching_layer_head.csv`, `head_attention_layer_head.csv`, `head_circuit_ranking.csv` | **head patching heatmap** (layer×head, `bias_effect`); **attention heatmap** (layer×head, readout→identity attn); **scatter** (x=attn, y=bias_effect, color=layer); **top-circuit bar**; **per-layer summary** (twin axis) |
| `run_winoqueer_head_ablation.py` | head **necessity**: resample-ablate `hook_z` (control→queer); cumulative knockout of top-k | `head_ablation_raw.csv`, `head_knockout_raw.csv`, `head_ablation_ranking.csv`, `head_knockout_curve.{csv,png}` | **necessity heatmap** (layer×head, top-10 boxed); **volcano** (x=mean ablation effect, y=sign-consistency, color=layer); **cumulative knockout curve** (x=#heads, y=frac bias remaining, ±SE) |
| `run_winoqueer_mlp_neuron_attribution.py` | first-order **AtP** over all MLP neurons (sufficiency + necessity), then exact-patch top-K to validate | `mlp_neuron_attribution.csv`, `..._top.csv`, per-group CSVs, `mlp_verified_top.csv` | **layer profile** (per-layer attribution mass); **top-neuron barh** (suff & nec); **suff-vs-nec hexbin**; **layer×neuron signed fingerprint** (2 panels); **AtP-vs-exact scatter** (Pearson/Spearman/sign-agreement) |
| `run_winoqueer_steering_sweep.py` | difference-of-means steering vector per layer; sweep layer×alpha on held-out set (induce on control, de-bias on queer); KL fluency guard | `steering_sweep_raw.csv`, layer×alpha CSVs | **layer×alpha heatmaps** (control & queer, color=bias fraction 0..1); **best-layer curves** (real vs random ±SE); **frontier scatter** (x=KL symlog, y=bias fraction); **steering-vs-patching layer** (twin axis) |
| `run_winoqueer_greedy_knockout.py` | greedy: repeatedly add the head that most reduces *remaining* bias | `greedy_knockout_curve.{csv,png}`, optional `greedy_selection_frequency.{csv,png}` | **greedy curve** (frac remaining vs #heads, overlaid with marginal curve); **selection-frequency barh** (bootstrap) |
| `winoqueer_head_sufficiency_necessity.py` | join head sufficiency (patching) × necessity (ablation), `core_score = z(bias)+z(ablation)` | `head_sufficiency_necessity.{csv,png}` | **scatter** x=sufficiency, y=necessity, size=ablation consistency, color=layer, top-K labeled |

---

## Stage 3 — WinoQueer segmented analysis (decompose by identity / predicate)

These re-aggregate the Stage-2 raw CSVs by social axis / identity / predicate. They are
the **WinoQueer-specific (taxonomy-hardcoded)** ancestors of the generalized Stage-7
scripts; the generalized versions (Stage 7) are what now feed the cross-dataset
comparison, so several of these are effectively legacy for the OOD work.

| Script | Consumes | Plots |
|---|---|---|
| `winoqueer_segmented_head_analysis.py` | `head_patching_raw.csv`, `head_ablation_raw.csv`, WinoQueer cohort | `segmented_heads_groups_matrix.png` (heads×identities WRITE + within-head share); `segmented_head_selectivity_scatter.png`; `segmented_head_jaccard_{write,read}.png`; `segmented_read_vs_write.png`; `segmented_top_heads_per_identity.png`; `segmented_umbrella_union_test.csv` |
| `winoqueer_segmented_mlp_analysis.py` | per-group `mlp_neuron_attribution__*.csv` + `mlp_layer_profile_by_group.csv` | `segmented_mlp_layer_profiles_by_axis.png`; `segmented_mlp_layer_profiles.png`; `segmented_mlp_selectivity_vs_magnitude.png`; `segmented_mlp_neuron_jaccard.png` |
| `winoqueer_segmented_predicate_analysis.py` | `segmented_head_stats__idpred.csv` + `segmented_pooled_head_ranking.csv` | `segmented_predicate_factor_tuning.png` (η²(identity) vs η²(predicate)); `segmented_predicate_circuit_jaccard.png`; `segmented_within_predicate_sharing.png` |
| `winoqueer_segmented_greedy_compare.py` | per-group greedy curve CSVs (+ optional freq) | `segmented_greedy_curves.png`; `segmented_greedy_selected_jaccard.png`; `segmented_greedy_umbrella_union.csv`; `segmented_greedy_freq__<label>.png` |

---

## Stage 4 — BBQ data prep (TWO sub-pipelines)

### 4A. BBQ MCQ residual-patching (documented in README)

Order: 4A.1 → 4A.2 → 4A.3 → 4A.4.

| # | Script | Consumes | Produces | Plots |
|---|---|---|---|---|
| 4A.1 | `identify_biased_bbq_items.py` | BBQ JSONL (`--bbq_path`) | `<prefix>_scored_examples_tl.csv`, `_most_biased_ranked_tl.csv`, `_summary_by_polarity_tl.csv`, `_top_biased_{neg,nonneg}_tl.csv` | none |
| 4A.2 | `select_bbq_patching_candidates.py` | `*_scored_examples_tl.csv` | `*_patching_candidates_{all,neg,nonneg,summary}.csv` | none |
| 4A.3 | `build_bbq_patching_pairs.py` | candidates CSV + BBQ JSONL | `bbq_patching_pairs_{all,neg,nonneg,summary}.csv` (token-aligned minimal-swap pairs) | none |
| 4A.4 | `run_bbq_resid_patching.py` | `bbq_patching_pairs_all.csv` | `bbq_resid_pre_bias_effect_raw.csv` + aggregates below | see below |

`run_bbq_resid_patching.py` plots:
- **span heatmap** `_span_heatmap_{all,neg,nonneg}.png` — y=layer, x=span
  {instruction/context/question/option_A/B/C/answer}, color=mean `bias_effect`.
- **span × identity** `_span_identity_{all,neg,nonneg}.png` — two panels (swapped-identity
  tokens vs shared scaffold), layer×span. *The most direct localization view.*
- **per-pair** `per_pair_bias_effect/pair_*.png` — y=layer, x=token (dead leading zeros
  trimmed); target identity green, distractor orange, identity xticklabels crimson.
- diagnostics `_token_position_label_diagnostics.csv`.

`bias_effect = patched_metric − corrupt_metric`, where
`metric = logit(biased_letter) − logit(unknown_letter)` at the final answer token.
`bias_effect > 0` ⇒ patching the clean (stereotyped) state increased stereotype
preference. (Verified against the code; sign matches the README.)

### 4B. BBQ continuation-scoring (WinoQueer-style, for cross-dataset circuit work)

Order: 4B.1 → 4B.2 → `build_bbq_winoqueer_pairs.py` → 4B.3 → 4B.4 → 4B.5. The Prodigy
review side-loop (`prepare_bbq_review.py` → human → `apply_bbq_review.py`) is optional.
`bbq_group_taxonomy.py` is the shared imported taxonomy module.

| # | Script | Consumes | Produces |
|---|---|---|---|
| 4B.1 | `parse_bbq_stereotypes.py` | `data/bbq/templates/new_templates - *.csv` | `bbq_stereotypes_raw.csv` |
| 4B.2 | `build_bbq_curated_predicates.py` | `bbq_stereotypes_raw.csv` | `bbq_predicates_curated.csv` |
| — | `build_bbq_winoqueer_pairs.py` | raw stereotypes + curated predicates | `bbq_pairs_all.csv` |
| 4B.3 | `score_bbq_bias.py` | `bbq_pairs_all.csv` | `bbq_pairs_scored.csv` + per-(category,frame) candidate CSVs |
| 4B.4 | `finalize_bbq_candidates.py` | `bbq_pairs_scored.csv` | `bbq_candidates_final.csv`, `bbq_scored_summary.csv`, per-category CSVs |
| 4B.5 | `build_bbq_segmented_cohort.py` | `bbq_candidates_final.csv` | per-axis `cohort.csv`, `cohort_coverage.csv` |

None of Stage 4B produces plots (4B.3 prints group means; review scripts render HTML cards).

---

## Stage 5 — CrowS-Pairs prep (continuation probe)

| Script | Consumes | Produces | Plots |
|---|---|---|---|
| `parse_crows_pairs.py` | `data/crows-pairs/crows-pairs.csv` | `crows_pairs_prompts.csv` (main, **stereo only**) + `crows_pairs_prompts_antistereo.csv` (separate) | none |

Then the main (stereo) file is scored with the same continuation scorer →
`pod_results/scoring/crows/crows_pairs_scored.csv`. **Antistereo** items go to a separate file
(`--antistereo_out`) and are never in the main analysis set — they're kept (tagged, sign-flipped:
`Group_x` is from `sent_more` = the anti-stereotypical sentence) for a later stereo-vs-antistereo
comparison.

---

## Stage 6 — Combine BBQ + CrowS candidates

| Script | Consumes | Produces | Plots |
|---|---|---|---|
| `build_combined_candidates.py` | `bbq_candidates_final.csv` + `crows_pairs_scored.csv` | `data/combined/bbq_crows_candidates.csv` (source-tagged, fresh global `row_id`) | none (prints axis×source pivot) |

This combined cohort is the input for one-shot patching on the pod, whose raw outputs
(`head_patching_raw.csv`, `head_ablation_raw.csv`, MLP attribution) land under
`pod_results/bbq_crows_combined/`.

---

## Stage 7 — Cross-dataset / segmented analysis (the validation figures)

These are **dataset-agnostic** generalizations of Stage 3. They produce most of the
publication figures and the actual "do these datasets share a circuit?" answer.

| Script | Consumes | Key figures (axes) |
|---|---|---|
| `segmented_circuit_analysis.py` (core) | `head_patching_raw.csv` (+ optional `head_ablation_raw.csv`) + cohort | `cross_axis_head_jaccard.png` (axis×axis Jaccard of top-k WRITE heads); `fig_read_vs_write_by_axis.png`; `fig_axis_layer_write_profile.png` (layer profile per axis, mean **positive** WRITE); `fig_source_agreement_by_axis.png` (BBQ vs CrowS: mean WRITE bars + top-head Jaccard); `fig_selectivity_by_axis.png`; `fig_heads_by_identity_per_axis.png`. CSVs: `head_stats__{axis,block,identity}.csv`, `pooled_head_ranking.csv`, `identity_selectivity_within_axis.csv`, `read_vs_write_within_axis.csv`, `identity_overlap_within_axis.csv`, `cross_axis_head_jaccard.csv`, `source_agreement_by_axis.csv` |
| `segmented_mlp_circuit_analysis.py` | per-group MLP attribution CSVs + cohort | `mlp_axis_layer_profile.png` (2 panels: SUFFICIENCY/NECESSITY mass by layer); `mlp_cross_axis_neuron_jaccard.png`; `mlp_selectivity_by_axis.png` |
| `segmented_resid_analysis.py` | resid patching raw + cohort | `resid_identity_layer_by_axis.png` (+ per-source) — layer curve of `normalized_restoration` per axis; `resid_span_heatmaps_by_axis.png` (per-axis layer×span small multiples); `resid_peak_layer_by_axis.csv` |
| `compare_segmented_runs.py` | ≥2 analysis output dirs (WinoQueer vs BBQ+CrowS) | `compare_read_vs_write_all_axes.png`; `compare_selectivity_all_axes.png`; **`compare_cross_dataset_head_jaccard.png`** (dataset×dataset Jaccard of top-k pooled WRITE heads — the headline overlap figure); `compare_axis_layer_profiles.png` |
| `head_sharing_analysis.py` (newest) | ≥2 (name, head_patching_raw, cohort) | **`D_identity_similarity_focus/all.png`** (identity×identity Pearson-correlation of per-head WRITE vectors, clustered — the centerpiece); `E_head_x_identity_focus.png`; `A_xdataset_head_scatter.png` (per-head WRITE dataset0 vs dataset1); `B_head_x_axis_dataset_heatmap.png`; `C_shared_axis_replication_scatter.png` |

**Metric note for Stage 7:** every script here uses `normalized_restoration` as the WRITE
metric (so axes/datasets with different baseline gaps are comparable). *(As of this
review, `head_sharing_analysis.py` was switched from raw `bias_effect` to
`normalized_restoration` to match — re-run it to regenerate panels A–E.)*

---

## Quick "which figure answers which question"

- *Where in depth does bias live?* → resid span heatmaps (Stage 2/4A), `resid_identity_layer_by_axis` (Stage 7).
- *Which heads write the stereotype?* → head patching heatmap + `pooled_head_ranking`.
- *Which heads are necessary?* → ablation heatmap, knockout/greedy curves.
- *Is it a two-stage read→write circuit?* → `fig_read_vs_write_by_axis`, head attention heatmap.
- *Do different social axes share heads?* → `cross_axis_head_jaccard`.
- *Do BBQ and CrowS agree (same dataset-internal axis)?* → `fig_source_agreement_by_axis`.
- *Do WinoQueer and BBQ/CrowS share a circuit (cross-dataset OOD validation)?* →
  `compare_cross_dataset_head_jaccard` + `head_sharing_analysis` panels A/C/D.
- *Which MLP neurons / layers?* → MLP attribution + `mlp_axis_layer_profile`, `mlp_cross_axis_neuron_jaccard`.
- *Can one direction control the bias?* → steering layer×alpha heatmaps + frontier.

---

*See `docs/REVIEW_FINDINGS.md` for the logic/math review and open questions.*
