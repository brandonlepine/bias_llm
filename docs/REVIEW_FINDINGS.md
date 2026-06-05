# Pipeline Review — Logic / Math Findings

Review of the mech-interp pipeline (WinoQueer → BBQ/CrowS OOD validation). Items are
grouped by how much they affect the **OOD-validation conclusion**. Each item notes
file:line, severity, and my confidence. **No code has been changed yet** — the items
marked ⚠️ NEEDS DECISION are research-judgment calls.

Legend: 🔴 affects a stated validation figure · 🟠 code bug · 🟡 robustness/interpretation
caveat · ⚪ minor/cosmetic · ✅ checked and correct.

---

## Fixes applied

### Round 1
| Item | File | Change |
|---|---|---|
| T1.1 | `head_sharing_analysis.py:41` | `WRITE` switched `bias_effect → normalized_restoration` to match every other Stage-7 script; colorbar/docstring labels updated. **Re-run to regenerate panels A–E.** |
| T2.1 | `finalize_bbq_candidates.py:53-65` | `--keep_resisted` now produces the resisted-only set (stereotypes with mean_bias<0, pairs with bias_score<0, sorted most-resisted first); dead `keep_mask` removed. Default path unchanged. |
| T2.2 | `run_bbq_resid_patching.py` | Resume re-keyed on a content-stable `pair_key` = `clean_example_id\|swap_identities\|question_polarity\|context_condition` (new `pair_key` column in the raw). Legacy raws without the column fall back to position-based resume **with a loud warning**. |

### Round 2
| Item | File | Change |
|---|---|---|
| T1.3 | `parse_crows_pairs.py` | Antistereo items now pass the same validity filters but are written to a **separate** file (`--antistereo_out`, default `crows_pairs_prompts_antistereo.csv`), never mixed into the main set; tagged + documented as sign-flipped for a later stereo-vs-antistereo comparison. Verified: 875 stereo / 122 antistereo split cleanly. |
| T3.1 | `run_winoqueer_steering_sweep.py` | Cross-pair aggregation of `bias_fraction` switched **mean → median** (heatmaps, best-layer pick, frontier, console); best-layer curve now median + IQR band. (KL stays a mean.) |
| T3.1 | `run_winoqueer_mlp_neuron_attribution.py` | Per-pair ratio averaging replaced with a **pooled ratio** (Σ numerator / Σ denom) in both `run_attribution` and `verify_topk`, so a tiny-denom pair can't blow up the mean. (True per-neuron median was infeasible: ~21 GB.) Sign-consistency counters unchanged; AtP-vs-exact scatter stays comparable. |

**Reverted / not changed (with reason):**
- **T1.2 — reverted.** BBQ is genuinely pre-filtered to `bias_score>0` upstream
  (`finalize_bbq_candidates.py`), so selection is *not* substantively asymmetric — just done
  at a different step. The change was a no-op nit; reverted to avoid churn.
- **T1.4 — accepted as-is.** Combined `row_id = range(len)` is positional, but you will never
  regenerate the combined file without re-running patching, so the mis-join risk can't trigger.
- **T2.3 — kept.** Redundant upfront baseline compute, but not a correctness bug and it serves
  as a pre-loop sanity check; left alone to avoid risk.
- **T3.2 — no change (resolved).** The accepted WinoQueer autoregressive metric
  (`winoqueer/code/metric_autoregressive.py:149-158`) sums log-prob over the **entire** shared
  continuation, so reading/intervening across the continuation in patching/steering is
  *aligned* with the community standard — it measures bias *propagation*, not a circular
  readout. A readout-only variant would be narrower than the accepted metric; it's an optional
  extra localization check, not a fix.

### Round 3
| Item | File | Change |
|---|---|---|
| T3.3 | `run_bbq_resid_patching.py` | Span aggregation changed from mean-over-all-(pair×position) to **per-pair SUM within span, then mean across pairs** (in both `aggregate_by_span` and the identity/scaffold split). A concentrated causal signal is no longer diluted by inert tokens or long contexts; identity+scaffold panels sum back to the main heatmap. Synthetic-tested (0.9 vs old diluted 0.45). |
| T3.4 | `segmented_mlp_circuit_analysis.py`, `winoqueer_segmented_mlp_analysis.py` | Interpretation comment added at the neuron-Jaccard null pool: it uses top-trimmed neurons (not the full neuron space), so the null is **conservative** (can understate sharing, never invents it). No logic change, per your call. |
| T3.5 | `compare_segmented_runs.py` | Added a shared `AXIS_ORDER`; bar charts and the layer-profile now order axes consistently across datasets, and the layer-profile colors by axis (line style = dataset) so the same axis is the same color in both runs. |
| T3.6 | `run_winoqueer_steering_sweep.py` | Random control now averages over **N=5** norm-matched random directions (`--n_random_seeds`) instead of one, so a single lucky/unlucky draw can't drive the control. New `rand_seed` column in the raw. |
| T3.7 | `run_winoqueer_steering_sweep.py` | `stratified_split` now **shuffles within each predicate group** (seeded) so train/test isn't split by bias_score rank. |

**T3.8 — no action (clarified).** It's in `identify_biased_bbq_items.py`, the BBQ **MCQ** scorer
(Pipeline A / residual-patching branch), affecting only the `nonneg`-polarity readout slot. It does
**not** touch the BBQ-WinoQueer-style continuation dataset (Pipeline B) that drives the cross-dataset
validation. Only revisit if you rely on the nonneg split of the BBQ MCQ span heatmaps.

**Review complete** — all Tier-1/2/3 items are resolved, reverted-with-reason, or documented.

---

## 🔴 Tier 1 — Directly affects the cross-dataset validation conclusion

### T1.1 — `head_sharing_analysis.py:36` uses raw `bias_effect`, not `normalized_restoration`  ⚠️ NEEDS DECISION
`WRITE = "bias_effect"`. Every other Stage-7 script uses `normalized_restoration`
(`segmented_circuit_analysis.py:100`, `segmented_resid_analysis.py:38`) precisely so
datasets with different baseline log-prob gaps are comparable. Raw `bias_effect`
magnitude scales with each dataset's gap.
- **Panel D** (identity×identity *correlation* matrix, `W.corr()` line ~139) is **safe** —
  Pearson correlation per column is scale-invariant.
- **Panels A & C** (per-head WRITE dataset-A vs dataset-B scatter, Pearson r + a `y=x`
  reference line) are **confounded**: the r and the diagonal are scale-dependent across
  datasets. Spearman in those panels is fine.
- **Panels B & E** (heatmaps with a shared `vmax`) let the larger-gap dataset dominate
  the color scale.
- Confidence: high. This is the single most likely thing to weaken a headline figure
  (panel A is "the same heads do bias work in both datasets, r=…").
- **Fix options:** (a) switch `WRITE` to `normalized_restoration` (column exists in both
  raws); (b) keep raw but lead with Spearman and drop the `y=x` line; (c) leave as-is and
  document that only D + Spearman are load-bearing.

### T1.2 — `build_combined_candidates.py:50-53` asymmetric candidate selection  ⚠️ NEEDS DECISION
CrowS is filtered to `bias_score > min_bias` (default 0, keeps ~70% positive-bias rows);
BBQ is **not** filtered here (it arrives pre-curated, already 100% `bias_score>0`). So
"BBQ vs CrowS agreement" (`fig_source_agreement_by_axis`) compares a curated set against
a self-filtered set, which **inflates apparent sign agreement** by construction (you kept
only positive-bias CrowS pairs).
- Confidence: high that it's asymmetric; medium on the magnitude of inflation.
- **Fix options:** (a) document the asymmetry in the figure caption; (b) symmetrize —
  either also positive-filter BBQ, or keep all CrowS and rely on downstream sign stats.

### T1.3 — `parse_crows_pairs.py:104-126` antistereo direction flagged but never reversed  🟡 latent
For `stereo_antistereo == "antistereo"` the comment says direction is reversed, but
target/dominant assignment is unconditional. Safe today because the default keeps
**stereo only**; but `--keep_antistereo` would pool sign-inverted rows into the same axis
means. Recommend: drop antistereo, or actually swap x/y for those rows.

### T1.4 — `build_combined_candidates.py:56` `row_id = range(len)` is order/filter-dependent  🟡
The fresh global `row_id` is positional. All pod raws join back on this `row_id`. If the
combined CSV is ever regenerated with a different filter/order, every existing raw
silently mis-joins by position. Consider a content-hash key, or never regenerate without
re-running patching. (Same class as T2.2 below.)

---

## 🟠 Tier 2 — Confirmed code bugs

### T2.1 — `finalize_bbq_candidates.py:53-59` `--keep_resisted` is broken; `keep_mask` is dead  ⚠️ NEEDS DECISION
`keep_mask` (line 53) is computed and never used. With `--keep_resisted`, line 54's
`(resisted==False) | True` selects **all** stereotypes, and the line-55 override only
fires when `not keep_resisted`. Then line 59 filters `bias_score > 0`, which strips the
resisted (negative-bias) rows the flag was meant to surface. Net: `--keep_resisted`
yields "everything except the thing you asked for."
- Confidence: high. The default path (`keep_resisted=False`) is correct.
- **Question:** what should `--keep_resisted` output — the resisted-only set (bias<0)?
  If so, the `bias_score>0` gate must be conditional on the flag.

### T2.2 — `run_bbq_resid_patching.py:821-840` resume keyed on positional `pair_id`  ⚠️ NEEDS DECISION
`pair_id` is the `pairs.iterrows()` index after `reset_index`. Resume reconstructs
`done_pair_ids` from the existing CSV and skips them. If you resume with a **different**
`--max_pairs`, context/quality filter, or an edited pairs CSV, the pair_id↔pair mapping
shifts and the run silently skips the wrong pairs / mixes two pairings into one file.
- Confidence: high. Crash-resume with the *same* command is safe.
- **Fix:** key resume on `(clean_example_id, corrupt_example_id)`; optionally store a
  hash of the pairs file and refuse to resume if it changed.

### T2.3 — `run_bbq_resid_patching.py` baseline metrics computed twice  ⚪
`compute_prompt_metrics_batch` (lines ~233, 783-786) computes clean/corrupt metrics with
padding, but the per-pair loop recomputes them unpadded (lines 869-870) and **those** are
what's written (914-919). The batch values are only used for a console preview. Not a
correctness bug; dead compute + a divergence risk if padding ever changes. Safe to remove
the batch baseline or actually use it.

---

## 🟡 Tier 3 — Robustness & interpretation caveats (research judgment)

### T3.1 — Per-pair ratio averaging with a loose denom floor (`1e-6`)  ⚠️ worth a decision
MLP AtP fracs (`run_winoqueer_mlp_neuron_attribution.py`, denom cutoff ~line 143) and
steering `bias_fraction` divide a per-pair effect by a per-pair `denom = queer_avg −
control_avg`, then average across pairs. A pair with `denom ≈ 0.001` nat can blow up its
ratio ~1000× and dominate the mean. The rank-percentile `core_score` is insulated, but the
`suff_frac`/`nec_frac` columns, layer-profile masses, and top-neuron bars are **not**.
- **Fix:** raise the denom floor to ~0.05–0.1 nat and/or use median instead of mean for
  per-pair-normalized columns.

### T3.2 — Self-injection at the scored continuation tokens  ⚠️ worth a decision
Two places perturb the residual **at the very tokens whose log-prob is the metric**,
which is partly circular for a localization/steering claim:
- `run_winoqueer_resid_patching.py` patches *every* control position including
  continuation positions → the `continuation` span column of the span heatmap is "model
  re-reading an injected state at the readout," not localization.
- `run_winoqueer_steering_sweep.py:71-72` injects the steering vector at
  `arange(readout, length)` (all continuation tokens), so the induce/de-bias effect is
  inflated by a direct unembed-aligned perturbation at the scored tokens.
- **Recommendation:** for headline figures rely on the **identity-span** (patching) and a
  **readout-only** steering mode; consider adding readout-only injection and comparing.

### T3.3 — Span aggregation is token-count-weighted, not pair-weighted  🟡
`run_bbq_resid_patching.py:287-296` averages `bias_effect` over all (pair, layer,
position) rows in a span, so a pair with a long context contributes more rows and
dominates the `context`/scaffold spans. The identity-token panel is fine (few tokens). If
you want a per-pair estimate, average within pair then across pairs.

### T3.4 — Neuron-Jaccard null pool = top-trimmed neurons, not full neuron space  🟡
`segmented_mlp_circuit_analysis.py:~101` and `winoqueer_segmented_mlp_analysis.py:~147`
compute the random-null Jaccard against `nunique` of the *top-trimmed* neurons, not the
full 14336×32 neuron space. This **inflates the null** → conservatively *understates*
neuron sharing (won't create false positives). Head-Jaccard is honest (all 1024 heads
always present). Decide whether the neuron null should use the full neuron count.

### T3.5 — `compare_segmented_runs.py:53,84` value-sort axes independently per dataset  🟡
The grouped-bar comparison figures sort axes by value within each dataset, so the same
axis isn't vertically alignable across the two dataset blocks — undercuts the "compare
across datasets" purpose. Use a shared categorical axis order.

### T3.6 — Steering random control is a single fixed direction / one seed  🟡
`run_winoqueer_steering_sweep.py:~109` uses one seed-1234 random direction per layer. A
single draw can coincidentally align with the bias direction. Average several seeds for a
rigorous "it's the direction, not the norm" control.

### T3.7 — Steering train/test split is bias-stratified  🟡
`stratified_split` takes the first `cut` rows per predicate after a `bias_score`-descending
sort, so the vector is learned on the highest-bias pairs and tested on lower-bias ones.
Shuffle within group before the cut.

### T3.8 — nonneg BBQ readout slot  ⚠️ worth confirming
For `nonneg` polarity, `biased_letters` is the set of non-target persons, so the readout
slot (`biased_letter`, first entry) is the **distractor/non-target** slot, and
`biased_vs_unknown_logp_diff` uses `max` over biased letters while `p_biased` uses `sum`
(`identify_biased_bbq_items.py:265,273-274`). Confirm that reading the bias metric at the
distractor slot on nonneg items is the intended sign convention — this determines how
nonneg `bias_effect` should be interpreted.

---

## ⚪ Tier 4 — Minor / cross-module hygiene

- **`build_bbq_winoqueer_pairs.py:128`** sets `predicate_label_provisional = social_value`
  (free-text BBQ descriptor), not the WinoQueer 10-category taxonomy. Any cross-dataset
  segmentation by `predicate_label_provisional` compares incommensurable label spaces.
  (CrowS sets it to constant `"CROWS"`.) Predicate-level cross-dataset decomposition is
  therefore not meaningful; axis/identity-level is fine.
- **`build_bbq_patching_pairs.py:174-182`** `DOMINANT_REFERENCE_GROUPS` likely misses some
  real `answer_info` reference tokens (e.g. physical "average"; possibly cis labels) →
  silently shrinks the reference-mode cohort; and it disagrees with
  `bbq_group_taxonomy.reference` (two sources of truth for "reference group").
- **`build_bbq_patching_pairs.py:129-164`** `swap_surfaces` substring/word-boundary edge
  cases (e.g. "Asian" inside "South Asian") and an over-broad alignment gate (checks
  diff-token ∈ union-of-surface-tokens, not that the swap is span-correct). Usually
  degrades to dropping a pair rather than corrupting one — verify the drop count on a real
  run.
- **`score_winoqueer_bias.py:117-128`** tokenizes the prefix in isolation to find the
  continuation start (`cstart`); BPE merges across the prefix/continuation boundary can put
  `cstart` one token early, shifting the scored span and the first-token-rank metric.
  Largely cancels between queer/control but worth a unit check that
  `decode(full_ids[cstart:])` reconstructs the intended continuation.
- **`apply_bbq_review.py:67-69`** 4-key join can silently miss on type mismatch (CSV
  `fillna("")` vs JSON native types) → decisions become "undecided". `:75` `n_edit_notes`
  count is `.drop_duplicates().sum()` → only ever 0/1 (cosmetic print).
- **`winoqueer_segmented_greedy_compare.py:31`** `selected_at_k` assumes 1-row-per-step,
  1-indexed; confirm the curve's `step` convention.
- **`segmented_circuit_analysis.py:384`** `detect_spec` reads only `nrows=200` of the
  cohort to auto-detect columns/umbrella identities; confirm the first 200 WinoQueer rows
  include umbrella identities (LGBTQ/Queer).

---

## ✅ Checked and confirmed CORRECT (highest-risk items that pass)

- `bias_effect` definition & sign throughout (patched − corrupt; queer − ablated; AtP
  suff/nec signs; KL direction). The README formula matches the code in
  `run_bbq_resid_patching.py:65-66,894-895`.
- `continuation_logp` next-token shift slice and the queer-baseline `cont_start`
  arithmetic (end-aligned suffix).
- Letter token ids use leading-space single tokens consistently across all three scorers;
  answer logits read at the true final token.
- `resid_pre` hook site, clean→corrupt direction, per-position single-site patch, hooks
  removed each iteration, `patch_batch_size` numerically equivalent to N=1.
- GQA head indexing via TransformerLens's expanded query-head `hook_z` (32 heads).
- Late-binding closures in ablation/greedy are bound via default args (no classic bug).
- `is_identity_token` detection; dead-leading-column trim is exact-zero-safe by
  construction.
- No invalid raw-token-position cross-pair averaging is produced (only span/identity
  aggregates), per the README's warning.
- `source_agreement` per-source Jaccard (`segmented_circuit_analysis.py`) is valid despite
  grouping by "axis"; `compare_cross_dataset_head_jaccard` null pool is honest (all 1024
  heads present). *(Both were suspected, then verified correct.)*
- BH/RBO/Jaccard/EB-shrink/bootstrap/weighted-η² implementations in the taxonomy libs are
  mathematically correct.
