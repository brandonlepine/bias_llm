# Organizational decision benchmarks

**Conceptual claim:** this benchmark studies **identity-conditioned allocation of
organizational resources, status, credibility, and opportunity under AI-mediated
decision-making** — generalizing the hiring benchmark beyond selection-at-the-door.

The hiring benchmark is unchanged. This layer adds new *decision domains* on top of
the same identity factorial, paired design, deterministic readouts, noise floor, and
diagnostics.

## Why beyond hiring
Hiring bias is one allocation decision. The same model mediates many others where
identity may shift outcomes — often *more* subtly (e.g. as credibility discounting or
"potential" skepticism rather than a competence score). Each domain measures a
different organizational resource:

| domain | resource at stake | status | what bias looks like |
|---|---|---|---|
| hiring_selection | offer / interview slot | entry | lower score / not chosen |
| **promotion_advancement** | level / title | internal status | lower readiness / "not ready yet" / leadership skepticism |
| opportunity_allocation | stretch / funding / mentorship / visibility | trajectory | not selected for scarce opportunity |
| compensation_allocation | money / equity | material reward | lower allocation / below a real increment |
| performance_evaluation | rating / narrative | reputation | lower rating / warmth-not-competence language |
| trust_credibility | credibility / signoff authority | epistemic status | judgment discounted in disagreements |
| discipline_accountability | sanction severity | penalty | harsher sanction for ambiguous issues |

## How identity signals are reused
The identity factorial (channel × salience × explicitness × professional-relevance ×
location × load) is **surface-agnostic**. `identity_signal_surface` selects where the
signal lives: `resume_section` (hiring), `employee_bio` (promotion/opportunity),
`performance_review_history`, `project_history`, `professional_development_record`,
`leadership_record`. Same `by_channel` triples, same dose configs.

## Isolated scoring vs pairwise vs ranking/selection
- **Isolated** (score_0_100 / yes-no): per-candidate; bias = paired control−treatment
  delta vs the noise floor.
- **Pairwise** (A/B): treatment vs matched control head-to-head; the raw A/B logit is
  position-saturated, so use the **position-debiased** preference
  `T=(logit_AB|treat=A − logit_AB|treat=B)/2`.
- **Ranking / scarce selection** (choose K of N): bias = treatment vs control
  **selection rate / odds ratio / position-adjusted preference**, with the focal
  candidate **rotated through every position** to remove position bias (runnable; K=1).

## What's runnable now
- **hiring_selection** — full (existing benchgen).
- **promotion_advancement** — runnable: `orgbench/generate_promotion.py` renders
  internal employee profiles + the identity factorial + promotion prompts, emitting
  rows in the **same format**, so `run_eval` and `diagnostics` work unchanged.
  Prompt conditions reuse the hiring readouts: `promotion_readiness_score_0_100`,
  `leadership_potential_score_0_100`, `promote_yes_no`, `years_until_ready`.
  Promotion lets identity effects show up in **leadership/potential** judgments, not
  only technical competence.

```bash
python -m new_schemas.orgbench.generate_promotion --config new_schemas/experiments/promotion_pilot.json
RUN=$(ls -td new_schemas/runs/*__promotion_pilot | head -1)
python -m new_schemas.benchgen.run_eval    --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct
python -m new_schemas.benchgen.diagnostics --scored "$RUN/scored.jsonl"   # channel/dose/floor diagnostics, all reused
python -m new_schemas.benchgen.analyze     --scored "$RUN/scored.jsonl"
```

- **opportunity_allocation / scarce selection** — runnable: `orgbench/generate_selection.py`
  builds slates of N equally-qualified candidates with ONE focal candidate carrying the
  identity signal, ROTATED through every position; `run_eval` (output_type `selection_n`)
  reads the choice distribution over candidate numbers; `orgbench/analyze_selection.py`
  reports focal selection rate by variant, odds ratio, **position-adjusted preference**
  (paired same-position Δp_focal vs the control-neutral floor), and a **position-bias**
  table. Position bias is strong, which is why the focal is counterbalanced across all
  positions and the paired Δ cancels it.

```bash
python -m new_schemas.orgbench.generate_selection --config new_schemas/experiments/opportunity_selection_pilot.json
RUN=$(ls -td new_schemas/runs/*__opportunity_selection_pilot | head -1)
python -m new_schemas.benchgen.run_eval          --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct
python -m new_schemas.orgbench.analyze_selection --scored "$RUN/scored.jsonl"
```

## More runnable domains
- **compensation_allocation** — runnable: `orgbench/generate_compensation.py` renders
  per-employee bonus / merit-raise / equity decisions over DISCRETE realistic increments
  (bonus $1k, raise 1%, equity 250 shares; output_type bonus_increment) + equity yes/no
  (binary). run_eval scores them; the diagnostics MONETARY section gives exact-match
  rate + modal offer per arm (unit-agnostic). Sub-increment EV is never reported as a $
  effect. config: compensation_pilot.json.

```bash
python -m new_schemas.orgbench.generate_compensation --config new_schemas/experiments/compensation_pilot.json
RUN=$(ls -td new_schemas/runs/*__compensation_pilot | head -1)
python -m new_schemas.benchgen.run_eval    --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct
python -m new_schemas.benchgen.diagnostics --scored "$RUN/scored.jsonl"
```

- **performance_evaluation** — runnable: `orgbench/generate_performance.py`. Numeric
  `performance_rating_0_100` reuses the score readout (base or instruct). Text outputs
  (`review_text`, `strengths_weaknesses_text`, `promotion_narrative_text`) use
  **deterministic greedy generation** (output_type `text_generation`, `--max-new-tokens`)
  — **Instruct only** (base Llama emits no text). `orgbench/analyze_text.py` extracts
  lexicon features (warmth / competence / agency / certainty / hedging / risk /
  leadership / technical_credibility) per 100 words and reports **paired control−treatment
  deltas vs the control−neutral floor** (+ = treatment text has LESS of the feature; e.g.
  warmth-up + competence-down = warmth-not-competence). Lexicon features are coarse proxies.

```bash
python -m new_schemas.orgbench.generate_performance --config new_schemas/experiments/performance_pilot.json
RUN=$(ls -td new_schemas/runs/*__performance_pilot | head -1)
python -m new_schemas.benchgen.run_eval    --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 220
python -m new_schemas.benchgen.diagnostics --scored "$RUN/scored.jsonl"   # numeric rating
python -m new_schemas.orgbench.analyze_text --scored "$RUN/scored.jsonl"  # text features
```

- **trust_credibility** — runnable: `orgbench/generate_trust.py`. Two equally-qualified
  engineers give competing (symmetric) analyses of a `technical_artifact`; the FOCAL
  engineer carries the identity signal; both orders (focal=A/B). `run_eval` (output_type
  `pairwise_direct`) reads logit(A)−logit(B); `orgbench/analyze_trust.py` reports the
  POSITION-DEBIASED focal credibility preference `T=(logit_AB|focal=A − focal=B)/2` and
  the identity effect (treat−control) vs the control−neutral floor. **Negative = the
  identity engineer's judgment is DISCOUNTED** — bias as credibility discounting, not a
  lower score.

```bash
python -m new_schemas.orgbench.generate_trust --config new_schemas/experiments/trust_pilot.json
RUN=$(ls -td new_schemas/runs/*__trust_pilot | head -1)
python -m new_schemas.benchgen.run_eval     --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct
python -m new_schemas.orgbench.analyze_trust --scored "$RUN/scored.jsonl"
```

## Staged next
- **full ranking** (order all N / select K>1) extends the selection readout (parse a
  ranked list); scarce selection (K=1) is done.
- **discipline_accountability** — schema present, `implementation_status: placeholder`
  (kept separate; sensitive).

## Add a new scenario
1. Add a domain to `decision_domains/decision_domains.json` (or reuse one).
2. Add a template to `decision_scenarios/<name>.json` (scenario_id, decision_domain,
   prompt_template, candidate_count, selection_count, outcome_schema, required_inputs,
   evaluation_metrics, randomization_rules, counterbalancing_rules).
3. For per-profile scoring, emit benchgen-format rows (see `generate_promotion.py`) and
   reuse `run_eval`/`diagnostics`. For ranking/selection, use the (forthcoming) ranking
   readout.

## Interpreting effects
Same hierarchy as hiring (`WHY_FACTORIAL.md`, factor-aware diagnostics): channel main
effects vs interactions vs dose-response; monetary on the **actual increment** with
exact-match rate; choice-forcing (pairwise/ranking) reveals preferences invisible in
isolated scoring; always vs the per-outcome **noise floor**. Do not collapse to
"identity load."
