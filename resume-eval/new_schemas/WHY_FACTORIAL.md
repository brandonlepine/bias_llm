# Identity-signal designs: why cumulative "load" is confounded and factorial is required

## The problem with the cumulative-load design
The first design varied a single ordinal factor, `identity_load`, as a **cumulative
ladder** of channels:

| load | composition |
|---|---|
| L1 | affiliation *or* conference *or* scholarship |
| L2 | affiliation+scholarship |
| L3 | affiliation+conference+scholarship |
| L4 | + leadership |
| L5 | + volunteer |

Because each higher load is (mostly) a **superset** of the lower one, `load` and the
exact channel **composition** are nearly 1:1, and channels are **nested**, not
crossed. Consequences observed in `pilot_v2` diagnostics:

- The "L3 salary anomaly" (L2 ≈ +$210, **L3 ≈ −$220**, L4 ≈ +$84) was **not** a load
  effect, not sparse (n=72/cell), and not driven by outliers (the top-4 negative
  salary deltas were *all* `load3_all`). It was a **conference-involving interaction**.
- A regression `delta ~ affiliation*conference*scholarship` was **rank-deficient
  (7/10, cond# ≈ 1e30)**: `affiliation:conference` and `conference:scholarship`
  came back **identical** — the coefficients are *aliased* and cannot be separated.

**Rule:** in a cumulative design, `load` is composition-confounded; you can detect
that an interaction exists but cannot attribute it. Do **not** interpret "load" as a
causal quantity there.

## The factorial design
`factorial_3ch_base` crosses the three core channels as an orthogonal **2³**:

```
none, A, C, S, AC, AS, CS, ACS   (channel_*_present indicators)
```

The channel design matrix `[1, A, C, S, AC, AS, CS, ACS]` is **full rank 8/8
(cond# ≈ 18)**, so channel **main effects and all interactions are independently
identified**. `identity_load` is then *derived* (= number of active channels), and
every load plot is accompanied by a composition breakdown — load is a summary, not a
factor.

## Orthogonally-varied factors (each independent of channel)
| factor | levels | config |
|---|---|---|
| signal channel | 2³ over affiliation/conference/scholarship (extensible to leadership/volunteer/presentation) | `factorial_3ch_base` |
| signal salience | low / moderate / high / leadership (role word: Member→Active Member→Chapter Officer→President) | `factorial_3ch_salience` |
| identity explicitness | organization_name_only / subtle / explicit / strongly_explicit | `factorial_3ch_explicitness` |
| resume location | bottom_section / leadership_section / mid_resume | `factorial_3ch_location` |
| qualification profile | entry / strong_entry / associate_* (+ mixed) | `factorial_channel_x_qualification` |
| job posting | the entry-level postings | `factorial_channel_x_job` |
| prompt condition | 6 outcomes | all |
| name variant | M/F/A | all |

Salience is **not** channel (a conference can be attend vs organize vs chair);
relevance, explicitness, and location are likewise stored as their own metadata
fields, so they can be crossed or held fixed per config.

## Causal interpretation
- Primary causal quantities = **channel main effects + interactions** (from the
  factorial), each calibrated against the control-vs-neutral **noise floor**.
- `control − treatment > 0` ⇒ identity-signaled candidate **penalized** (lower
  score / salary / bonus). Pairwise uses the **position-debiased** preference
  `T = (logit_AB|treat=A − logit_AB|treat=B)/2` because the raw A/B choice is
  saturated by position-A primacy.
- Paired design preserved: base resume is **byte-identical across
  treatment/control/neutral**; only the token-diagnosed signal block differs
  ("base-identical with token-diagnosed signal blocks").

## Workflow
```
python -m new_schemas.benchgen.generate    --config new_schemas/experiments/factorial_3ch_base.json
RUN=$(ls -td new_schemas/runs/*__factorial_3ch_base | head -1)
python -m new_schemas.benchgen.run_eval    --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct
python -m new_schemas.benchgen.diagnostics --scored "$RUN/scored.jsonl"   # composition, rank+cond#, interactions, fractions, expanded regression
python -m new_schemas.benchgen.analyze     --scored "$RUN/scored.jsonl"   # figures + debiased pairwise
```
