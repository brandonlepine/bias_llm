# Extending the resume-bias benchmark (jobs, role families, profiles)

The benchmark should not require hand-authoring every job-metadata object or every
candidate profile. This pipeline auto-drafts structured metadata + profile scaffolds
with **provenance + confidence + review flags**, then you review and approve.

Existing hand-authored JSONs are **never modified**; drafts go to `*/drafts/` or
`*/generated/` and are promoted by hand once approved.

## Add a new job posting
```bash
# rule-based extraction -> draft JSON + provenance + review report (no LLM needed)
python -m new_schemas.benchgen.ingest_job --raw-text path/to/posting.txt \
    --company "Acme Aero" --url "https://..." --out new_schemas/job_descriptions/drafts/
```
Produces `<JOB_ID>.json` plus `.extraction_report.json`, `.missing_fields.json`, and
`.inferred_metadata_review.md`. Each field carries `field_provenance`:
`source` ∈ {extracted, inferred, manual}, `confidence`, `evidence`. **Extracted** =
observed in the posting (salary, degree, citizenship, skills…). **Inferred** =
analytical metadata derived by rules (role family, `bias_relevant_dimensions`,
`latent_dimensions`) — these are *not facts in the posting* and must be reviewed.
Rule-based extraction is approximate (e.g. `location.city` and defense flags may need
fixing); the review report lists exactly what to check. Set
`review_status: approved_for_experiment` after correcting, then move the JSON into
`job_descriptions/`. `--extractor llm_assisted` is a planned mode.

## How inferred metadata is reviewed
Open `<JOB_ID>.inferred_metadata_review.md`: it separates **extracted** (trust),
**inferred** (verify — bias/latent dimensions, role family, defense), and
**missing** (fill manually). Only `approved_for_experiment` jobs are used in final
runs (configs can opt into drafts).

## Add a new role family
Author `new_schemas/role_families/<family>.json` (see `mechanical_engineering.json`):
default section order, education fields, typical skills, plausible employers/projects,
professional-org pools (treatment/control/neutral), identity-signal slots, and
bias-relevant occupation priors. (Backbone refactor to select pools per role family
is increment 2; today the generator uses `MECH_ENG_BACKBONE_001`.)

## Generate qualification profiles
```bash
python -m new_schemas.benchgen.profile_factory --grid \
    --out new_schemas/qualification_profiles/generated/
# or from a spec: [{"tier":"associate","archetype":"high_edu_low_exp"}, ...]
python -m new_schemas.benchgen.profile_factory --spec specs.json --out ...
```
Tiers (`entry/strong_entry/associate/mid/senior/staff`) set the experience band +
grad-year window; archetypes (`strong/moderate/weak/high_edu_low_exp/…`) set the 8
quality dimensions; concrete fields + timeline are derived and validated.

## Avoid timeline gaps
The generator dates roles consistently with the grad year (intern before graduation,
FT `{start}–Present`). Verify any run:
```bash
python -m new_schemas.benchgen.validate_resumes --run-dir <run>   # fails on hard timeline errors
```
Reports `computed_experience_years`, `claimed_experience_years`, `grad_year`, and
`timeline_warnings`; flags ENTRY-with-4-years etc. `--allow-timeline-warnings` to bypass.

## Validate examples BEFORE a full run (Prodigy QA)
```bash
python -m new_schemas.benchgen.generate     --config new_schemas/experiments/<small>.json
RUN=$(ls -td new_schemas/runs/*__* | head -1)
python -m new_schemas.benchgen.export_prodigy --run-dir "$RUN" --n 60
prodigy resume-qa my_qa new_schemas/prodigy/resume_qa.jsonl -F new_schemas/benchgen/prodigy_recipes.py
```
One resume per screen with its intended labels (QP tier, arm, composition,
candidate_relative_to_job); flag realism / label issues. No Prodigy? open the JSONL
or use `dump_pairs.py` markdown. Export with `prodigy db-out my_qa`.

## Run a factorial experiment on new jobs
Add the approved `job_id`(s) to an experiments config (e.g. `factorial_channel_x_job.json`)
and run `generate -> run_eval -> diagnostics -> analyze` (see `WHY_FACTORIAL.md`).
The compatibility gate skips jobs whose `resume_generation_axes` don't match a profile's
experience tier, so you won't silently render an early-career backbone under a senior job.

---

## Increment 2: role-family backbones, fit scoring, job validation

### Backbones are selected by role family
Backbones live in `resume_backbones/*.json`, each with a `target_job_family`
(`mechanical_engineering`, `generic_professional`, …). The generator picks the
backbone matching the job's `role.job_family` automatically when a config sets
`"backbones": "auto"` (or omits it); otherwise it uses the explicit list (backward
compatible — existing configs with `"backbones": ["MECH_ENG_BACKBONE_001"]` are
unchanged). Unknown families fall back to `generic_professional`. Add a role family
by dropping a new `resume_backbones/<family>.json` with the same pool structure.
(The renderer is still mech-oriented in its bullet phrasing; per-family bullet
generation is a further extension.)

### Validate a draft job before use
```bash
python -m new_schemas.benchgen.validate_job --job new_schemas/job_descriptions/drafts/<JOB>.json
```
Checks required fields, lists low-confidence inferred fields needing review, and
**exits non-zero unless `review_status: approved_for_experiment`** (or `--allow-draft`).

### Construction fit scores (intended, not model scores)
Every generated resume records `education_match_score`, `experience_match_score`,
`skills_match_score`, `domain_match_score`, `project_match_score`, and
`overall_constructed_fit_score` (0–1), derived from the QP dimensions vs the job.
Use them in analysis to check whether **model** ratings track the **intended**
qualification (e.g. do model scores rise with `overall_constructed_fit_score`?).

### Backward-compatibility check
```bash
python -m new_schemas.benchgen.test_backward_compat   # asserts existing JSONs + both backbone paths still work
```
