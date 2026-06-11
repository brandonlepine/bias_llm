# Resume-bias benchmark (new_schemas) — command reference

Orthogonal-factorial identity-signal benchmark for LLM hiring evaluation. Run all
commands from `resume-eval/` with the repo venv (`../.venv/bin/python -m ...`).
Design rationale: `WHY_FACTORIAL.md`. Extending jobs/profiles: `EXTENDING_BENCHMARK.md`.

## Pipeline at a glance
```
ingest_job ─► validate_job ─► (approve) ─► job_descriptions/
profile_factory ─► qualification_profiles/generated/
                         │
experiments/<cfg>.json ──┴─► generate ─► validate_resumes ─► export_prodigy (QA)
                                   │
                                   └─► run_eval ─► diagnostics ─► analyze ─► export_figures
```

## 1. Add a job (auto-draft from raw text)
```bash
python -m new_schemas.benchgen.ingest_job --raw-text posting.txt --company "Acme" \
    --url "https://..." --out new_schemas/job_descriptions/drafts/
python -m new_schemas.benchgen.validate_job --job new_schemas/job_descriptions/drafts/<JOB>.json
# review the .inferred_metadata_review.md, fix flagged fields, set
# review_status: approved_for_experiment, then move the JSON into job_descriptions/
```

## 2. Generate qualification profiles
```bash
python -m new_schemas.benchgen.profile_factory --grid --out new_schemas/qualification_profiles/generated/
# or: --spec specs.json   ([{"tier":"associate","archetype":"high_edu_low_exp"}, ...])
```

## 3. Generate examples (orthogonal factorial)
```bash
python -m new_schemas.benchgen.generate --config new_schemas/experiments/factorial_3ch_base.json
RUN=$(ls -td new_schemas/runs/*__factorial_3ch_base | head -1)
```
Configs: `factorial_3ch_base` (2³ channels), `_explicitness` / `_location` / `_salience`
(vary one factor), `factorial_channel_x_qualification`, `factorial_channel_x_job`.
Set `"backbones": "auto"` to select the backbone by the job's role family.

## 4. QA BEFORE a full run
```bash
python -m new_schemas.benchgen.validate_resumes --run-dir "$RUN"          # timeline gaps -> non-zero exit
python -m new_schemas.benchgen.export_prodigy   --run-dir "$RUN" --n 60   # -> new_schemas/prodigy/resume_qa.jsonl
prodigy resume-qa my_qa new_schemas/prodigy/resume_qa.jsonl -F new_schemas/benchgen/prodigy_recipes.py
prodigy db-out my_qa > qa_annotations.jsonl                               # export your judgments
# no Prodigy? inspect: python -m new_schemas.benchgen.dump_pairs --run-dir "$RUN"
```

## 5. Score (deterministic logit readouts — NO sampling/temperature)
```bash
python -m new_schemas.benchgen.run_eval --run-dir "$RUN" --model meta-llama/Llama-3.1-8B-Instruct
# base model: --model ../models/Llama-3.1-8B   (watch the number-mass QC line)
```

## 6. Analyze
```bash
python -m new_schemas.benchgen.diagnostics --scored "$RUN/scored.jsonl"  # composition, rank+cond#, interactions, fractions, expanded regression
python -m new_schemas.benchgen.analyze     --scored "$RUN/scored.jsonl"  # figures (debiased pairwise, floor-normalized heatmap)
```

## 7. Get figures/outputs to a local checkout
```bash
python -m new_schemas.benchgen.export_figures --run-dir "$RUN"   # -> tracked new_schemas/figures/<tag>/ (git add/commit/push, then pull)
# or scp from the pod:
scp -P <PORT> -i ~/.ssh/<key> -r root@<HOST>:~/bias_llm/resume-eval/new_schemas/runs/<RUN> \
    /local/.../resume-eval/new_schemas/runs/
```

## Maintenance
```bash
python -m new_schemas.benchgen.test_backward_compat   # existing JSONs + backbone selection still work
```

## Modules
`ingest_job` `validate_job` `profile_factory` `generate` `validate_resumes`
`export_prodigy`+`prodigy_recipes` `run_eval` `diagnostics` `analyze` `export_figures`
`dump_pairs` `tokens_audit` `loaders` `render` `test_backward_compat`.
