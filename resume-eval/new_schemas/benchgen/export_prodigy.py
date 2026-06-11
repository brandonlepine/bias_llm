#!/usr/bin/env python3
"""Export a (stratified) sample of generated resumes as a Prodigy stream for
human QA -- validate realism + that the metadata LABELS are right BEFORE running a
full eval. Single resume per screen with its intended labels shown; reviewer flags
issues via choice options.

  # 1. generate a small sample run first, then:
  python -m new_schemas.benchgen.export_prodigy --run-dir <run> --n 60 \
      --out new_schemas/prodigy/resume_qa.jsonl
  # 2. (with Prodigy installed) review:
  prodigy resume-qa my_qa new_schemas/prodigy/resume_qa.jsonl -F new_schemas/benchgen/prodigy_recipes.py
"""
import argparse, collections, html, json, os, random

SCHEMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIONS = [
    {"id": "realistic", "text": "✓ Realistic & labels correct"},
    {"id": "not_realistic", "text": "Resume not plausible"},
    {"id": "qp_tier_wrong", "text": "QP tier doesn't match resume"},
    {"id": "signal_wrong", "text": "Identity/control signal block wrong or unrealistic"},
    {"id": "cand_rel_wrong", "text": "candidate_relative_to_job looks wrong"},
    {"id": "other", "text": "Other issue (add note)"},
]


def task_html(r):
    meta = (f"<b>intended labels</b> &mdash; QP: <code>{r['qualification_profile_id']}</code> | "
            f"arm: <code>{r['treatment_or_control']}</code> | composition: <code>{r.get('exact_signal_composition','?')}</code> | "
            f"cand_rel_to_job: <code>{r.get('candidate_relative_to_job','?')}</code> | "
            f"salience: <code>{r.get('signal_salience_level','?')}</code> | "
            f"location: <code>{r.get('resume_location_level','?')}</code> | "
            f"job: <code>{r['job_id']}</code> | resume_tokens: {r['token_counts']['full_resume_tokens']}")
    resume = html.escape(r["rendered_resume"])
    return (f"<div style='font-family:sans-serif'>"
            f"<div style='background:#f0f4f8;padding:8px;border-radius:6px;margin-bottom:8px'>{meta}</div>"
            f"<pre style='white-space:pre-wrap;font-size:12px;line-height:1.35'>{resume}</pre></div>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=60, help="sample size (stratified by qual x arm x composition)")
    ap.add_argument("--out", default=os.path.join(SCHEMA_DIR, "prodigy", "resume_qa.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # one row per (pair, arm) -- resume doesn't depend on prompt condition
    seen, rows = set(), []
    for line in open(os.path.join(args.run_dir, "examples.jsonl")):
        r = json.loads(line)
        key = (r["paired_example_id"], r["treatment_or_control"])
        if key in seen:
            continue
        seen.add(key); rows.append(r)

    # stratified sample across (qual, arm, composition)
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["qualification_profile_id"], r["treatment_or_control"], r.get("exact_signal_composition", ""))].append(r)
    rng = random.Random(args.seed)
    picked, keys = [], list(strata)
    rng.shuffle(keys)
    i = 0
    while len(picked) < min(args.n, len(rows)):
        k = keys[i % len(keys)]
        if strata[k]:
            picked.append(strata[k].pop(rng.randrange(len(strata[k]))))
        i += 1
        if i > len(rows) * 4:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps({
                "html": task_html(r), "options": OPTIONS,
                "meta": {"paired_example_id": r["paired_example_id"], "arm": r["treatment_or_control"],
                         "qualification_profile_id": r["qualification_profile_id"],
                         "composition": r.get("exact_signal_composition"), "job_id": r["job_id"]},
            }) + "\n")
    print(f"wrote {len(picked)} Prodigy QA tasks -> {args.out}")
    print("review with:  prodigy resume-qa <dataset> " + args.out +
          " -F new_schemas/benchgen/prodigy_recipes.py")
    print("(no Prodigy? open the JSONL or use dump_pairs.py markdown for manual review.)")


if __name__ == "__main__":
    main()
