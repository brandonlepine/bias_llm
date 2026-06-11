#!/usr/bin/env python3
"""Validate a (draft or approved) job JSON: required fields present, low-confidence
inferred fields flagged for review, review_status gate.

  python -m new_schemas.benchgen.validate_job --job new_schemas/job_descriptions/drafts/<JOB>.json
  # exits non-zero unless review_status == approved_for_experiment (or --allow-draft)
"""
import argparse, json, sys

REQUIRED = [("job_id",), ("employer", "company_name"), ("role", "job_title"),
            ("compensation", "salary_min"), ("compensation", "salary_max"), ("raw_posting_text",)]


def dig(d, path):
    cur = d
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.7)
    args = ap.parse_args()
    job = json.load(open(args.job))

    missing = [".".join(p) for p in REQUIRED if dig(job, p) in (None, "")]
    prov = job.get("field_provenance", {})
    low_conf = [(k, v) for k, v in prov.items() if v.get("source") == "inferred" and (v.get("confidence") or 0) < args.min_confidence]
    listed_missing = job.get("missing_fields", [])
    status = job.get("review_status", "draft_unreviewed" if prov else "manual")

    print(f"job_id: {job.get('job_id')}  | review_status: {status}")
    print(f"required fields missing: {missing or 'none'}")
    print(f"listed missing_fields: {listed_missing or 'none'}")
    print(f"low-confidence inferred (<{args.min_confidence}) needing review: {len(low_conf)}")
    for k, v in low_conf:
        print(f"  - {k} (conf {v.get('confidence')}): {v.get('evidence')}")

    hard_fail = bool(missing)
    if hard_fail:
        sys.exit(f"\nFAIL: required fields missing: {missing}")
    if status != "approved_for_experiment" and not args.allow_draft:
        sys.exit(f"\nNOT APPROVED (review_status={status}). Review/correct, set "
                 f"review_status: approved_for_experiment, or pass --allow-draft.")
    print("\nOK: usable" + (" (draft, --allow-draft)" if status != "approved_for_experiment" else ""))


if __name__ == "__main__":
    main()
