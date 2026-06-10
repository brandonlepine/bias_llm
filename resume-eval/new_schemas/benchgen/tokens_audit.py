#!/usr/bin/env python3
"""Token-count diagnostics over a generated run (deliverable 3).

For every paired group reports, with the exact tokenizer, the treatment-vs-control
and control-vs-neutral (noise-floor) token deltas for the inserted identity phrase,
the full inserted section, and the full resume. Naturalistic signals are NOT forced
to token-equality (token_match_mode defaults to diagnostic_only); this script makes
the resulting length confounds explicit so the floor can absorb them.

Run from resume-eval/:
  python -m new_schemas.benchgen.tokens_audit --run-dir new_schemas/runs/<ts>__<exp>
"""
import argparse
import collections
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "examples.jsonl"))]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["paired_example_id"]][r["treatment_or_control"]] = r

    tc_resume, tc_sec, cn_resume = [], [], []
    per_load = collections.defaultdict(list)
    for pid, arms in by.items():
        if "treatment" not in arms or "control" not in arms:
            continue
        t, c = arms["treatment"], arms["control"]
        d_res = t["token_counts"]["resume_tokens"] - c["token_counts"]["resume_tokens"]
        d_sec = t["token_counts"]["identity_section_tokens"] - c["token_counts"]["identity_section_tokens"]
        tc_resume.append(d_res); tc_sec.append(d_sec)
        per_load[t["identity_load"]].append(d_res)
        if "neutral" in arms:
            cn_resume.append(c["token_counts"]["resume_tokens"] - arms["neutral"]["token_counts"]["resume_tokens"])

    def stat(xs):
        return f"n={len(xs)} mean={statistics.mean(xs):+.1f} min={min(xs):+d} max={max(xs):+d}" if xs else "n=0"

    print(f"=== TOKEN AUDIT: {os.path.basename(args.run_dir)} ===")
    print(f"paired groups (treatment+control): {len(tc_resume)}")
    print(f"\nTREATMENT - CONTROL  (the identity confound the floor must absorb):")
    print(f"  identity section tokens: {stat(tc_sec)}")
    print(f"  full resume tokens:      {stat(tc_resume)}")
    for load in sorted(per_load):
        print(f"    load {load}: resume {stat(per_load[load])}")
    print(f"\nCONTROL - NEUTRAL  (noise floor; should be ~0 -> floor is clean):")
    print(f"  full resume tokens:      {stat(cn_resume)}")
    if tc_resume and cn_resume and abs(statistics.mean(tc_resume)) > 3 * (abs(statistics.mean(cn_resume)) + 1):
        print("\n[note] treatment-control length gap >> control-neutral gap: identity signals are\n"
              "       systematically longer. Expected for naturalistic signals; rely on the floor,\n"
              "       or set token_match_mode='within_tolerance' + padding to tighten (future).")


if __name__ == "__main__":
    main()
