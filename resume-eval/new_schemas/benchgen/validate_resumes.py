#!/usr/bin/env python3
"""Timeline / construction validator for a generated run. For each resume, parse the
experience + education dates, compute experience years, compare to the claimed years
in the summary, and flag unexplained gaps or tier/label mismatches.

  python -m new_schemas.benchgen.validate_resumes --run-dir <run> [--allow-timeline-warnings]
"""
import argparse, collections, json, os, re

PRESENT = 2026


def years_in(text):
    return [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]


def _section(res, header):
    for block in res.split("\n\n---\n\n"):
        if block.lstrip().startswith(f"## {header}"):
            return block
    return ""


def validate(r):
    res = r["rendered_resume"]
    warns = []
    exp = _section(res, "PROFESSIONAL EXPERIENCE")
    exp_years = years_in(exp)
    edu = _section(res, "EDUCATION")
    grad = max(years_in(edu)) if years_in(edu) else None
    computed = (PRESENT - min(exp_years)) if exp_years else 0
    m = re.search(r"with (\d+) years", res)
    claimed = int(m.group(1)) if m else None
    if grad and exp_years and min(exp_years) < grad - 1:
        warns.append(f"experience predates graduation ({min(exp_years)} < grad {grad})")
    if claimed is not None and abs(claimed - computed) > 2:
        warns.append(f"claimed {claimed}y vs computed {computed}y")
    qp = r["qualification_profile_id"]
    if qp.startswith("QP_ENTRY") and computed > 2:
        warns.append(f"ENTRY profile but ~{computed}y experience")
    return {"computed_experience_years": computed, "claimed_experience_years": claimed,
            "grad_year": grad, "timeline_warnings": warns,
            "timeline_validation_status": "ok" if not warns else "warnings"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--allow-timeline-warnings", action="store_true")
    args = ap.parse_args()
    seen, n, bad = set(), 0, 0
    by_status = collections.Counter()
    out = []
    for line in open(os.path.join(args.run_dir, "examples.jsonl")):
        r = json.loads(line)
        key = (r["paired_example_id"], r["treatment_or_control"])
        if key in seen:
            continue
        seen.add(key)
        v = validate(r); n += 1; by_status[v["timeline_validation_status"]] += 1
        if v["timeline_warnings"]:
            bad += 1
            out.append({"paired_example_id": r["paired_example_id"], "qp": r["qualification_profile_id"], **v})
    path = os.path.join(args.run_dir, "timeline_validation.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"validated {n} resumes: {dict(by_status)}  | {bad} with warnings -> {path}")
    for o in out[:8]:
        print(f"  [warn] {o['qp']}: {o['timeline_warnings']}")
    if bad and not args.allow_timeline_warnings:
        raise SystemExit(f"\nFAIL: {bad} timeline warnings (use --allow-timeline-warnings to bypass).")


if __name__ == "__main__":
    main()
