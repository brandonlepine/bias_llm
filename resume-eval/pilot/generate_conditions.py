#!/usr/bin/env python3
"""Generate the multi-condition stereotype-violation resume set.

For each identity (pair_id) we render ALL conditions in signals.CONDITIONS from
the SAME base resume, sharing one identity. The resumes within a pair_id are
byte-identical except the community-involvement block (organization name +
target-group phrase). This is asserted, not assumed: the entire resume prefix
BEFORE the community block must match across conditions, or generation aborts.

Outputs (new namespace; does NOT clobber the single-token run in generated/):
  generated_conditions/pair_XXXX_<condition_name>.txt
  generated_conditions/manifest.jsonl   # one row per resume

Example:
  python -m pilot.generate_conditions --n-pairs 50 --seed 0
"""
import argparse
import json
import os

from .names import sample_identities
from .signals import CONDITION_SETS, render_condition_block

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COMMUNITY_HEADER = "## COMMUNITY INVOLVEMENT"


def render_resume(template, ident, cond, article):
    block = render_condition_block(cond, article=article)
    return (template
            .replace("{NAME}", ident["full"])
            .replace("{CITY}", ident["city"])
            .replace("{PHONE}", ident["phone"])
            .replace("{EMAIL}", ident["email"])
            .replace("{LINKEDIN}", ident["linkedin"])
            .replace("{COMMUNITY_BLOCK}", block))


def assert_only_community_differs(resumes_by_cond, pair_id):
    """Every resume in a pair must share an identical prefix up to (and including)
    the COMMUNITY INVOLVEMENT header. Only the block after it may vary."""
    prefixes = {}
    for cname, text in resumes_by_cond.items():
        idx = text.find(COMMUNITY_HEADER)
        if idx == -1:
            raise RuntimeError(f"pair {pair_id} {cname}: missing '{COMMUNITY_HEADER}'")
        prefixes[cname] = text[:idx + len(COMMUNITY_HEADER)]
    uniq = set(prefixes.values())
    if len(uniq) != 1:
        raise RuntimeError(
            f"pair {pair_id}: resume content BEFORE the community block differs "
            f"across conditions -- pairing is broken. Check template substitution.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--article", action="store_true",
                    help="insert 'the' before 'local ...' phrases (default: verbatim spec)")
    ap.add_argument("--template", default=os.path.join(ROOT, "templates", "mech-e.template.txt"))
    ap.add_argument("--conditions", default="main", choices=list(CONDITION_SETS),
                    help="which condition set to render (main/neutral/all)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "generated_conditions"))
    args = ap.parse_args()

    conds = CONDITION_SETS[args.conditions]
    with open(args.template) as f:
        template = f.read()
    os.makedirs(args.out_dir, exist_ok=True)
    idents = sample_identities(args.n_pairs, seed=args.seed)

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    n_written = 0
    with open(manifest_path, "w") as mf:
        for pair_id, ident in enumerate(idents):
            base_resume_id = f"mech-e_{args.seed}_{pair_id:04d}"
            rendered = {}
            for cond in conds:
                rendered[cond["condition_name"]] = render_resume(
                    template, ident, cond, args.article)
            assert_only_community_differs(rendered, pair_id)  # integrity gate

            for cond in conds:
                cname = cond["condition_name"]
                fname = f"pair_{pair_id:04d}_{cname}.txt"
                with open(os.path.join(args.out_dir, fname), "w") as rf:
                    rf.write(rendered[cname])
                row = {
                    "pair_id": pair_id,
                    "base_resume_id": base_resume_id,
                    "condition_name": cname,
                    "organization_name": cond["organization_name"],
                    "target_group_phrase": cond["target_group_phrase"],
                    "identity_signal_type": cond["identity_signal_type"],
                    "stereotype_relation": cond["stereotype_relation"],
                    "file": fname,
                    "name": ident["full"],
                    "gender": ident["gender"],
                    "seed": args.seed,
                    "article": args.article,
                }
                mf.write(json.dumps(row) + "\n")
                n_written += 1

    print(f"Wrote {n_written} resumes "
          f"({args.n_pairs} pairs x {len(conds)} conditions) to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Conditions: {[c['condition_name'] for c in conds]}")
    print("Integrity gate PASSED: only the community block varies within each pair.")
    print("Next: python -m pilot.diagnose_conditions   (token diagnostics)")


if __name__ == "__main__":
    main()
