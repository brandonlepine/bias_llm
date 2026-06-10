#!/usr/bin/env python3
"""Generate N token-matched (queer, control) resume pairs from the template.

Within each pair the two resumes share one identity (name/email/phone) and
differ ONLY in the queer-coded community-involvement signal. Across pairs the
identity varies (name + perceived gender) for statistical power.

Outputs:
  generated/pair_XXXX_queer.txt
  generated/pair_XXXX_control.txt
  generated/manifest.jsonl     # one row PER RESUME (2 rows per pair)

Example:
  python -m pilot.generate_pairs --n-pairs 50 --variant minimal --seed 0
"""
import argparse
import json
import os

from .names import sample_identities
from .signals import (render_community_block, DEFAULT_QUEER_WORD,
                       DEFAULT_CONTROL_WORD, SIGNAL_VARIANTS)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # resume-eval/


def render_resume(template, ident, condition, variant, queer_word, control_word):
    block = render_community_block(condition, variant, queer_word, control_word)
    return (template
            .replace("{NAME}", ident["full"])
            .replace("{CITY}", ident["city"])
            .replace("{PHONE}", ident["phone"])
            .replace("{EMAIL}", ident["email"])
            .replace("{LINKEDIN}", ident["linkedin"])
            .replace("{COMMUNITY_BLOCK}", block))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="minimal", choices=list(SIGNAL_VARIANTS))
    ap.add_argument("--queer-word", default=DEFAULT_QUEER_WORD)
    ap.add_argument("--control-word", default=DEFAULT_CONTROL_WORD)
    ap.add_argument("--template", default=os.path.join(ROOT, "templates", "mech-e.template.txt"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "generated"))
    args = ap.parse_args()

    with open(args.template, "r", encoding="utf-8") as f:
        template = f.read()

    os.makedirs(args.out_dir, exist_ok=True)
    idents = sample_identities(args.n_pairs, seed=args.seed)

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for pair_id, ident in enumerate(idents):
            for condition in ("queer", "control"):
                resume = render_resume(template, ident, condition, args.variant,
                                       args.queer_word, args.control_word)
                fname = f"pair_{pair_id:04d}_{condition}.txt"
                fpath = os.path.join(args.out_dir, fname)
                with open(fpath, "w", encoding="utf-8") as rf:
                    rf.write(resume)
                row = {
                    "pair_id": pair_id,
                    "condition": condition,
                    "file": fname,
                    "name": ident["full"],
                    "gender": ident["gender"],
                    "variant": args.variant,
                    "queer_word": args.queer_word,
                    "control_word": args.control_word,
                    "seed": args.seed,
                }
                mf.write(json.dumps(row) + "\n")
                n_written += 1

    print(f"Wrote {n_written} resumes ({args.n_pairs} pairs) to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Variant: {args.variant!r} | queer={args.queer_word!r} control={args.control_word!r}")
    print("Next: python -m pilot.audit_tokens   (verify token-length match)")


if __name__ == "__main__":
    main()
