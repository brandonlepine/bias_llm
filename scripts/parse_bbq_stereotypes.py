#!/usr/bin/env python3
"""Extract the stereotype content from the BBQ templates into one flat table.

BBQ encodes each stereotype mainly in `Question_negative_stereotype` (a "Who <verb phrase>?" question),
with `Relevant_social_values` giving a short trait label and `Known_stereotyped_groups` the target
group(s). This script flattens every category's template CSV into one row per template — the worklist
for predicate curation (bbq_predicates_curated.csv) and the downstream pair builder.

Output: data/bbq/stereotypes/bbq_stereotypes_raw.csv
  category, q_id, social_value, target_groups, question, answer_negative, answer_non_negative, stated_gender
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
from pathlib import Path

import pandas as pd

SKIP = {"Filler_items"}
TEMPLATE_GLOB = "new_templates - *.csv"


def parse_groups(value) -> str:
    """Known_stereotyped_groups is a JSON-ish list string; return a clean '; '-joined surface list."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        items = ast.literal_eval(value)
        if isinstance(items, (list, tuple)):
            return "; ".join(str(x).strip() for x in items if str(x).strip())
    except (ValueError, SyntaxError):
        pass
    return value.strip()


def clean(value) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten BBQ templates into a stereotype worklist.")
    ap.add_argument("--templates_dir", type=Path, default=Path("data/bbq/templates"))
    ap.add_argument("--out", type=Path, default=Path("data/bbq/stereotypes/bbq_stereotypes_raw.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(str(args.templates_dir / TEMPLATE_GLOB))):
        category = os.path.basename(path).replace("new_templates - ", "").replace(".csv", "")
        if category in SKIP:
            continue
        df = pd.read_csv(path)
        get = lambda r, c: r[c] if c in df.columns else None
        for _, r in df.iterrows():
            q = clean(get(r, "Question_negative_stereotype"))
            if not q:
                continue  # no negative-stereotype question -> nothing to extract
            rows.append({
                "category": category,
                "q_id": get(r, "Q_id"),
                "social_value": clean(get(r, "Relevant_social_values")),
                "target_groups": parse_groups(get(r, "Known_stereotyped_groups")),
                "question": q,
                "question_non_negative": clean(get(r, "Question_non_negative")),
                "answer_negative": clean(get(r, "Answer_negative")),
                "answer_non_negative": clean(get(r, "Answer_non_negative")),
                "stated_gender": clean(get(r, "Stated_gender_info")),
            })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print(f"Wrote {args.out}  ({len(out)} stereotype rows from {out['category'].nunique()} categories)")
    print("\nrows per category:")
    print(out.groupby("category").size().to_string())
    missing_groups = out[out["target_groups"] == ""]
    print(f"\nrows with no target_groups (intersectional / unlabeled): {len(missing_groups)} "
          f"({sorted(missing_groups['category'].unique())})")
    print(f"distinct social_values: {out['social_value'].nunique()}")


if __name__ == "__main__":
    main()
