#!/usr/bin/env python3
"""Pool BBQ + CrowS-Pairs biased candidates per axis, with a `source` column for clean separation.

Both datasets share the same axes and the same continuation-scoring schema. This concatenates the
BBQ final candidates (template-based) and the CrowS biased pairs (naturalistic) into ONE candidate
table tagged `source` (bbq / crows-pairs), with a fresh unique `row_id` (so the two streams never
collide downstream). Everything downstream (cohort builder, segmented analysis) can pool by `axis`
yet split by `source` whenever we want to report them separately.

Output: data/combined/bbq_crows_candidates.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

# columns kept in the unified candidate schema
COLS = ["source", "axis", "block", "Group_x", "Group_y", "predicate_label_provisional", "frame",
        "bias_score", "sent_x", "sent_y", "prefix_x", "prefix_y", "continuation", "category",
        "source_row_id", "identity_mapped"]


def harmonize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source
    df["source_row_id"] = df["row_id"] if "row_id" in df.columns else range(len(df))
    if "identity_mapped" not in df.columns:
        df["identity_mapped"] = True  # BBQ identities are canonical by construction
    if source == "crows-pairs":  # make block a per-identity grouping like BBQ (axis for unmapped)
        df["block"] = [g if m else a for g, m, a in zip(df["Group_x"], df["identity_mapped"], df["axis"])]
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pool BBQ + CrowS candidates with a source tag.")
    ap.add_argument("--bbq", type=Path, default=Path("data/bbq/stereotypes/bbq_candidates_final.csv"))
    ap.add_argument("--crows_scored", type=Path, default=None,
                    help="crows_pairs_scored.csv (default: auto-find under pod_results)")
    ap.add_argument("--out", type=Path, default=Path("data/combined/bbq_crows_candidates.csv"))
    ap.add_argument("--min_bias", type=float, default=0.0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # BBQ arrives already curated to bias_score>0 (finalize_bbq_candidates.py); CrowS is
    # filtered to the same bias_score>min_bias criterion here. Same inclusion rule, applied
    # at whichever step each dataset is built.
    bbq = harmonize(pd.read_csv(args.bbq), "bbq")
    crows_path = args.crows_scored or Path(sorted(glob.glob("pod_results/**/crows_pairs_scored.csv", recursive=True))[-1])
    crows_all = pd.read_csv(crows_path)
    crows = harmonize(crows_all[crows_all["bias_score"] > args.min_bias], "crows-pairs")

    combined = pd.concat([bbq, crows], ignore_index=True)
    combined.insert(0, "row_id", range(len(combined)))  # fresh unique key across sources
    combined.to_csv(args.out, index=False)

    print(f"BBQ candidates: {len(bbq)} | CrowS biased: {len(crows)} (of {len(crows_all)}) | combined: {len(combined)}")
    print(f"Wrote {args.out}")
    print("\ncandidate pairs per axis x source:")
    print(combined.pivot_table(index="axis", columns="source", values="row_id", aggfunc="size", fill_value=0).to_string())


if __name__ == "__main__":
    main()
