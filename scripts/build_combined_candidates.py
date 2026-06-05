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


def gender_subaxis(group: str) -> str:
    """Split the coarse `gender` axis by construct. WinoQueer's gender_identity axis is
    trans/nonbinary IDENTITY bias; BBQ/CrowS `gender` is mostly cis man/woman ROLE bias with a
    trans minority. Routing them to separate axes lets the trans/NB cells be compared directly to
    WinoQueer (gender_identity), while the binary man/woman cells stay a distinct comparison set."""
    g = str(group).lower()
    return "gender_identity" if any(t in g for t in ("trans", "nonbinary", "non-binary", "enby")) else "gender_binary"


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
    ap.add_argument("--no_gender_split", dest="gender_split", action="store_false",
                    help="keep one coarse `gender` axis instead of splitting into "
                         "gender_identity (trans/NB) vs gender_binary (man/woman)")
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
    # Split the coarse `gender` axis into the trans/NB identity construct (aligns with WinoQueer's
    # gender_identity) vs binary man/woman role bias. block is already per-identity, so this only
    # needs to refine the axis label. Toggle off with --no_gender_split to keep one `gender` axis.
    if args.gender_split:
        gmask = combined["axis"] == "gender"
        combined.loc[gmask, "axis"] = combined.loc[gmask, "Group_x"].map(gender_subaxis)
    combined.insert(0, "row_id", range(len(combined)))  # fresh unique key across sources
    combined.to_csv(args.out, index=False)

    print(f"BBQ candidates: {len(bbq)} | CrowS biased: {len(crows)} (of {len(crows_all)}) | combined: {len(combined)}")
    print(f"Wrote {args.out}")
    print("\ncandidate pairs per axis x source:")
    print(combined.pivot_table(index="axis", columns="source", values="row_id", aggfunc="size", fill_value=0).to_string())


if __name__ == "__main__":
    main()
