#!/usr/bin/env python3
"""Build a frozen, balanced segmented cohort from the BBQ candidate pairs (per axis).

Analogous to build_winoqueer_segmented_cohort.py, but the segmentation fields are already columns
in the BBQ data: `axis` (race/gender/…), `block` (= nationality region, else = identity), and
`Group_x` (the target identity). The existing patching run-scripts consume this cohort unchanged via
--no_resort (they read sent_x/sent_y/prefix_y); the segmented analysis joins back on `row_id` to get
axis/identity/block.

Balancing: cap each (block, predicate_label_provisional) cell at --cap (keeping highest bias_score).
Run one cohort per axis (--axis) or all axes together (default).

Outputs (under --out_dir, default data/bbq/results/<axis>/segmented):
  cohort.csv            pre-sorted; carries the run-script + segmentation columns + cohort_pair_id
  cohort_coverage.csv   per (identity, predicate) coverage + block/axis rollups
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CELL = "predicate_label_provisional"
RUN_COLS = ["row_id", "Group_x", "Group_y", "sent_x", "sent_y", "prefix_x", "prefix_y",
            "continuation", "predicate", CELL, "bias_score"]
SEG_COLS = ["category", "axis", "block", "frame"]


def build_one(df: pd.DataFrame, cap: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["bias_score", "row_id"], ascending=False, kind="mergesort")
    kept = df.groupby(["block", CELL], sort=False, group_keys=False).head(cap)
    kept = kept.sort_values(["bias_score", "row_id"], ascending=False, kind="mergesort").reset_index(drop=True)
    kept.insert(0, "cohort_pair_id", range(len(kept)))
    cols = ["cohort_pair_id"] + [c for c in RUN_COLS + SEG_COLS if c in kept.columns]
    kept[cols].to_csv(out_dir / "cohort.csv", index=False)

    cov = kept.groupby(["axis", "block", "Group_x", CELL]).size().rename("n_kept").reset_index()
    cov.to_csv(out_dir / "cohort_coverage.csv", index=False)
    print(f"  {out_dir}/cohort.csv  ({len(kept)} pairs | "
          f"{kept['Group_x'].nunique()} identities | {kept['block'].nunique()} blocks | "
          f"{cov.shape[0]} identity×predicate cells, {int((cov['n_kept']>=30).sum())} with n>=30)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build balanced segmented cohorts from BBQ candidates.")
    ap.add_argument("--candidates", type=Path, default=Path("data/bbq/stereotypes/bbq_candidates_final.csv"))
    ap.add_argument("--out_root", type=Path, default=Path("data/bbq/results"))
    ap.add_argument("--cap", type=int, default=60, help="max pairs per (block, predicate) cell")
    ap.add_argument("--axis", type=str, default=None, help="restrict to one axis (default: one cohort per axis)")
    ap.add_argument("--combined", action="store_true", help="also write a single all-axis cohort")
    args = ap.parse_args()

    d = pd.read_csv(args.candidates)
    axes = [args.axis] if args.axis else sorted(d["axis"].unique())
    print(f"Candidates: {len(d)} | axes: {axes} | cap={args.cap}")
    for ax in axes:
        sub = d[d["axis"] == ax]
        if sub.empty:
            continue
        print(f"\n[{ax}]  {len(sub)} candidate pairs")
        build_one(sub, args.cap, args.out_root / ax / "segmented")
    if args.combined:
        print("\n[ALL]")
        build_one(d, args.cap, args.out_root / "_all" / "segmented")


if __name__ == "__main__":
    main()
