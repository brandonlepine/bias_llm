#!/usr/bin/env python3
"""Build the frozen, balanced cohort for the identity-segmented WinoQueer circuit analysis.

All four circuit run-scripts internally re-sort the pairs by `bias_score` and emit a *positional*
`pair_id`, so `pair_id` is NOT a stable key across scripts. This builder produces ONE pre-sorted
cohort that every run-script consumes in file order (via their `--no_resort` flag), with a stable
`cohort_pair_id` and the `row_id` that all segmentation joins actually key on.

Balancing: cap each `Gender_ID_x x predicate_label_provisional` cell at `--cap` (default 100),
keeping the highest-`bias_score` pairs.

Outputs (under --out_dir, default data/winoqueer/results/segmented):
  cohort.csv           pre-sorted; carries every column the run-scripts read + axis/identity/is_umbrella/cohort_pair_id
  cohort_coverage.csv  per-cell n_available / n_kept / capped / qualifies(n>=MIN_CELL) / structural_zero + rollups
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winoqueer_identity_taxonomy import (  # noqa: E402
    MIN_CELL, PREDICATE_CATEGORIES, IDENTITY_AXIS, annotate, axis_mismatch_count,
)

# Columns the run-scripts read (align_pair needs sent_x/sent_y/prefix_y; raws re-emit the rest).
RUN_SCRIPT_COLS = [
    "row_id", "Gender_ID_x", "Gender_ID_y", "sent_x", "sent_y", "prefix_x", "prefix_y",
    "continuation", "predicate", "predicate_label_provisional", "bias_score",
]
CELL_COL = "predicate_label_provisional"


def build_coverage(annotated: pd.DataFrame, kept: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Per (identity x predicate_label) coverage, with axis/identity rollups appended."""
    avail = annotated.groupby(["identity", CELL_COL]).size().rename("n_available")
    keptn = kept.groupby(["identity", CELL_COL]).size().rename("n_kept")
    # Every identity x category combination, so structural zeros are explicit.
    identities = sorted(annotated["identity"].unique())
    full_idx = pd.MultiIndex.from_product([identities, PREDICATE_CATEGORIES], names=["identity", CELL_COL])
    cov = pd.DataFrame(index=full_idx).join(avail).join(keptn).fillna(0).astype(int).reset_index()
    cov["axis"] = cov["identity"].map(IDENTITY_AXIS)
    cov["capped"] = cov["n_available"] > cap
    cov["qualifies"] = cov["n_kept"] >= MIN_CELL
    cov["structural_zero"] = cov["n_available"] == 0
    cov["level"] = "identity_x_predicate"

    rollups = []
    for col in ["axis", "identity"]:
        r = kept.groupby(col).size().rename("n_kept").reset_index()
        a = annotated.groupby(col).size().rename("n_available").reset_index()
        r = r.merge(a, on=col, how="outer").fillna(0)
        r["level"] = col
        rollups.append(r.rename(columns={col: "identity"}))
    return pd.concat([cov, *rollups], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the balanced, frozen segmented WinoQueer cohort.")
    ap.add_argument("--candidates_csv", type=Path,
                    default=Path("data/winoqueer/results/patching_candidates/winoqueer_patching_candidates_all.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/winoqueer/results/segmented"))
    ap.add_argument("--cap", type=int, default=100, help="Max pairs per Gender_ID_x x predicate_label cell.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.candidates_csv)
    print(f"Loaded {len(df)} candidate rows from {args.candidates_csv}")

    # row_id must be a usable stable key.
    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError("row_id has nulls or duplicates; it is the segmentation join key and must be unique.")

    mism = axis_mismatch_count(df)
    print(f"axis-mismatched rows: {mism} (expected 0)")
    df = annotate(df, strict=True)

    # Sort by bias_score desc, row_id desc as a deterministic tiebreak, then cap per cell.
    df = df.sort_values(["bias_score", "row_id"], ascending=False, kind="mergesort").reset_index(drop=True)
    kept = df.groupby(["Gender_ID_x", CELL_COL], sort=False, group_keys=False).head(args.cap)
    # Re-sort the survivors globally (highest bias first) and freeze the order.
    kept = kept.sort_values(["bias_score", "row_id"], ascending=False, kind="mergesort").reset_index(drop=True)
    kept.insert(0, "cohort_pair_id", range(len(kept)))

    keep_cols = ["cohort_pair_id", *RUN_SCRIPT_COLS, "axis", "identity", "is_umbrella"]
    keep_cols = [c for c in keep_cols if c in kept.columns]
    cohort = kept[keep_cols]
    cohort_path = args.out_dir / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    coverage = build_coverage(df, kept, args.cap)
    cov_path = args.out_dir / "cohort_coverage.csv"
    coverage.to_csv(cov_path, index=False)

    cells = coverage[coverage["level"] == "identity_x_predicate"]
    populated = cells[cells["n_available"] > 0]
    qualifying = cells[cells["qualifies"]]
    print(f"\nWrote {cohort_path}  ({len(cohort)} pairs)")
    print(f"Wrote {cov_path}")
    print(f"\nCohort: {len(cohort)} pairs | {cohort['identity'].nunique()} identities | "
          f"{cohort['axis'].nunique()} axes")
    print(f"identity x predicate cells: {len(populated)} populated, {len(qualifying)} qualify (n>={MIN_CELL}), "
          f"{int(cells['structural_zero'].sum())} structural zeros")
    print("\nPairs per axis:")
    print(cohort.groupby("axis").size().to_string())
    print("\nPairs per identity:")
    print(cohort.groupby("identity").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
