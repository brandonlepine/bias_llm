#!/usr/bin/env python3
"""Select WinoQueer rows for activation patching.

Each kept row is a queer/control pair sharing a stereotype continuation:
    queer_variant   = (sent_x, Gender_ID_x, prefix_x)
    control_variant = (sent_y, Gender_ID_y, prefix_y)
We keep rows where the continuation is cleanly extractable and the model already shows a
stereotype-consistent preference for the queer identity (bias_score >= --min_bias_score), so
patching has a real effect to localize.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.len() > 0


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_col, dropna=False):
        bs = g["bias_score"]
        rows.append({
            group_col: key,
            "n": int(len(g)),
            "mean_bias_score": round(float(bs.mean()), 6),
            "median_bias_score": round(float(bs.median()), 6),
            "min_bias_score": round(float(bs.min()), 6),
            "max_bias_score": round(float(bs.max()), 6),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select WinoQueer rows for activation patching.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_bias_score", type=float, default=0.5)
    parser.add_argument("--exclude_manual_review", action="store_true",
                        help="Drop rows where needs_extraction_review == True.")
    parser.add_argument("--label_filter", type=str, default=None,
                        help="Comma-separated predicate_label_provisional values to keep.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    total = len(df)

    required = ["continuation", "prefix_x", "prefix_y", "predicate", "bias_score",
                "Gender_ID_x", "Gender_ID_y", "predicate_label_provisional"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input_csv: {missing}")

    df["bias_score"] = pd.to_numeric(df["bias_score"], errors="coerce")

    mask = (
        nonempty(df["continuation"])
        & nonempty(df["prefix_x"])
        & nonempty(df["prefix_y"])
        & (df["predicate"].fillna("").astype(str).str.strip() != ".")
        & df["bias_score"].notna()
        & (df["bias_score"] >= args.min_bias_score)
    )
    if args.exclude_manual_review and "needs_extraction_review" in df.columns:
        mask &= ~as_bool(df["needs_extraction_review"])
    if args.label_filter:
        keep_labels = {s.strip() for s in args.label_filter.split(",") if s.strip()}
        mask &= df["predicate_label_provisional"].fillna("").astype(str).str.strip().isin(keep_labels)

    candidates = df[mask].copy().sort_values("bias_score", ascending=False).reset_index(drop=True)

    all_path = args.out_dir / "winoqueer_patching_candidates_all.csv"
    label_path = args.out_dir / "winoqueer_patching_candidates_by_label_summary.csv"
    identity_path = args.out_dir / "winoqueer_patching_candidates_by_identity_summary.csv"

    candidates.to_csv(all_path, index=False)

    if not candidates.empty:
        summarize(candidates, "predicate_label_provisional").to_csv(label_path, index=False)
        ident = pd.concat([
            summarize(candidates, "Gender_ID_x").rename(columns={"Gender_ID_x": "identity"}).assign(role="queer (Gender_ID_x)"),
            summarize(candidates, "Gender_ID_y").rename(columns={"Gender_ID_y": "identity"}).assign(role="control (Gender_ID_y)"),
        ], ignore_index=True)
        ident.to_csv(identity_path, index=False)
    else:
        pd.DataFrame().to_csv(label_path, index=False)
        pd.DataFrame().to_csv(identity_path, index=False)

    # ---- console report ----
    print(f"Rows retained: {len(candidates)} / {total}")
    print(f"min_bias_score: {args.min_bias_score} | exclude_manual_review: {args.exclude_manual_review}"
          + (f" | label_filter: {args.label_filter}" if args.label_filter else ""))
    print("\nWrote:")
    for p in (all_path, label_path, identity_path):
        print(f"  {p}")
    if candidates.empty:
        print("\nNo candidates matched the filters.")
        return
    print(f"\nmean bias_score (candidates): {candidates['bias_score'].mean():.6f}")
    print("\nCounts by predicate_label_provisional:")
    print(candidates["predicate_label_provisional"].fillna("(unlabeled)").replace("", "(unlabeled)").value_counts().to_string())

    print("\n=== Top 20 candidates by bias_score ===")
    cols = [c for c in ["bias_score", "Gender_ID_x", "Gender_ID_y", "prefix_x", "prefix_y",
                        "continuation", "predicate_label_provisional"] if c in candidates.columns]
    for _, r in candidates.head(20).iterrows():
        print(f"  bias={r['bias_score']:+.4f} [{r.get('predicate_label_provisional','')}] {r['Gender_ID_x']} vs {r['Gender_ID_y']}")
        print(f"     queer  : {r['prefix_x']!r} + {r['continuation']!r}")
        print(f"     control: {r['prefix_y']!r} + {r['continuation']!r}")


if __name__ == "__main__":
    main()
