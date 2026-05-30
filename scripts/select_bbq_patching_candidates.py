#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def non_empty_string_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.len().gt(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select activation-patching candidates from TransformerLens BBQ scores."
    )
    parser.add_argument("--scored_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--context_condition",
        choices=["ambig", "disambig", "all"],
        default="ambig",
    )
    parser.add_argument("--min_biased_vs_unknown", type=float, default=0.0)
    parser.add_argument("--top_k_per_polarity", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scored_csv)

    required_cols = [
        "question_polarity",
        "context_condition",
        "biased_letters",
        "unknown_letters",
        "biased_vs_unknown_logp_diff",
        "p_biased",
        "p_unknown",
        "is_biased_prediction",
        "is_unknown_prediction",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mask = (
        non_empty_string_mask(df["biased_letters"])
        & non_empty_string_mask(df["unknown_letters"])
        & df["biased_vs_unknown_logp_diff"].notna()
        & (df["biased_vs_unknown_logp_diff"] > args.min_biased_vs_unknown)
    )

    if args.context_condition != "all":
        mask &= df["context_condition"] == args.context_condition

    candidates = df[mask].copy()
    candidates = candidates.sort_values(
        by=["biased_vs_unknown_logp_diff", "p_biased"],
        ascending=[False, False],
    )

    if args.top_k_per_polarity is not None:
        candidates = (
            candidates.groupby("question_polarity", group_keys=False, sort=False)
            .head(args.top_k_per_polarity)
            .copy()
        )
        candidates = candidates.sort_values(
            by=["biased_vs_unknown_logp_diff", "p_biased"],
            ascending=[False, False],
        )

    neg = candidates[candidates["question_polarity"] == "neg"].copy()
    nonneg = candidates[candidates["question_polarity"] == "nonneg"].copy()

    all_path = args.out_dir / "race_bbq_patching_candidates_all.csv"
    neg_path = args.out_dir / "race_bbq_patching_candidates_neg.csv"
    nonneg_path = args.out_dir / "race_bbq_patching_candidates_nonneg.csv"
    summary_path = args.out_dir / "race_bbq_patching_candidates_summary.csv"

    candidates.to_csv(all_path, index=False)
    neg.to_csv(neg_path, index=False)
    nonneg.to_csv(nonneg_path, index=False)

    summary = (
        candidates.groupby("question_polarity")
        .agg(
            n=("question_polarity", "count"),
            mean_p_biased=("p_biased", "mean"),
            mean_p_unknown=("p_unknown", "mean"),
            mean_biased_vs_unknown_logp_diff=("biased_vs_unknown_logp_diff", "mean"),
            biased_prediction_rate=("is_biased_prediction", "mean"),
            unknown_prediction_rate=("is_unknown_prediction", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print("\nWrote:")
    print(f"  {all_path}")
    print(f"  {neg_path}")
    print(f"  {nonneg_path}")
    print(f"  {summary_path}")

    print("\nSelection config:")
    print(f"  context_condition: {args.context_condition}")
    print(f"  min_biased_vs_unknown: {args.min_biased_vs_unknown}")
    print(f"  top_k_per_polarity: {args.top_k_per_polarity}")

    print("\nCounts:")
    print(f"  retained: {len(candidates)} / {len(df)}")
    print(f"  neg: {len(neg)}")
    print(f"  nonneg: {len(nonneg)}")

    print("\nSummary by polarity:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
