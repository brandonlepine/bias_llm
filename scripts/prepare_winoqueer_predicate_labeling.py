#!/usr/bin/env python3
"""Aggregate unique WinoQueer predicates from the continuation-scoring output for manual
labeling in Prodigy.

One row per unique predicate, with bias-score summary stats and two example sentences, so a
human can label each predicate (e.g. harmful / neutral / off-topic) once instead of per row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def representative(series: pd.Series) -> Any:
    """Most common value in a group (first wins on ties); '' if empty."""
    s = series.dropna()
    if s.empty:
        return ""
    return s.value_counts().index[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unique WinoQueer predicates for Prodigy labeling.")
    parser.add_argument("--scored_csv", type=Path, required=True)
    parser.add_argument("--out_jsonl", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument(
        "--keep_nonalpha",
        action="store_true",
        help="Keep predicates with no letters (e.g. '.', from identity-final sentences). "
        "By default these degenerate predicates are dropped.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.scored_csv)
    required = ["predicate", "continuation", "bias_score", "sent_x", "sent_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scored_csv: {missing} (found {list(df.columns)})")

    # Keep rows with a real predicate and a scored bias value.
    df = df.copy()
    df["predicate"] = df["predicate"].fillna("").astype(str)
    df = df[df["predicate"].str.strip().str.len() > 0]
    if not args.keep_nonalpha:
        n_before = len(df)
        df = df[df["predicate"].str.contains(r"[A-Za-z]", regex=True)]
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with non-alphabetic predicates (e.g. '.'); pass --keep_nonalpha to retain.")
    df["bias_score"] = pd.to_numeric(df["bias_score"], errors="coerce")
    df = df.dropna(subset=["bias_score"])
    if df.empty:
        raise ValueError("No rows with a non-empty predicate and a valid bias_score were found.")

    records: list[dict[str, Any]] = []
    for predicate, g in df.groupby("predicate", sort=False):
        bs = g["bias_score"]
        records.append({
            "predicate": predicate,
            "continuation": representative(g["continuation"]),
            "n_rows": int(len(g)),
            "mean_bias_score": round(float(bs.mean()), 6),
            "median_bias_score": round(float(bs.median()), 6),
            "min_bias_score": round(float(bs.min()), 6),
            "max_bias_score": round(float(bs.max()), 6),
            "positive_bias_fraction": round(float((bs > 0).mean()), 6),
            "example_sent_x": str(representative(g["sent_x"])),
            "example_sent_y": str(representative(g["sent_y"])),
        })

    out = pd.DataFrame(records)
    out["_abs_mean"] = out["mean_bias_score"].abs()
    out = out.sort_values(["n_rows", "_abs_mean"], ascending=[False, False]).drop(columns="_abs_mean").reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for r in out.to_dict(orient="records"):
            task = {
                "text": r["predicate"],
                "predicate": r["predicate"],
                "continuation": r["continuation"],
                "meta": {
                    "n_rows": r["n_rows"],
                    "mean_bias_score": r["mean_bias_score"],
                    "positive_bias_fraction": r["positive_bias_fraction"],
                    "example_sent_x": r["example_sent_x"],
                    "example_sent_y": r["example_sent_y"],
                },
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Unique predicates: {len(out)} (from {len(df)} scored rows)")
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_csv}")
    print("\nTop 10 predicates by n_rows then |mean_bias_score|:")
    print(out.head(10)[["predicate", "n_rows", "mean_bias_score", "positive_bias_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
