#!/usr/bin/env python3
"""Merge Prodigy predicate labels back onto WinoQueer scored rows and flag predicates whose
extraction is likely too short or context-dependent to trust as a standalone stereotype.

The bias scoring treats `continuation`/`predicate` as the stereotype unit. Some predicates are
single words or carry meaning only with surrounding context (e.g. "gay", "normal", or
continuations with negations / "needs to be ...") — those should be reviewed before drawing
conclusions. This script attaches the human label and an automatic `needs_extraction_review`
flag, both per-row and aggregated per predicate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


# Single-word / context-dependent predicates that are unreliable on their own.
SHORT_AMBIGUOUS_PREDICATES = {"fixed", "gay", "happy", "religious", "women", "men", "straight", "normal"}
NEGATION_TERMS = ["not", "can't", "cannot", "does not", "do not", "never"]
BROADER_MEANING_PHRASES = [
    "needs to be", "called", "faking", "pretending", "condemned",
    "unhappy", "bad endings", "to be with",
]
UNCLEAR_LABEL = "UNCLEAR_OTHER"


def extract_label(record: dict[str, Any]) -> str:
    """Pull the provisional label out of a Prodigy task record, robust to recipe shape.

    Handles: an explicit `predicate_label`/`label` string, a `choice`/`accept` list, and
    `answer` of reject/ignore. Returns "" when unlabeled."""
    for key in ("predicate_label", "predicate_label_provisional", "label"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    accept = record.get("accept")
    if isinstance(accept, list) and accept:
        return "|".join(str(a) for a in accept)
    answer = record.get("answer")
    if isinstance(answer, str) and answer in ("reject", "ignore"):
        return answer.upper()
    return ""


def load_prodigy_labels(path: Path) -> dict[str, str]:
    """Map predicate -> provisional label from a Prodigy JSONL (last label wins)."""
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pred = rec.get("predicate")
            if pred is None:
                pred = rec.get("text")
            if pred is None:
                continue
            label = extract_label(rec)
            if label:
                labels[str(pred).strip()] = label
    return labels


def representative(series: pd.Series) -> Any:
    s = series.dropna()
    return s.value_counts().index[0] if not s.empty else ""


def token_count(predicate: str) -> int:
    return len(str(predicate).split())


def contains_any(text: str, needles: list[str]) -> list[str]:
    low = str(text).lower()
    return [n for n in needles if n in low]


def review_flags(predicate: str, continuation: str, label: str) -> tuple[bool, str]:
    reasons: list[str] = []
    if token_count(predicate) <= 1:
        reasons.append("predicate <= 1 token")
    if str(predicate).strip().lower() in SHORT_AMBIGUOUS_PREDICATES:
        reasons.append("predicate in short-ambiguous list")
    neg = contains_any(continuation, NEGATION_TERMS)
    if neg:
        reasons.append("continuation has negation: " + ", ".join(neg))
    broad = contains_any(continuation, BROADER_MEANING_PHRASES)
    if broad:
        reasons.append("continuation suggests broader meaning: " + ", ".join(broad))
    if str(label).strip() == UNCLEAR_LABEL:
        reasons.append(f"label == {UNCLEAR_LABEL}")
    return bool(reasons), "; ".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit WinoQueer predicate labels and extraction quality.")
    parser.add_argument("--scored_csv", type=Path, required=True)
    parser.add_argument("--prodigy_jsonl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(args.scored_csv)
    required = ["predicate", "continuation", "bias_score", "sent_x", "sent_y"]
    missing = [c for c in required if c not in scored.columns]
    if missing:
        raise ValueError(f"Missing required columns in scored_csv: {missing}")
    scored = scored.copy()
    scored["predicate"] = scored["predicate"].fillna("").astype(str)
    scored["continuation"] = scored["continuation"].fillna("").astype(str)

    labels = load_prodigy_labels(args.prodigy_jsonl)
    print(f"Loaded {len(labels)} labeled predicates from {args.prodigy_jsonl}")

    # Per-predicate metadata, label, and review flag (predicate + continuation determine it).
    per_pred: dict[str, dict[str, Any]] = {}
    scored["bias_score"] = pd.to_numeric(scored["bias_score"], errors="coerce")
    for predicate, g in scored.groupby("predicate", sort=False):
        if not str(predicate).strip():
            continue
        continuation = representative(g["continuation"])
        label = labels.get(str(predicate).strip(), "")
        flag, reason = review_flags(predicate, continuation, label)
        valid_bs = g["bias_score"].dropna()
        per_pred[predicate] = {
            "predicate": predicate,
            "continuation": continuation,
            "predicate_label_provisional": label,
            "needs_extraction_review": flag,
            "extraction_review_reason": reason,
            "n_rows": int(len(g)),
            "mean_bias_score": round(float(valid_bs.mean()), 6) if len(valid_bs) else float("nan"),
            "positive_bias_fraction": round(float((valid_bs > 0).mean()), 6) if len(valid_bs) else float("nan"),
            "example_sent_x": str(representative(g["sent_x"])),
            "example_sent_y": str(representative(g["sent_y"])),
        }

    # ---- Output 1: per-row scored with labels ----
    map_label = {k: v["predicate_label_provisional"] for k, v in per_pred.items()}
    map_flag = {k: v["needs_extraction_review"] for k, v in per_pred.items()}
    map_reason = {k: v["extraction_review_reason"] for k, v in per_pred.items()}
    scored_out = scored.copy()
    scored_out["predicate_label_provisional"] = scored_out["predicate"].map(map_label).fillna("")
    scored_out["predicate_label_source"] = "prodigy"
    scored_out["needs_extraction_review"] = scored_out["predicate"].map(map_flag).fillna(False)
    scored_out["extraction_review_reason"] = scored_out["predicate"].map(map_reason).fillna("")
    rows_path = args.out_dir / "winoqueer_scored_with_predicate_labels.csv"
    scored_out.to_csv(rows_path, index=False)

    # ---- Output 2: per-predicate audit ----
    audit_cols = [
        "predicate", "continuation", "example_sent_x", "example_sent_y",
        "predicate_label_provisional", "n_rows", "mean_bias_score", "positive_bias_fraction",
        "needs_extraction_review", "extraction_review_reason",
    ]
    audit = pd.DataFrame(list(per_pred.values()))[audit_cols]
    audit["_abs_mean"] = audit["mean_bias_score"].abs()
    audit = audit.sort_values(
        ["needs_extraction_review", "n_rows", "_abs_mean"], ascending=[False, False, False]
    ).drop(columns="_abs_mean").reset_index(drop=True)
    audit_path = args.out_dir / "winoqueer_predicate_label_audit.csv"
    audit.to_csv(audit_path, index=False)

    # ---- Output 3: summary counts ----
    summary_rows: list[dict[str, Any]] = []
    for label, count in audit["predicate_label_provisional"].fillna("").replace("", "(unlabeled)").value_counts().items():
        summary_rows.append({"group": "predicate_label_provisional", "key": label, "n_predicates": int(count)})
    for flag, count in audit["needs_extraction_review"].value_counts().items():
        summary_rows.append({"group": "needs_extraction_review", "key": bool(flag), "n_predicates": int(count)})
    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "winoqueer_predicate_label_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nWrote:")
    for p in (rows_path, audit_path, summary_path):
        print(f"  {p}")
    print(f"\nPredicates: {len(audit)} | needs_extraction_review: {int(audit['needs_extraction_review'].sum())}")
    print("\nCounts by predicate_label_provisional:")
    print(audit["predicate_label_provisional"].replace("", "(unlabeled)").value_counts().to_string())
    print("\nCounts by needs_extraction_review:")
    print(audit["needs_extraction_review"].value_counts().to_string())


if __name__ == "__main__":
    main()
