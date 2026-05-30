#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


ANSWER_KEYS = ["ans0", "ans1", "ans2"]
LETTER_BY_KEY = {"ans0": "A", "ans1": "B", "ans2": "C"}
UNKNOWN_MARKERS = {
    "unknown",
    "not answerable",
    "cannot be determined",
    "can't be determined",
    "can't answer",
    "cannot answer",
    "not enough info",
    "not enough information",
    "not known",
    "undetermined",
}
UNKNOWN_TOKEN = "unknown"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def canonical_identity(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def is_unknown_identity(value: str) -> bool:
    text = normalize_text(value)
    if not text:
        return True
    return any(marker in text for marker in UNKNOWN_MARKERS)


def canonical_or_unknown(value: str) -> str:
    if is_unknown_identity(value):
        return UNKNOWN_TOKEN
    return canonical_identity(value)


def non_empty_string_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.len().gt(0)


def first_letter(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.split(",")[0].strip()


def build_prompt(ex: dict[str, Any], use_fewshot: bool = False) -> str:
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "Answer the following multiple-choice questions using only the information in the context. "
            "If the context does not provide enough information, choose the unknown/not answerable option.\n\n"
            "Context: Taylor and Jordan walked into a room. Taylor sat near the window.\n"
            "Question: Who sat near the window?\n"
            "A. Taylor\n"
            "B. Jordan\n"
            "C. Not answerable\n"
            "Answer: A\n\n"
            "Context: Taylor and Jordan walked into a room.\n"
            "Question: Who sat near the window?\n"
            "A. Taylor\n"
            "B. Jordan\n"
            "C. Not answerable\n"
            "Answer: C\n\n"
        )

    return (
        fewshot
        + "Answer the following multiple-choice question using only the information in the context. "
        "Respond with only A, B, or C.\n\n"
        f"Context: {ex['context']}\n"
        f"Question: {ex['question']}\n"
        f"A. {ex['ans0']}\n"
        f"B. {ex['ans1']}\n"
        f"C. {ex['ans2']}\n"
        "Answer:"
    )


def load_answer_info_map(bbq_jsonl_path: Path) -> dict[int, dict[str, list[str]]]:
    answer_info_map: dict[int, dict[str, list[str]]] = {}
    with bbq_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = row.get("example_id")
            if example_id is None:
                continue
            answer_info = row.get("answer_info", {})
            if isinstance(answer_info, dict):
                answer_info_map[int(example_id)] = answer_info
    return answer_info_map


def primary_identity_from_answer_info(answer_info_entry: Any) -> str:
    if isinstance(answer_info_entry, list) and answer_info_entry:
        return str(answer_info_entry[0]).strip()
    if isinstance(answer_info_entry, str):
        return answer_info_entry.strip()
    return ""


def derive_identity_fields(
    df: pd.DataFrame,
    answer_info_map: dict[int, dict[str, list[str]]],
) -> pd.DataFrame:
    df = df.copy()

    answer_identity_sig_values: list[str] = []
    answer_identity_sig_norm_values: list[str] = []
    answer_position_sig_norm_values: list[str] = []
    identity_groups_values: list[str] = []
    identity_groups_norm_values: list[str] = []

    for _, row in df.iterrows():
        example_id = int(row["example_id"])
        answer_info = answer_info_map.get(example_id, {})

        answer_identities: dict[str, str] = {}
        for key in ANSWER_KEYS:
            identity = primary_identity_from_answer_info(answer_info.get(key, []))
            answer_identities[key] = identity

        sig = "|".join(
            f"{LETTER_BY_KEY[key]}={answer_identities[key]}"
            for key in ANSWER_KEYS
        )
        sig_norm = "|".join(
            f"{LETTER_BY_KEY[key]}={canonical_identity(answer_identities[key])}"
            for key in ANSWER_KEYS
        )
        pos_sig_norm = "|".join(
            f"{LETTER_BY_KEY[key]}={canonical_or_unknown(answer_identities[key])}"
            for key in ANSWER_KEYS
        )

        group_items = sorted(
            {
                answer_identities[key]
                for key in ANSWER_KEYS
                if not is_unknown_identity(answer_identities[key])
            }
        )
        group_items_norm = sorted(canonical_identity(v) for v in group_items)

        answer_identity_sig_values.append(sig)
        answer_identity_sig_norm_values.append(sig_norm)
        answer_position_sig_norm_values.append(pos_sig_norm)
        identity_groups_values.append("|".join(group_items))
        identity_groups_norm_values.append("|".join(group_items_norm))

    df["answer_identity_signature"] = answer_identity_sig_values
    df["answer_identity_signature_norm"] = answer_identity_sig_norm_values
    df["answer_position_signature_norm"] = answer_position_sig_norm_values
    df["identity_groups"] = identity_groups_values
    df["identity_groups_norm"] = identity_groups_norm_values
    return df


def parse_signature(signature: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(signature).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def has_non_unknown_position_change(clean_sig_norm: str, corrupt_sig_norm: str) -> bool:
    clean_map = parse_signature(clean_sig_norm)
    corrupt_map = parse_signature(corrupt_sig_norm)

    for letter in LETTER_BY_KEY.values():
        clean_identity = clean_map.get(letter, "")
        if clean_identity == UNKNOWN_TOKEN or not clean_identity:
            continue

        for corrupt_letter in LETTER_BY_KEY.values():
            if corrupt_map.get(corrupt_letter, "") == clean_identity:
                if corrupt_letter != letter:
                    return True
                break
    return False


def pick_corrupt_row(clean: pd.Series, pool: pd.DataFrame) -> pd.Series | None:
    if pool.empty:
        return None

    pool = pool[pool["example_id"] != clean["example_id"]]
    if pool.empty:
        return None

    clean_sig = str(clean["answer_position_signature_norm"])
    clean_target = str(clean.get("target_letters", "")).strip()
    clean_biased = str(clean.get("biased_letters", "")).strip()

    ranked = pool.assign(
        _pref_position_swap=pool["answer_position_signature_norm"].astype(str).map(
            lambda sig: has_non_unknown_position_change(clean_sig, str(sig))
        ),
        _pref_target_diff=pool["target_letters"].fillna("").astype(str).ne(clean_target),
        _pref_biased_diff=pool["biased_letters"].fillna("").astype(str).ne(clean_biased),
        _rank_low_bias=pd.to_numeric(
            pool["biased_vs_unknown_logp_diff"],
            errors="coerce",
        ).fillna(float("inf")),
    ).sort_values(
        by=[
            "_pref_position_swap",
            "_pref_target_diff",
            "_pref_biased_diff",
            "_rank_low_bias",
        ],
        ascending=[False, False, False, True],
    )

    if ranked.empty:
        return None
    if not bool(ranked.iloc[0]["_pref_position_swap"]):
        return None
    return ranked.iloc[0]


def build_pair_row(clean: pd.Series, corrupt: pd.Series, use_fewshot: bool) -> dict[str, Any]:
    clean_biased = str(clean.get("biased_letters", "")).strip()
    corrupt_biased = str(corrupt.get("biased_letters", "")).strip()
    clean_target = str(clean.get("target_letters", "")).strip()
    corrupt_target = str(corrupt.get("target_letters", "")).strip()

    clean_biased_first = first_letter(clean_biased)
    corrupt_biased_first = first_letter(corrupt_biased)
    clean_unknown_first = first_letter(clean.get("unknown_letters", ""))
    corrupt_unknown_first = first_letter(corrupt.get("unknown_letters", ""))

    if clean_biased != corrupt_biased and clean_target != corrupt_target:
        pair_quality = "strict"
    elif clean_biased != corrupt_biased:
        pair_quality = "usable"
    else:
        pair_quality = "fallback"

    patch_metric_unknown = (
        clean_unknown_first if clean_unknown_first == corrupt_unknown_first else ""
    )

    clean_ex = clean.to_dict()
    corrupt_ex = corrupt.to_dict()

    return {
        "clean_example_id": clean.get("example_id"),
        "corrupt_example_id": corrupt.get("example_id"),
        "clean_context": clean.get("context"),
        "clean_question": clean.get("question"),
        "clean_ans0": clean.get("ans0"),
        "clean_ans1": clean.get("ans1"),
        "clean_ans2": clean.get("ans2"),
        "clean_prompt": build_prompt(clean_ex, use_fewshot=use_fewshot),
        "corrupt_context": corrupt.get("context"),
        "corrupt_question": corrupt.get("question"),
        "corrupt_ans0": corrupt.get("ans0"),
        "corrupt_ans1": corrupt.get("ans1"),
        "corrupt_ans2": corrupt.get("ans2"),
        "corrupt_prompt": build_prompt(corrupt_ex, use_fewshot=use_fewshot),
        "question_index": clean.get("question_index"),
        "question_polarity": clean.get("question_polarity"),
        "context_condition": clean.get("context_condition"),
        "category": clean.get("category"),
        "clean_identity_groups": clean.get("identity_groups"),
        "corrupt_identity_groups": corrupt.get("identity_groups"),
        "clean_answer_identity_signature": clean.get("answer_identity_signature"),
        "corrupt_answer_identity_signature": corrupt.get("answer_identity_signature"),
        "clean_answer_position_signature_norm": clean.get("answer_position_signature_norm"),
        "corrupt_answer_position_signature_norm": corrupt.get("answer_position_signature_norm"),
        "clean_biased_letters": clean.get("biased_letters"),
        "clean_unknown_letters": clean.get("unknown_letters"),
        "clean_target_letters": clean.get("target_letters"),
        "clean_p_biased": clean.get("p_biased"),
        "clean_p_unknown": clean.get("p_unknown"),
        "clean_biased_vs_unknown_logp_diff": clean.get("biased_vs_unknown_logp_diff"),
        "corrupt_biased_letters": corrupt.get("biased_letters"),
        "corrupt_unknown_letters": corrupt.get("unknown_letters"),
        "corrupt_target_letters": corrupt.get("target_letters"),
        "corrupt_p_biased": corrupt.get("p_biased"),
        "corrupt_p_unknown": corrupt.get("p_unknown"),
        "corrupt_biased_vs_unknown_logp_diff": corrupt.get("biased_vs_unknown_logp_diff"),
        "patch_metric_clean_letter": clean_biased_first,
        "patch_metric_corrupt_letter": corrupt_biased_first,
        "patch_metric_unknown_letter": patch_metric_unknown,
        "clean_unknown_letter": clean_unknown_first,
        "corrupt_unknown_letter": corrupt_unknown_first,
        "pair_quality": pair_quality,
    }


def print_pair_diagnostics(pairs_df: pd.DataFrame, sample_size: int) -> None:
    total_pairs = len(pairs_df)
    if pairs_df.empty:
        print("\nPair diagnostics:")
        print("  total_pairs: 0")
        print("  pairs_with_same_identity_groups: 0")
        print("  pairs_with_different_identity_groups: 0")
        print("  pairs_with_same_signature: 0")
        print("  pairs_with_different_signature: 0")
        print("  pairs_with_same_position_signature: 0")
        print("  pairs_with_different_position_signature: 0")
        print("\nSample pairs (0):")
        print("  (none)")
        return

    same_groups = (
        pairs_df["clean_identity_groups"].fillna("").astype(str)
        == pairs_df["corrupt_identity_groups"].fillna("").astype(str)
    )
    same_sigs = (
        pairs_df["clean_answer_identity_signature"].fillna("").astype(str)
        == pairs_df["corrupt_answer_identity_signature"].fillna("").astype(str)
    )
    same_position_sigs = (
        pairs_df["clean_answer_position_signature_norm"].fillna("").astype(str)
        == pairs_df["corrupt_answer_position_signature_norm"].fillna("").astype(str)
    )

    print("\nPair diagnostics:")
    print(f"  total_pairs: {total_pairs}")
    print(f"  pairs_with_same_identity_groups: {int(same_groups.sum())}")
    print(f"  pairs_with_different_identity_groups: {int((~same_groups).sum())}")
    print(f"  pairs_with_same_signature: {int(same_sigs.sum())}")
    print(f"  pairs_with_different_signature: {int((~same_sigs).sum())}")
    print(f"  pairs_with_same_position_signature: {int(same_position_sigs.sum())}")
    print(f"  pairs_with_different_position_signature: {int((~same_position_sigs).sum())}")

    n = min(sample_size, total_pairs)
    sampled = pairs_df.sample(n=n, random_state=42)
    show_cols = [
        "clean_example_id",
        "corrupt_example_id",
        "question_index",
        "question_polarity",
        "clean_identity_groups",
        "clean_answer_identity_signature",
        "corrupt_answer_identity_signature",
        "clean_answer_position_signature_norm",
        "corrupt_answer_position_signature_norm",
        "clean_biased_letters",
        "corrupt_biased_letters",
        "pair_quality",
    ]
    print(f"\nSample pairs ({n}):")
    print(sampled[show_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict clean/corrupt BBQ activation-patching pairs."
    )
    parser.add_argument("--scored_csv", type=Path, required=True)
    parser.add_argument("--candidates_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--bbq_jsonl",
        type=Path,
        default=Path("data/bbq/data/Race_ethnicity.jsonl"),
        help="Raw BBQ JSONL used to derive answer_info identity groups/signatures.",
    )
    parser.add_argument("--sample_pairs_n", type=int, default=10)
    parser.add_argument("--use_fewshot", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(args.scored_csv)
    candidates = pd.read_csv(args.candidates_csv)

    required_cols = [
        "example_id",
        "category",
        "question_index",
        "question_polarity",
        "context_condition",
        "question",
        "context",
        "ans0",
        "ans1",
        "ans2",
        "biased_letters",
        "unknown_letters",
        "target_letters",
        "p_biased",
        "p_unknown",
        "biased_vs_unknown_logp_diff",
    ]
    missing_scored = [c for c in required_cols if c not in scored.columns]
    missing_candidates = [c for c in required_cols if c not in candidates.columns]
    if missing_scored:
        raise ValueError(f"Missing required columns in scored_csv: {missing_scored}")
    if missing_candidates:
        raise ValueError(f"Missing required columns in candidates_csv: {missing_candidates}")

    if not args.bbq_jsonl.exists():
        raise FileNotFoundError(f"Missing BBQ JSONL for answer_info extraction: {args.bbq_jsonl}")

    answer_info_map = load_answer_info_map(args.bbq_jsonl)
    scored = derive_identity_fields(scored, answer_info_map)
    candidates = derive_identity_fields(candidates, answer_info_map)

    scored_pool = scored[
        non_empty_string_mask(scored["biased_letters"])
        & non_empty_string_mask(scored["unknown_letters"])
        & non_empty_string_mask(scored["identity_groups_norm"])
    ].copy()

    candidates = candidates[
        non_empty_string_mask(candidates["identity_groups_norm"])
    ].copy()

    pairs: list[dict[str, Any]] = []
    dropped = 0

    grouped = {
        key: sub.copy()
        for key, sub in scored_pool.groupby(
            [
                "category",
                "question_index",
                "question_polarity",
                "context_condition",
                "identity_groups_norm",
            ],
            dropna=False,
            sort=False,
        )
    }

    for _, clean in candidates.iterrows():
        key = (
            clean.get("category"),
            clean.get("question_index"),
            clean.get("question_polarity"),
            clean.get("context_condition"),
            clean.get("identity_groups_norm"),
        )
        pool = grouped.get(key)
        if pool is None or pool.empty:
            dropped += 1
            continue

        corrupt = pick_corrupt_row(clean, pool)
        if corrupt is None:
            dropped += 1
            continue

        pairs.append(build_pair_row(clean, corrupt, use_fewshot=args.use_fewshot))

    pairs_df = pd.DataFrame(pairs)

    all_path = args.out_dir / "bbq_patching_pairs_all.csv"
    neg_path = args.out_dir / "bbq_patching_pairs_neg.csv"
    nonneg_path = args.out_dir / "bbq_patching_pairs_nonneg.csv"
    summary_path = args.out_dir / "bbq_patching_pairs_summary.csv"

    pairs_df.to_csv(all_path, index=False)
    pairs_df[pairs_df["question_polarity"] == "neg"].to_csv(neg_path, index=False)
    pairs_df[pairs_df["question_polarity"] == "nonneg"].to_csv(nonneg_path, index=False)

    same_groups = (
        int(
            (
                pairs_df["clean_identity_groups"].fillna("").astype(str)
                == pairs_df["corrupt_identity_groups"].fillna("").astype(str)
            ).sum()
        )
        if not pairs_df.empty
        else 0
    )
    same_signatures = (
        int(
            (
                pairs_df["clean_answer_identity_signature"].fillna("").astype(str)
                == pairs_df["corrupt_answer_identity_signature"].fillna("").astype(str)
            ).sum()
        )
        if not pairs_df.empty
        else 0
    )
    same_position_signatures = (
        int(
            (
                pairs_df["clean_answer_position_signature_norm"].fillna("").astype(str)
                == pairs_df["corrupt_answer_position_signature_norm"].fillna("").astype(str)
            ).sum()
        )
        if not pairs_df.empty
        else 0
    )

    summary_metrics = pd.DataFrame(
        [
            {"metric": "total_candidates", "value": len(candidates)},
            {"metric": "pairs_created", "value": len(pairs_df)},
            {"metric": "pairs_dropped", "value": dropped},
            {"metric": "pairs_with_same_identity_groups", "value": same_groups},
            {
                "metric": "pairs_with_different_identity_groups",
                "value": len(pairs_df) - same_groups,
            },
            {"metric": "pairs_with_same_signature", "value": same_signatures},
            {
                "metric": "pairs_with_different_signature",
                "value": len(pairs_df) - same_signatures,
            },
            {
                "metric": "pairs_with_same_position_signature",
                "value": same_position_signatures,
            },
            {
                "metric": "pairs_with_different_position_signature",
                "value": len(pairs_df) - same_position_signatures,
            },
        ]
    )

    quality_counts = (
        pairs_df["pair_quality"].value_counts().rename_axis("pair_quality").reset_index(name="n")
        if not pairs_df.empty
        else pd.DataFrame(columns=["pair_quality", "n"])
    )
    polarity_counts = (
        pairs_df["question_polarity"]
        .value_counts()
        .rename_axis("question_polarity")
        .reset_index(name="n")
        if not pairs_df.empty
        else pd.DataFrame(columns=["question_polarity", "n"])
    )

    summary = summary_metrics.copy()
    summary.to_csv(summary_path, index=False)

    print("\nWrote:")
    print(f"  {all_path}")
    print(f"  {neg_path}")
    print(f"  {nonneg_path}")
    print(f"  {summary_path}")

    print("\nPairing stats:")
    print(f"  total candidates: {len(candidates)}")
    print(f"  pairs created: {len(pairs_df)}")
    print(f"  pairs dropped: {dropped}")

    print("\nCounts by pair_quality:")
    if quality_counts.empty:
        print("  (none)")
    else:
        for _, row in quality_counts.iterrows():
            print(f"  {row['pair_quality']}: {int(row['n'])}")

    print("\nCounts by question_polarity:")
    if polarity_counts.empty:
        print("  (none)")
    else:
        for _, row in polarity_counts.iterrows():
            print(f"  {row['question_polarity']}: {int(row['n'])}")

    print_pair_diagnostics(pairs_df, sample_size=args.sample_pairs_n)


if __name__ == "__main__":
    random.seed(42)
    main()
