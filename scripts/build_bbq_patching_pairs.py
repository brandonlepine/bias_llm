#!/usr/bin/env python3
"""Build token-aligned minimal-swap clean/corrupt pairs for BBQ activation patching.

Method (token-aligned minimal counterfactual)
---------------------------------------------
For each high-bias candidate example we synthesize the *corrupt* prompt from the
*same* example by swapping the two named identity surface forms in place (in both
the context and the answer options), keeping the unknown option, answer order, and
every other token identical. Because the only thing that changes is the identity,
the clean and corrupt prompts are token-for-token aligned except at the identity
token positions themselves.

This is the standard prerequisite for causal-tracing/activation patching: position
`i` in the clean run must correspond to position `i` in the corrupt run. The
previous approach paired two *different* BBQ examples with deliberately swapped
answer positions, which destroyed token alignment (empirically 0/1032 pairs were
fully aligned) and made the per-token-position heatmap axis uninterpretable.

Alignment gate
--------------
A swap only preserves token alignment if the two surface forms tokenize to the same
number of tokens *as they appear in context* (i.e. with a leading space). We require
``len(tok(" " + a)) == len(tok(" " + b))`` and then re-tokenize both full prompts and
verify that (a) they have equal length and (b) every differing position decodes to an
identity token. Pairs that fail either check are dropped and counted.

Metric letters (fixed readout)
------------------------------
Both clean and corrupt are read out at the *same* answer letters:
  - biased letter  = the slot the model is biased toward in the clean example
  - unknown letter = the unknown slot (unchanged by the swap)
In the clean prompt the stereotyped identity occupies the biased slot; in the corrupt
prompt the swap moves it out of that slot, so the corrupt baseline is de-biased. The
patch then tests where the "prefer the stereotyped slot" signal lives.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


ANSWER_KEYS = ["ans0", "ans1", "ans2"]
LETTER_BY_KEY = {"ans0": "A", "ans1": "B", "ans2": "C"}
KEY_BY_LETTER = {v: k for k, v in LETTER_BY_KEY.items()}
LETTERS = ["A", "B", "C"]
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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def is_unknown_identity(value: str) -> bool:
    text = normalize_text(value)
    if not text:
        return True
    return any(marker in text for marker in UNKNOWN_MARKERS)


def first_letter(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.split(",")[0].strip()


def build_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    return (
        "Answer the following multiple-choice question using only the information in the context. "
        "Respond with only A, B, or C.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"A. {ans0}\n"
        f"B. {ans1}\n"
        f"C. {ans2}\n"
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


def surface_and_group(answer_info_entry: Any) -> tuple[str, str]:
    """answer_info[ansX] is typically [surface_form, group_label]."""
    if isinstance(answer_info_entry, list) and answer_info_entry:
        surface = str(answer_info_entry[0]).strip()
        group = str(answer_info_entry[1]).strip() if len(answer_info_entry) > 1 else ""
        return surface, group
    if isinstance(answer_info_entry, str):
        return answer_info_entry.strip(), ""
    return "", ""


def swap_surfaces(text: str, a: str, b: str) -> str:
    """Simultaneously swap whole-word occurrences of surface `a` and surface `b`."""
    if not a or not b:
        return text
    pattern = re.compile(r"\b(" + re.escape(a) + r"|" + re.escape(b) + r")\b")
    return pattern.sub(lambda m: b if m.group(0) == a else a, text)


def space_token_len(tokenizer, surface: str) -> int:
    return len(tokenizer(" " + surface, add_special_tokens=False)["input_ids"])


def surface_token_ids(tokenizer, *surfaces: str) -> set[int]:
    ids: set[int] = set()
    for surface in surfaces:
        if not surface:
            continue
        ids.update(tokenizer(" " + surface, add_special_tokens=False)["input_ids"])
        ids.update(tokenizer(surface, add_special_tokens=False)["input_ids"])
    return ids


def verify_alignment(
    tokenizer, clean_prompt: str, corrupt_prompt: str, allowed_token_ids: set[int]
) -> tuple[bool, int]:
    """Return (aligned, n_diff_tokens). Aligned iff equal length and every differing
    position decodes to an identity token in both prompts."""
    clean_ids = tokenizer(clean_prompt, add_special_tokens=True)["input_ids"]
    corrupt_ids = tokenizer(corrupt_prompt, add_special_tokens=True)["input_ids"]
    if len(clean_ids) != len(corrupt_ids):
        return False, -1
    diff_positions = [i for i in range(len(clean_ids)) if clean_ids[i] != corrupt_ids[i]]
    for i in diff_positions:
        if clean_ids[i] not in allowed_token_ids or corrupt_ids[i] not in allowed_token_ids:
            return False, len(diff_positions)
    return True, len(diff_positions)


def build_pair_row(
    clean: pd.Series,
    answer_info: dict[str, list[str]],
    tokenizer,
) -> tuple[dict[str, Any] | None, str]:
    """Return (pair_row, status). status == 'ok' when a pair was produced, else a
    drop reason for diagnostics."""
    example_id = int(clean["example_id"])

    # Resolve surface forms / groups per answer slot.
    surfaces: dict[str, str] = {}
    groups: dict[str, str] = {}
    for key in ANSWER_KEYS:
        surface, group = surface_and_group(answer_info.get(key, []))
        letter = LETTER_BY_KEY[key]
        surfaces[letter] = surface
        groups[letter] = group

    named = [(letter, surfaces[letter]) for letter in LETTERS if surfaces[letter] and not is_unknown_identity(surfaces[letter])]
    if len(named) != 2:
        return None, "not_two_named_identities"

    biased_letter = first_letter(clean.get("biased_letters", ""))
    unknown_letter = first_letter(clean.get("unknown_letters", ""))
    if biased_letter not in LETTERS or unknown_letter not in LETTERS:
        return None, "bad_metric_letters"
    if biased_letter == unknown_letter:
        return None, "biased_equals_unknown"

    named_letters = {letter for letter, _ in named}
    if biased_letter not in named_letters:
        return None, "biased_letter_not_named"
    if unknown_letter in named_letters:
        return None, "unknown_letter_is_named"

    (letter_1, v1), (letter_2, v2) = named
    if normalize_text(v1) == normalize_text(v2):
        return None, "identical_surfaces"

    # Token-alignment gate: the two surfaces must occupy the same number of tokens.
    if space_token_len(tokenizer, v1) != space_token_len(tokenizer, v2):
        return None, "surface_token_len_mismatch"

    # Identity that sits in the biased slot of the CLEAN prompt (the stereotyped one).
    biased_identity = surfaces[biased_letter]
    biased_group = groups[biased_letter]
    other_letter = (named_letters - {biased_letter}).pop()
    other_identity = surfaces[other_letter]
    other_group = groups[other_letter]

    clean_context = str(clean.get("context", ""))
    question = str(clean.get("question", ""))
    clean_ans = {letter: str(clean.get(KEY_BY_LETTER[letter], "")) for letter in LETTERS}

    corrupt_context = swap_surfaces(clean_context, v1, v2)
    corrupt_ans = {letter: swap_surfaces(clean_ans[letter], v1, v2) for letter in LETTERS}

    clean_prompt = build_prompt(clean_context, question, clean_ans["A"], clean_ans["B"], clean_ans["C"])
    corrupt_prompt = build_prompt(corrupt_context, question, corrupt_ans["A"], corrupt_ans["B"], corrupt_ans["C"])

    allowed = surface_token_ids(tokenizer, v1, v2)
    aligned, n_diff = verify_alignment(tokenizer, clean_prompt, corrupt_prompt, allowed)
    if not aligned:
        return None, "alignment_verify_failed"
    if n_diff == 0:
        return None, "no_diff_tokens"

    row = {
        "clean_example_id": example_id,
        # Corrupt is synthesized from the same source example.
        "corrupt_example_id": example_id,
        "question_index": clean.get("question_index"),
        "question_polarity": clean.get("question_polarity"),
        "context_condition": clean.get("context_condition"),
        "category": clean.get("category"),
        "clean_context": clean_context,
        "corrupt_context": corrupt_context,
        "clean_question": question,
        "corrupt_question": question,
        "clean_ans0": clean_ans["A"],
        "clean_ans1": clean_ans["B"],
        "clean_ans2": clean_ans["C"],
        "corrupt_ans0": corrupt_ans["A"],
        "corrupt_ans1": corrupt_ans["B"],
        "corrupt_ans2": corrupt_ans["C"],
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        # Fixed readout: clean and corrupt are scored at the SAME letters.
        "clean_biased_letters": biased_letter,
        "corrupt_biased_letters": biased_letter,
        "clean_unknown_letters": unknown_letter,
        "corrupt_unknown_letters": unknown_letter,
        "clean_unknown_letter": unknown_letter,
        "corrupt_unknown_letter": unknown_letter,
        "clean_target_letters": clean.get("target_letters"),
        "corrupt_target_letters": clean.get("target_letters"),
        # Identity metadata.
        "biased_identity": biased_identity,
        "biased_identity_group": biased_group,
        "other_identity": other_identity,
        "other_identity_group": other_group,
        "swap_identities": f"{v1}<->{v2}",
        # Carry-through scoring signal for the clean example.
        "clean_p_biased": clean.get("p_biased"),
        "clean_p_unknown": clean.get("p_unknown"),
        "clean_biased_vs_unknown_logp_diff": clean.get("biased_vs_unknown_logp_diff"),
        # Alignment diagnostics.
        "n_diff_tokens": n_diff,
        "prompt_token_len": len(tokenizer(clean_prompt, add_special_tokens=True)["input_ids"]),
        "pair_type": "minimal_swap",
        "pair_quality": "strict",
    }
    return row, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build token-aligned minimal-swap BBQ activation-patching pairs."
    )
    parser.add_argument("--candidates_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--bbq_jsonl",
        type=Path,
        default=Path("data/bbq/data/Race_ethnicity.jsonl"),
        help="Raw BBQ JSONL used to derive answer_info identity surface forms.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model/tokenizer used to enforce the token-alignment gate. Must match the patching model.",
    )
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--sample_pairs_n", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    ]
    missing = [c for c in required_cols if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing required columns in candidates_csv: {missing}")

    if not args.bbq_jsonl.exists():
        raise FileNotFoundError(f"Missing BBQ JSONL for answer_info extraction: {args.bbq_jsonl}")

    print(f"Loading tokenizer for alignment gate: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    answer_info_map = load_answer_info_map(args.bbq_jsonl)

    pairs: list[dict[str, Any]] = []
    drop_reasons: dict[str, int] = {}
    for _, clean in candidates.iterrows():
        answer_info = answer_info_map.get(int(clean["example_id"]), {})
        row, status = build_pair_row(clean, answer_info, tokenizer)
        if row is None:
            drop_reasons[status] = drop_reasons.get(status, 0) + 1
            continue
        pairs.append(row)
        if args.max_pairs is not None and len(pairs) >= args.max_pairs:
            break

    pairs_df = pd.DataFrame(pairs)

    all_path = args.out_dir / "bbq_patching_pairs_all.csv"
    neg_path = args.out_dir / "bbq_patching_pairs_neg.csv"
    nonneg_path = args.out_dir / "bbq_patching_pairs_nonneg.csv"
    summary_path = args.out_dir / "bbq_patching_pairs_summary.csv"

    pairs_df.to_csv(all_path, index=False)
    if not pairs_df.empty:
        pairs_df[pairs_df["question_polarity"] == "neg"].to_csv(neg_path, index=False)
        pairs_df[pairs_df["question_polarity"] == "nonneg"].to_csv(nonneg_path, index=False)
    else:
        pairs_df.to_csv(neg_path, index=False)
        pairs_df.to_csv(nonneg_path, index=False)

    summary_rows = [
        {"metric": "total_candidates", "value": len(candidates)},
        {"metric": "pairs_created", "value": len(pairs_df)},
        {"metric": "pairs_dropped", "value": int(len(candidates) - len(pairs_df))},
    ]
    for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
        summary_rows.append({"metric": f"dropped_{reason}", "value": count})
    if not pairs_df.empty:
        summary_rows.append({"metric": "mean_diff_tokens", "value": round(float(pairs_df["n_diff_tokens"].mean()), 3)})
        summary_rows.append({"metric": "neg_pairs", "value": int((pairs_df["question_polarity"] == "neg").sum())})
        summary_rows.append({"metric": "nonneg_pairs", "value": int((pairs_df["question_polarity"] == "nonneg").sum())})
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nWrote:")
    for p in (all_path, neg_path, nonneg_path, summary_path):
        print(f"  {p}")

    print("\nPairing stats:")
    print(f"  total candidates: {len(candidates)}")
    print(f"  pairs created:    {len(pairs_df)}")
    print(f"  pairs dropped:    {len(candidates) - len(pairs_df)}")
    print("\nDrop reasons:")
    if drop_reasons:
        for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")
    else:
        print("  (none)")

    if not pairs_df.empty:
        print("\nCounts by question_polarity:")
        for polarity, count in pairs_df["question_polarity"].value_counts().items():
            print(f"  {polarity}: {int(count)}")
        print(f"\nmean differing tokens per pair: {pairs_df['n_diff_tokens'].mean():.2f}")

        n = min(args.sample_pairs_n, len(pairs_df))
        sampled = pairs_df.sample(n=n, random_state=42)
        show_cols = [
            "clean_example_id",
            "question_polarity",
            "swap_identities",
            "clean_biased_letters",
            "clean_unknown_letters",
            "n_diff_tokens",
            "prompt_token_len",
        ]
        print(f"\nSample pairs ({n}):")
        print(sampled[show_cols].to_string(index=False))


if __name__ == "__main__":
    random.seed(42)
    main()
