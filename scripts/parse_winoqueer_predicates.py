#!/usr/bin/env python3
"""Parse WinoQueer paired sentences into autoregressive scoring units.

For each (sent_x, sent_y) pair that differs only by an identity phrase, this produces a
clean prefix -> continuation split where the prefix ends immediately before the harmful
predicate, so the continuation (the predicate) can be log-likelihood scored under each
identity:  log P(continuation | prefix_x)  vs  log P(continuation | prefix_y).

Parsing is fully deterministic (no LLM). The goal is not perfect parsing but a CSV that
is easy to eyeball in a spreadsheet, with explicit `needs_manual_review` flags for the
rows where the deterministic split is uncertain.

WinoQueer is mostly NOT the canonical "<IDENTITY> people are <predicate>" form. The
dominant families are identity-embedded, e.g.:
    "James is LGBTQ and abnormal"              (identity mid, predicate trailing)
    "most people who are LGBTQ are abnormal."  (identity mid, predicate trailing)
    "LGBTQ people are abnormal."               (identity-first, canonical)
In every family the predicate is the trailing phrase after the last linking verb that
follows the identity, so that single rule keeps the identity inside the prefix and the
predicate shared between x and y.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


PLACEHOLDER = "<IDENTITY>"

# Verb connectives (longest first so multi-word phrases match before their substrings).
VERB_CONNECTIVES = ["can be", "could be", "would be", "should be", "must be", "tend to be",
                    "are", "is", "have", "has", "had", "were", "was", "be", "being"]
# Non-verb connectives used only when no verb connective follows the identity.
OTHER_CONNECTIVES = ["and", "who are", "that are", "as"]
# Subject nouns: for "<IDENTITY> people <verb phrase>" with no linking verb (Pattern D).
SUBJECT_NOUNS = ["people", "individuals", "persons", "folks", "person", "folk", "men", "women"]

# Canonical-form (identity-first) patterns, only used to give clean parse_status_detail
# labels. The actual split is computed by `split_predicate` for all rows uniformly.
CANONICAL_PATTERNS = [
    ("pattern_A", re.compile(r"^" + re.escape(PLACEHOLDER) + r"\s+people\s+are\b", re.I)),
    ("pattern_B", re.compile(r"^" + re.escape(PLACEHOLDER) + r"\s+people\s+can be\b", re.I)),
    ("pattern_C", re.compile(r"^" + re.escape(PLACEHOLDER) + r"\s+people\s+(?:have|has)\b", re.I)),
    ("pattern_E", re.compile(r"^" + re.escape(PLACEHOLDER) + r"\s+individuals\s+are\b", re.I)),
    ("pattern_F", re.compile(r"^" + re.escape(PLACEHOLDER) + r"\s+persons\s+are\b", re.I)),
    ("pattern_D", re.compile(r"^" + re.escape(PLACEHOLDER) + r"\s+people\b", re.I)),
]


def first_index(haystack: str, needle: str) -> int:
    """Index of the first occurrence of `needle`, case-sensitive then case-insensitive."""
    if not needle:
        return -1
    i = haystack.find(needle)
    if i >= 0:
        return i
    return haystack.lower().find(needle.lower())


def replace_first(haystack: str, needle: str, repl: str, case_insensitive: bool = False) -> tuple[str, int]:
    """Replace the first occurrence of `needle`; return (new_string, index_of_match)."""
    if case_insensitive:
        i = haystack.lower().find(needle.lower())
    else:
        i = haystack.find(needle)
    if i < 0:
        return haystack, -1
    return haystack[:i] + repl + haystack[i + len(needle):], i


def longest_common_suffix(a: str, b: str) -> str:
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return a[len(a) - n:] if n else ""


def build_template(sent_x: str, sent_y: str, id_x: str, id_y: str) -> tuple[str, str, str]:
    """Return (template_text, parse_status, review_note).

    template_text carries the <IDENTITY> placeholder where the identity sits.
    """
    tx, ix = replace_first(sent_x, id_x, PLACEHOLDER)
    ty, iy = replace_first(sent_y, id_y, PLACEHOLDER)
    if ix >= 0 and iy >= 0 and tx == ty:
        return tx, "template_exact", ""

    tx_ci, ix_ci = replace_first(sent_x, id_x, PLACEHOLDER, case_insensitive=True)
    ty_ci, iy_ci = replace_first(sent_y, id_y, PLACEHOLDER, case_insensitive=True)
    if ix_ci >= 0 and iy_ci >= 0 and tx_ci == ty_ci:
        return tx_ci, "template_case_insensitive", ""

    # Fall back to the longest common suffix shared by both sentences (the part after the
    # identities diverge). We still place an <IDENTITY> marker before it for downstream
    # splitting, but the prefixes are recovered per-sentence later.
    suffix = longest_common_suffix(sent_x, sent_y)
    if suffix.strip():
        return PLACEHOLDER + suffix, "common_suffix", "templates did not match after identity replacement"

    return "", "failed", "could not align sent_x and sent_y by identity"


def _first_after(template: str, phrases: list[str], after: int) -> int:
    """End index of the earliest whole-word phrase from `phrases` starting at/after `after`."""
    best_start, best_end = None, -1
    for phrase in phrases:
        m = re.search(r"\b" + re.escape(phrase) + r"\b", template[after:])
        if m and (best_start is None or after + m.start() < best_start):
            best_start, best_end = after + m.start(), after + m.end()
    return best_end


def find_split_index(template: str) -> tuple[int, str]:
    """Return (char index where the continuation begins, method label).

    The continuation begins at the FIRST connective that occurs after the identity, so the
    identity always stays inside the prefix and the predicate is the trailing shared phrase:
      1. first verb connective after identity ("are"/"is"/"can be"/"have"/...),
      2. else first non-verb connective ("and"/"who are"/...),
      3. else (no connective) split after a subject noun ("<ID> people | <verb phrase>"),
      4. else content-word fallback.
    """
    id_pos = template.find(PLACEHOLDER)
    after = id_pos + len(PLACEHOLDER) if id_pos >= 0 else 0

    end = _first_after(template, VERB_CONNECTIVES, after)
    if end >= 0:
        return end, "linking_verb_split"
    end = _first_after(template, OTHER_CONNECTIVES, after)
    if end >= 0:
        return end, "connective_split"
    m = re.match(r"\s*(" + "|".join(SUBJECT_NOUNS) + r")\b", template[after:], re.I)
    if m:
        return after + m.end(), "subject_noun_split"
    m = re.search(r"\S", template[after:])
    if m:
        return after + m.start(), "fallback_predicate"
    return len(template), "fallback_predicate"


def split_predicate(template_text: str) -> dict[str, Any]:
    """Split a template into prefix_template / predicate / continuation.

    `continuation` is the literal suffix of the template from the split point on (it sits
    after the identity, so it is byte-identical to the tail of both sent_x and sent_y) —
    this guarantees prefix + continuation reconstructs the sentence exactly.
    """
    split_idx, method = find_split_index(template_text)
    continuation = template_text[split_idx:]
    predicate = continuation.strip()

    # A predicate with no letters (e.g. split landed on a trailing "are." -> ".") is
    # degenerate: re-split before the first content word after the identity and flag.
    if not re.search(r"[A-Za-z]", predicate):
        id_pos = template_text.find(PLACEHOLDER)
        after = id_pos + len(PLACEHOLDER) if id_pos >= 0 else 0
        m = re.search(r"\S", template_text[after:])
        if m:
            split_idx = after + m.start()
            continuation = template_text[split_idx:]
            predicate = continuation.strip()
        method = "fallback_predicate"

    prefix_template = template_text[:split_idx]

    # Nicer label for the canonical identity-first forms (unless we fell back).
    if method != "fallback_predicate":
        for label, pat in CANONICAL_PATTERNS:
            if pat.match(template_text):
                method = label
                break

    return {
        "prefix_template": prefix_template,
        "predicate": predicate,
        "continuation": continuation,
        "method": method,
    }


def parse_row(row: pd.Series) -> dict[str, Any]:
    sent_x = "" if pd.isna(row["sent_x"]) else str(row["sent_x"])
    sent_y = "" if pd.isna(row["sent_y"]) else str(row["sent_y"])
    id_x = "" if pd.isna(row["Gender_ID_x"]) else str(row["Gender_ID_x"])
    id_y = "" if pd.isna(row["Gender_ID_y"]) else str(row["Gender_ID_y"])

    template_text, parse_status, status_note = build_template(sent_x, sent_y, id_x, id_y)

    if parse_status == "failed" or not template_text:
        out = _empty_parse(row, sent_x, sent_y, id_x, id_y)
        out["parse_status"] = "failed"
        out["parse_status_detail"] = "failed"
        out["needs_manual_review"] = True
        out["review_reason"] = status_note or "parse failed"
        return out

    split = split_predicate(template_text)
    cont = split["continuation"]
    pred = split["predicate"]
    detail = split["method"]

    # `cont` is the post-identity shared suffix, so it is a literal suffix of both sentences.
    prefix_x = sent_x[: len(sent_x) - len(cont)] if cont and sent_x.endswith(cont) else ""
    prefix_y = sent_y[: len(sent_y) - len(cont)] if cont and sent_y.endswith(cont) else ""

    # Identity char spans in the original sentences.
    ix = first_index(sent_x, id_x)
    iy = first_index(sent_y, id_y)
    id_x_start, id_x_end = (ix, ix + len(id_x)) if ix >= 0 else (-1, -1)
    id_y_start, id_y_end = (iy, iy + len(id_y)) if iy >= 0 else (-1, -1)

    # Predicate char spans (predicate = continuation without surrounding whitespace).
    lead = len(cont) - len(cont.lstrip())
    pred_x_start = (len(prefix_x) + lead) if (prefix_x and pred) else -1
    pred_x_end = (pred_x_start + len(pred)) if pred_x_start >= 0 else -1
    pred_y_start = (len(prefix_y) + lead) if (prefix_y and pred) else -1
    pred_y_end = (pred_y_start + len(pred)) if pred_y_start >= 0 else -1

    # Manual-review logic.
    reasons: list[str] = []
    if "fallback" in detail:
        reasons.append("fallback predicate split")
    if not pred:
        reasons.append("empty predicate")
    if not cont:
        reasons.append("empty continuation")
    if not prefix_x:
        reasons.append("prefix_x not recoverable from sent_x")
    if not prefix_y:
        reasons.append("prefix_y not recoverable from sent_y")
    if parse_status == "common_suffix":
        reasons.append("aligned by common suffix (not exact template)")

    return {
        "row_id": row.get("row_id"),
        "Gender_ID_x": id_x,
        "Gender_ID_y": id_y,
        "sent_x": sent_x,
        "sent_y": sent_y,
        "parse_status": parse_status,
        "parse_status_detail": detail,
        "template_text": template_text,
        "prefix_template": split["prefix_template"],
        "predicate": pred,
        "continuation": split["continuation"],
        "prefix_x": prefix_x,
        "prefix_y": prefix_y,
        "full_x": sent_x,
        "full_y": sent_y,
        "identity_x_start_char": id_x_start,
        "identity_x_end_char": id_x_end,
        "identity_y_start_char": id_y_start,
        "identity_y_end_char": id_y_end,
        "predicate_x_start_char": pred_x_start,
        "predicate_x_end_char": pred_x_end,
        "predicate_y_start_char": pred_y_start,
        "predicate_y_end_char": pred_y_end,
        "needs_manual_review": bool(reasons),
        "review_reason": "; ".join(reasons),
    }


def _empty_parse(row: pd.Series, sent_x: str, sent_y: str, id_x: str, id_y: str) -> dict[str, Any]:
    ix = first_index(sent_x, id_x)
    iy = first_index(sent_y, id_y)
    return {
        "row_id": row.get("row_id"),
        "Gender_ID_x": id_x,
        "Gender_ID_y": id_y,
        "sent_x": sent_x,
        "sent_y": sent_y,
        "parse_status": "failed",
        "parse_status_detail": "failed",
        "template_text": "",
        "prefix_template": "",
        "predicate": "",
        "continuation": "",
        "prefix_x": "",
        "prefix_y": "",
        "full_x": sent_x,
        "full_y": sent_y,
        "identity_x_start_char": ix if ix >= 0 else -1,
        "identity_x_end_char": (ix + len(id_x)) if ix >= 0 else -1,
        "identity_y_start_char": iy if iy >= 0 else -1,
        "identity_y_end_char": (iy + len(id_y)) if iy >= 0 else -1,
        "predicate_x_start_char": -1,
        "predicate_x_end_char": -1,
        "predicate_y_start_char": -1,
        "predicate_y_end_char": -1,
        "needs_manual_review": True,
        "review_reason": "",
    }


OUTPUT_COLUMNS = [
    "row_id", "Gender_ID_x", "Gender_ID_y", "sent_x", "sent_y",
    "parse_status", "parse_status_detail",
    "template_text", "prefix_template", "predicate", "continuation",
    "prefix_x", "prefix_y", "full_x", "full_y",
    "identity_x_start_char", "identity_x_end_char", "identity_y_start_char", "identity_y_end_char",
    "predicate_x_start_char", "predicate_x_end_char", "predicate_y_start_char", "predicate_y_end_char",
    "needs_manual_review", "review_reason",
]


def write_summary(parsed: pd.DataFrame, summary_csv: Path) -> pd.DataFrame:
    total = len(parsed)
    n_review = int(parsed["needs_manual_review"].sum())
    rows: list[dict[str, Any]] = [
        {"metric": "total_rows", "value": total},
        {"metric": "needs_manual_review", "value": n_review},
        {"metric": "needs_manual_review_pct", "value": round(100.0 * n_review / total, 2) if total else 0.0},
    ]
    for status, count in parsed["parse_status"].value_counts().items():
        rows.append({"metric": f"parse_status::{status}", "value": int(count)})
    for detail, count in parsed["parse_status_detail"].value_counts().items():
        rows.append({"metric": f"parse_status_detail::{detail}", "value": int(count)})
    for tmpl, count in parsed["prefix_template"].value_counts().head(15).items():
        rows.append({"metric": f"top_prefix_template::{tmpl}", "value": int(count)})
    summary = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse WinoQueer pairs into prefix/predicate scoring units.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--sample_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    # Preserve a stable row id (handles an unnamed index column or none).
    first_col = df.columns[0]
    if str(first_col).startswith("Unnamed") or first_col == "" or first_col == "index":
        df = df.rename(columns={first_col: "row_id"})
    else:
        df = df.reset_index().rename(columns={"index": "row_id"})

    required = ["Gender_ID_x", "Gender_ID_y", "sent_x", "sent_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input_csv: {missing} (found {list(df.columns)})")

    parsed = pd.DataFrame([parse_row(r) for _, r in df.iterrows()])[OUTPUT_COLUMNS]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(args.out_csv, index=False)
    summary = write_summary(parsed, args.summary_csv)

    print(f"Parsed {len(parsed)} rows -> {args.out_csv}")
    print(f"Summary -> {args.summary_csv}\n")
    print(summary.to_string(index=False))

    n = min(args.sample_n, len(parsed))
    sample = parsed.sample(n=n, random_state=args.seed)
    print(f"\n=== {n} random examples ===")
    for _, r in sample.iterrows():
        print("-" * 88)
        print(f"sent_x      : {r['sent_x']}")
        print(f"sent_y      : {r['sent_y']}")
        print(f"prefix_x    : {r['prefix_x']!r}")
        print(f"prefix_y    : {r['prefix_y']!r}")
        print(f"continuation: {r['continuation']!r}")
        print(f"predicate   : {r['predicate']!r}")
        print(f"parse_status: {r['parse_status']} / {r['parse_status_detail']}  | review={r['needs_manual_review']}"
              + (f"  ({r['review_reason']})" if r['review_reason'] else ""))


if __name__ == "__main__":
    main()
