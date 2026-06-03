#!/usr/bin/env python3
"""Extract scorable minimal-pair prompts from CrowS-Pairs, in the WinoQueer/BBQ schema.

CrowS-Pairs (Nangia et al. 2020) gives sent_more / sent_less that differ only in the identity term,
across the same 9 axes we use. We convert each into a target-vs-dominant continuation-scoring prompt:
the differing words are the identity (Group_x in sent_more = target/disadvantaged, Group_y in
sent_less = dominant), and the shared SUFFIX after the identity is the scored continuation.

Known caveats (Blodgett et al. 2021, "Stereotyping Norwegian Salmon"): CrowS-Pairs has validity
noise — some pairs differ in MORE than the identity, some are mislabeled. Mitigations here:
  - keep only a SINGLE short contiguous difference (drops multi-difference items);
  - require a non-trivial shared suffix after the identity (so there is something to score);
  - keep only `stereo` items (sent_more = stereotype) for a clean bias direction;
  - tag everything `source=crows-pairs` so it stays separable from BBQ in analysis.
These are SUPPLEMENTARY identity-level power, cross-checked against BBQ — not a standalone claim.

Output: data/crows-pairs/crows_pairs_prompts.csv  (sent_x/sent_y/prefix_x/prefix_y/continuation/
        Group_x/Group_y/axis/predicate_label_provisional/frame/source/stereo_antistereo)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

AXIS_MAP = {
    "race-color": "race", "gender": "gender", "socioeconomic": "socioeconomic",
    "nationality": "nationality", "religion": "religion", "age": "age",
    "sexual-orientation": "sexual_orientation", "physical-appearance": "physical_appearance",
    "disability": "disability",
}

# Normalize the raw CrowS differing span to a canonical identity (for alignment with BBQ). Substring
# match on the lowercased span; first hit wins. Unmatched spans keep the cleaned raw text (mapped=False)
# and should be used only at the axis level, not per-identity.
CANON = {
    "race": [("black", "Black"), ("african", "Black"), ("white", "White"), ("asian", "Asian"),
             ("hispanic", "Hispanic"), ("latino", "Hispanic"), ("mexican", "Hispanic"),
             ("arab", "Middle Eastern"), ("native american", "Native American"), ("jew", "Jewish")],
    "gender": [("women", "woman"), ("woman", "woman"), ("girl", "woman"), ("female", "woman"), (" she", "woman"),
               ("men", "man"), ("man", "man"), ("boy", "man"), ("male", "man"), (" he", "man"), ("trans", "transgender")],
    "socioeconomic": [("poor", "poor"), ("povert", "poor"), ("rich", "rich"), ("wealth", "rich")],
    "sexual_orientation": [("gay", "gay"), ("homosexual", "gay"), ("lesbian", "lesbian"),
                           ("bisexual", "bisexual"), ("straight", "straight")],
    "religion": [("muslim", "Muslim"), ("islam", "Muslim"), ("jew", "Jewish"), ("christian", "Christian"),
                 ("catholic", "Catholic"), ("hindu", "Hindu"), ("mormon", "Mormon"), ("atheist", "Atheist")],
    "age": [("old", "old"), ("elder", "old"), ("senior", "old"), ("young", "young"), ("teen", "young"), ("child", "young")],
    "physical_appearance": [("fat", "obese"), ("overweight", "obese"), ("obese", "obese"), ("heavy", "obese"),
                            ("ugly", "unattractive"), ("thin", "thin"), ("skinny", "thin"), ("short", "short"), ("tall", "tall")],
    "disability": [("mental", "mentally ill"), ("autis", "autistic"), ("deaf", "deaf"), ("blind", "blind"),
                   ("retard", "disabled"), ("handicap", "disabled"), ("disab", "disabled"), ("wheelchair", "disabled")],
    "nationality": [],  # demonyms used as-is (cleaned)
}


def clean_identity(axis: str, span: str) -> tuple[str, bool]:
    low = re.sub(r"[^a-z ]", " ", span.lower())
    for needle, canon in CANON.get(axis, []):
        if needle.strip() in low:
            return canon, True
    cleaned = re.sub(r"[^A-Za-z -]", "", span).strip()
    return (cleaned or span.strip()), False


def words(s: str) -> list[str]:
    return str(s).strip().split()


def common_prefix(a: list[str], b: list[str]) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def common_suffix(a: list[str], b: list[str]) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract scorable minimal pairs from CrowS-Pairs.")
    ap.add_argument("--crows_csv", type=Path, default=Path("data/crows-pairs/crows-pairs.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/crows-pairs/crows_pairs_prompts.csv"))
    ap.add_argument("--max_identity_words", type=int, default=4, help="drop multi-difference items (long diff span)")
    ap.add_argument("--min_suffix_words", type=int, default=3, help="require this many shared words after the identity")
    ap.add_argument("--keep_antistereo", action="store_true", help="also keep antistereo items (direction flipped)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.crows_csv)
    rows, drop = [], {"axis": 0, "antistereo": 0, "no_single_span": 0, "long_identity": 0, "short_suffix": 0}
    for i, r in d.iterrows():
        axis = AXIS_MAP.get(str(r.get("bias_type", "")).strip())
        if axis is None:
            drop["axis"] += 1; continue
        sa = str(r.get("stereo_antistereo", "stereo")).strip()
        if sa != "stereo" and not args.keep_antistereo:
            drop["antistereo"] += 1; continue
        wx, wy = words(r["sent_more"]), words(r["sent_less"])
        P, S = common_prefix(wx, wy), common_suffix(wx, wy)
        S = min(S, len(wx) - P, len(wy) - P)  # don't let prefix/suffix overlap
        idx, idy = wx[P:len(wx) - S], wy[P:len(wy) - S]
        if not idx or not idy:
            drop["no_single_span"] += 1; continue
        if len(idx) > args.max_identity_words or len(idy) > args.max_identity_words:
            drop["long_identity"] += 1; continue
        if S < args.min_suffix_words:
            drop["short_suffix"] += 1; continue
        # target = sent_more's identity (stereo); for antistereo the direction is reversed (flag kept)
        cont = " ".join(wy[len(wy) - S:])
        canon_x, mapped_x = clean_identity(axis, " ".join(idx))
        canon_y, mapped_y = clean_identity(axis, " ".join(idy))
        rows.append({
            "row_id": int(i), "source": "crows-pairs", "category": axis, "axis": axis, "block": axis,
            "Group_x": canon_x, "Group_y": canon_y, "identity_mapped": bool(mapped_x and mapped_y),
            "Group_x_raw": " ".join(idx), "Group_y_raw": " ".join(idy),
            "predicate_label_provisional": "CROWS", "frame": "crows", "stereo_antistereo": sa,
            "predicate": cont, "continuation": cont,
            "prefix_x": " ".join(wx[:len(wx) - S]), "prefix_y": " ".join(wy[:len(wy) - S]),
            "sent_x": str(r["sent_more"]).strip(), "sent_y": str(r["sent_less"]).strip(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"CrowS-Pairs rows: {len(d)} -> {len(out)} scorable prompts")
    print("dropped:", drop)
    print("\nkept per axis:")
    print(out.groupby("axis").size().sort_values(ascending=False).to_string())
    print(f"\nidentity-mapped (clean, usable per-identity): {int(out['identity_mapped'].sum())}/{len(out)}")
    print("\ncanonical identity terms per axis (top, mapped):")
    for ax, g in out.groupby("axis"):
        top = g[g["identity_mapped"]]["Group_x"].value_counts().head(5).to_dict()
        print(f"  {ax:20s} {top}")


if __name__ == "__main__":
    main()
