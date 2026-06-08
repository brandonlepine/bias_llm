#!/usr/bin/env python3
"""Auto-seed an identity crosswalk so per-identity v_bias can be sourced from whichever cohort holds
each identity — instead of dropping the ~100 identities WinoQueer doesn't cover.

Maps each identity-only canonical_label (the v_identity side) to:
  - its WinoQueer cohort `identity` label (LGBTQ axes), and
  - its combined BBQ+CrowS cohort `block` label (the CLEAN identity key; Group_x is noisy surface text).

Matching is EXACT on a normalized form (lowercase, strip leading a/an/the) plus a small curated
synonym set + the `aliases` column in bbq_identity_normalized_forms.csv. Anything that doesn't match
exactly is NOT guessed — it's emitted as AMBIGUOUS (with fuzzy candidates) or UNMATCHED for human
review, so we neither drop data nor silently mis-map it.

Output: data/identity_crosswalk.csv (one row per canonical identity) + a console report of the rows
that need eyeballing. v_bias_source = winoqueer | combined | both | none(axis-level fallback).
"""
from __future__ import annotations

import csv
import difflib
from collections import defaultdict
from pathlib import Path

import pandas as pd

ID_CSV = Path("data/mi_identity_prompts.csv")
NORM_CSV = Path("data/bbq_identity_normalized_forms.csv")
WQ_CSV = Path("data/winoqueer/results/segmented/cohort.csv")
COMB_CSV = Path("data/combined/results/_all/segmented/cohort.csv")
OUT = Path("data/identity_crosswalk.csv")

# identity-dataset axis -> combined-cohort axis label
AXIS_TO_COMBINED = {
    "race_ethnicity": "race", "gender_identity": "gender", "socioeconomic_status": "socioeconomic",
    "disability_status": "disability", "sexual_orientation": "sexual_orientation",
    "religion": "religion", "nationality": "nationality", "physical_appearance": "physical_appearance",
}
# WinoQueer `identity` exists only for these identity-dataset axes
WQ_AXES = {"sexual_orientation", "gender_identity"}
# cross-vocabulary synonyms we DO trust (normalized form -> normalized form)
CURATED = {"nb": "nonbinary"}            # WinoQueer 'NB' == identity 'nonbinary'

# Human-confirmed overrides (2026-06-08): canonical_label -> combined `block` it maps to. The first
# two are surface-spelling variants of the same identity; the last three are near-synonyms the
# identity dataset lists separately but the combined cohort doesn't distinguish (accepted: recovers
# data, the shared v_bias is harmless for the appraisal). References/opposites were rejected.
MANUAL_COMBINED = {
    "mental illness": "mentally ill", "poorly dressed": "badly dressed",
    "African American": "Black", "Caucasian": "White", "homosexual": "gay",
}


def norm(s: str) -> str:
    s = str(s).strip().lower()
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
    return " ".join(s.split())


def main() -> None:
    idf = pd.read_csv(ID_CSV)
    identities = idf[["identity_id", "axis", "canonical_label"]].drop_duplicates("canonical_label")

    aliases = defaultdict(list)
    if NORM_CSV.exists():
        nf = pd.read_csv(NORM_CSV)
        for _, r in nf.iterrows():
            if isinstance(r.get("aliases"), str) and r["aliases"].strip():
                aliases[r["canonical_label"]] = [norm(a) for a in r["aliases"].replace(";", ",").split(",")]

    # WinoQueer identities: normalized -> original label
    wq = pd.read_csv(WQ_CSV)
    wq_norm = {}
    for v in wq["identity"].dropna().unique():
        wq_norm[norm(v)] = v
    for k, target in CURATED.items():      # e.g. nb -> nonbinary, so canonical 'nonbinary' finds WQ 'NB'
        if k in wq_norm:
            wq_norm[target] = wq_norm[k]

    # Combined cohort: per combined-axis, normalized block -> original block (drop generic placeholders)
    comb = pd.read_csv(COMB_CSV)
    comb_blocks = defaultdict(dict)
    for ax in comb["axis"].dropna().unique():
        for b in comb[comb.axis == ax]["block"].dropna().unique():
            nb = norm(b)
            if nb == ax or nb in ("race", "gender", "religion", "nationality"):
                continue                   # generic axis placeholder, not an identity
            comb_blocks[ax][nb] = b

    rows, ambiguous, unmatched = [], [], []
    for _, it in identities.iterrows():
        lab, axis = it["canonical_label"], it["axis"]
        keys = [norm(lab)] + aliases.get(lab, [])

        wq_hit = next((wq_norm[k] for k in keys if axis in WQ_AXES and k in wq_norm), "")
        cax = AXIS_TO_COMBINED.get(axis, "")
        blocks = comb_blocks.get(cax, {})
        comb_hit = next((blocks[k] for k in keys if k in blocks), "")
        manual = ""
        if not comb_hit and lab in MANUAL_COMBINED and norm(MANUAL_COMBINED[lab]) in blocks:
            comb_hit = blocks[norm(MANUAL_COMBINED[lab])]; manual = comb_hit

        src = ("both" if wq_hit and comb_hit else "winoqueer" if wq_hit
               else "combined" if comb_hit else "none")
        status = "matched" if src != "none" else "unmatched"
        cands = ""
        if src == "none" and blocks:       # propose fuzzy candidates for review (don't auto-apply)
            close = difflib.get_close_matches(norm(lab), list(blocks), n=3, cutoff=0.6)
            if close:
                cands = "; ".join(blocks[c] for c in close)
                status = "ambiguous"
        row = {"identity_id": it["identity_id"], "canonical_label": lab, "axis": axis,
               "wq_identity": wq_hit, "combined_axis": cax, "combined_block": comb_hit,
               "v_bias_source": src, "status": status, "manual_override": manual,
               "candidates_for_review": cands}
        rows.append(row)
        (ambiguous if status == "ambiguous" else unmatched if status == "unmatched" else rows and None)
        if status == "ambiguous":
            ambiguous.append(row)
        elif status == "unmatched":
            unmatched.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    c = Counter(r["v_bias_source"] for r in rows)
    print(f"Wrote {OUT}  ({len(rows)} identities)")
    print(f"  v_bias source: both={c['both']}  winoqueer={c['winoqueer']}  combined={c['combined']}  "
          f"none(axis-fallback)={c['none']}")
    print(f"  -> {len(rows) - c['none']} identities get a per-identity v_bias (vs 7 before)\n")

    if ambiguous:
        print(f"AMBIGUOUS — no exact match, please confirm/deny the candidate ({len(ambiguous)}):")
        for r in ambiguous:
            print(f"  [{r['axis']}] {r['canonical_label']!r}  ~?  {r['candidates_for_review']}")
    if unmatched:
        print(f"\nUNMATCHED — v_identity only, will fall back to AXIS-level v_bias ({len(unmatched)}):")
        for r in unmatched:
            print(f"  [{r['axis']}] {r['canonical_label']}")


if __name__ == "__main__":
    main()
