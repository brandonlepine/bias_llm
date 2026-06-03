#!/usr/bin/env python3
"""Build WinoQueer-format minority/dominant paired sentences from the curated BBQ stereotypes.

For each kept (template -> curated predicate) and each target group it names, pair the TARGET group
against its DOMINANT/reference group (from bbq_group_taxonomy) in two frames:
  - name      : "{Name} is {GROUP} and {predicate}"      (continuation = predicate)
  - groupnoun : "{GROUP_PLURAL} are {predicate_plural}"   (continuation = predicate_plural)
Both sentences in a pair differ only in the single group span, so downstream `align_pair` aligns them.

Schema matches the WinoQueer pipeline (sent_x/sent_y/prefix_x/prefix_y/continuation/predicate) plus
generic metadata (Group_x=target, Group_y=dominant, category, axis, predicate_label_provisional, frame).

Output: data/bbq/stereotypes/bbq_pairs_all.csv  (+ per-category files with --split)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bbq_group_taxonomy as tax  # noqa: E402


def pluralize(pred: str) -> str:
    """'a criminal'->'criminals', 'an illegal immigrant'->'illegal immigrants'; adjectives/PPs unchanged."""
    for art in ("a ", "an "):
        if pred.startswith(art):
            body = pred[len(art):]
            words = body.split()
            words[-1] = words[-1] + ("es" if words[-1].endswith(("s", "x", "z", "ch", "sh")) else "s")
            return " ".join(words)
    return pred


def split_groups(value: str) -> list[str]:
    return [t.strip() for t in str(value).split(";") if t.strip()]


# Carrier frames: each has a swappable {grp} slot and {pred} at the END (the scored continuation).
# requires_name -> uses a {name}; plural -> use the group-noun surface + pluralized predicate,
# else the copular surface + singular predicate. Add/remove freely to scale the dataset.
#   (frame_id, requires_name, plural, template)
FRAMES = [
    ("name_and",        True,  False, "{name} is {grp} and {pred}"),
    ("name_who",        True,  False, "{name}, who is {grp}, is {pred}"),
    ("name_being",      True,  False, "Being {grp}, {name} is {pred}"),
    ("group_are",       False, True,  "{grp} are {pred}"),
    ("group_tend",      False, True,  "{grp} tend to be {pred}"),
    ("group_most",      False, True,  "Most {grp} are {pred}"),
    ("group_known",     False, True,  "It is well known that {grp} are {pred}"),
]


def render_frame(tmpl: str, name: str, grp: str, pred: str) -> tuple[str, str]:
    """Return (prefix, sentence) for a frame, where prefix is everything up to (excl.) the predicate."""
    pre = tmpl.split("{pred}")[0].format(name=name, grp=grp).rstrip()
    return pre, f"{pre} {pred}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BBQ -> WinoQueer-style paired sentences.")
    ap.add_argument("--raw", type=Path, default=Path("data/bbq/stereotypes/bbq_stereotypes_raw.csv"))
    ap.add_argument("--curated", type=Path, default=Path("data/bbq/stereotypes/bbq_predicates_curated.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/bbq/stereotypes/bbq_pairs_all.csv"))
    ap.add_argument("--n_names", type=int, default=3, help="# neutral names per name-based frame.")
    ap.add_argument("--frames", type=str, default=",".join(f[0] for f in FRAMES),
                    help="comma list of frame ids to emit (default: all).")
    ap.add_argument("--split", action="store_true", help="also write per-category CSVs next to --out")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = [f.strip() for f in args.frames.split(",")]
    names = tax.NEUTRAL_NAMES[: args.n_names]

    raw = pd.read_csv(args.raw).fillna("")
    cur = pd.read_csv(args.curated).fillna("")
    cur = cur[(cur["keep"].astype(str).str.lower().isin({"true", "1"})) & (cur["predicate"] != "")]
    pred_map = {(r["category"], r["social_value"]): r["predicate"] for _, r in cur.iterrows()}

    rows, rid, skipped = [], 0, {"unsupported": 0, "no_predicate": 0, "degenerate": 0, "no_surface": 0}
    for _, t in raw.iterrows():
        cat = t["category"]
        if not tax.is_supported(cat):
            skipped["unsupported"] += 1
            continue
        predicate = pred_map.get((cat, t["social_value"]))
        if not predicate:
            skipped["no_predicate"] += 1
            continue
        pred_pl = pluralize(predicate)
        axis = tax.axis_of(cat)
        # canonicalize the template's target tokens; drop the umbrella group if a specific co-occurs
        canons = [c for c in (tax.canonicalize(cat, tok) for tok in split_groups(t["target_groups"])) if c]
        umb = tax.umbrella_canons(cat)
        specifics = [c for c in canons if c not in umb]
        use_canons = list(dict.fromkeys(specifics if specifics else canons))
        seen_pairs = set()
        for canon in use_canons:
            ref = tax.reference_of(cat, canon)
            if canon == ref:
                skipped["degenerate"] += 1
                continue
            st, sr = tax.surface(cat, canon), tax.surface(cat, ref)
            if not st or not sr:
                skipped["no_surface"] += 1
                continue
            if (canon, ref) in seen_pairs:
                continue
            seen_pairs.add((canon, ref))
            cop_t, noun_t = st
            cop_r, noun_r = sr
            for fid, req_name, plural, tmpl in FRAMES:
                if fid not in frames:
                    continue
                grp_t, grp_r = (noun_t, noun_r) if plural else (cop_t, cop_r)
                p = pred_pl if plural else predicate
                frame_names = names if req_name else [""]
                for nm in frame_names:
                    px, sx = render_frame(tmpl, nm, grp_t, p)
                    py, sy = render_frame(tmpl, nm, grp_r, p)
                    rows.append(dict(
                        row_id=rid, Group_x=canon, Group_y=ref, category=cat, axis=axis,
                        block=tax.block_of(cat, canon),
                        predicate_label_provisional=t["social_value"], frame=fid, name=nm,
                        q_id=t["q_id"], predicate=p, continuation=p,
                        prefix_x=px, prefix_y=py, sent_x=sx, sent_y=sy,
                    )); rid += 1

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(out)} pairs)")
    print("skipped:", skipped)
    print("\npairs per category x frame:")
    print(out.groupby(["category", "frame"]).size().to_string())
    if args.split:
        for cat, g in out.groupby("category"):
            p = args.out.parent / f"bbq_pairs_{cat}.csv"
            g.to_csv(p, index=False)
        print(f"\nwrote {out['category'].nunique()} per-category files")


if __name__ == "__main__":
    main()
