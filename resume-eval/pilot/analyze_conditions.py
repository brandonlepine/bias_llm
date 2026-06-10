#!/usr/bin/env python3
"""Condition-level paired analysis for the stereotype-violation design, WITH a
neutral-vs-neutral noise floor.

The readout is deterministic, so ANY token swap ripples a small consistent amount
through the network. To attribute a queer-vs-control delta to IDENTITY rather than
generic lexical sensitivity, we compare it against the NULL BAND: control vs
neutral-token swaps (neutral_1tok / neutral_2tok / neutral_arts). A queer effect
is only credible if its |delta| falls OUTSIDE the neutral spread for that DV.

References:
  vs control       delta = control - variant            (TOTAL effect)
  vs generic_lgbtq delta = generic_lgbtq - variant       (DOMAIN MODULATION)
  NULL BAND        delta = control - neutral_*            (noise floor)

Read MAGNITUDES (mean delta, dz), not p: the deterministic paired design makes
nearly everything p<1e-9; that means "stable across names", not "large or real".

Example:
  python -m pilot.analyze_conditions \
    --scores results/scores_conditions.jsonl results/scores_neutral.jsonl --by-gender
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

from .analyze import get_dv, stats_block
from .signals import (CONDITIONS, NEUTRAL_CONDITIONS, CONTROL_CONDITION,
                      GENERIC_LGBTQ_CONDITION)

# logit_diff is a monotone transform of p_yes (identical dz / favor counts), so
# we report only p_yes for the decision.
DVS = [
    ("p_yes", "decision p(adv)"),
    ("overall_fit", "overall_fit"),
    ("technical_qualifications", "technical"),
    ("relevant_experience", "experience"),
    ("communication_collaboration", "communic."),
]
NEUTRAL_NAMES = [c["condition_name"] for c in NEUTRAL_CONDITIONS]


def load_by_pair(paths):
    by_pair = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_pair.setdefault(r["pair_id"], {})[r["condition_name"]] = r
    return by_pair


def paired_deltas(by_pair, ref_cond, var_cond, dv_key):
    out = []
    for pid, d in by_pair.items():
        if ref_cond not in d or var_cond not in d:
            continue
        rv, vv = get_dv(d[ref_cond], dv_key), get_dv(d[var_cond], dv_key)
        if rv is None or vv is None:
            continue
        out.append((pid, d[var_cond].get("gender"), rv - vv))
    return out


def gfmt(x):
    return "   .   " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.4f}"


def matrix(by_pair, ref_cond, variants, title, null_band=None):
    star = "  (* = |delta| exceeds neutral noise floor)" if null_band else ""
    print(f"\n### {title}{star}")
    print(f"    cell = mean({ref_cond} - variant); + = variant scored LOWER")
    print(f"{'condition':<20}" + "".join(f"{lbl:>17}" for _, lbl in DVS))
    store = {}
    for v in variants:
        cells = []
        for dv, _ in DVS:
            res = stats_block(paired_deltas(by_pair, ref_cond, v, dv))
            store[(v, dv)] = res
            txt = gfmt(res["mean"]) if res else "   .   "
            if null_band and res and null_band.get(dv) is not None \
                    and abs(res["mean"]) > null_band[dv] + 1e-12:
                txt += "*"
            cells.append(txt)
        print(f"{v:<20}" + "".join(f"{c:>17}" for c in cells))
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+",
                    default=[os.path.join(ROOT, "results", "scores_conditions.jsonl"),
                             os.path.join(ROOT, "results", "scores_neutral.jsonl")])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "summary_conditions.json"))
    ap.add_argument("--by-gender", action="store_true")
    args = ap.parse_args()

    by_pair = load_by_pair(args.scores)
    present = set()
    for d in by_pair.values():
        present.update(d.keys())
    cond_names = [c["condition_name"] for c in CONDITIONS]
    lgbtq_variants = [c for c in cond_names if c != CONTROL_CONDITION]
    domain_variants = [c for c in cond_names
                       if c not in (CONTROL_CONDITION, GENERIC_LGBTQ_CONDITION)]
    neutrals = [n for n in NEUTRAL_NAMES if n in present]
    npairs = len(by_pair)
    print(f"Loaded {sum(len(d) for d in by_pair.values())} scores across {npairs} pairs.")
    print(f"Conditions present: {sorted(present)}")

    # --- NULL BAND: control vs neutral swaps -> per-DV noise floor -----------
    null_band = {}
    if neutrals:
        null_store = matrix(by_pair, CONTROL_CONDITION, neutrals,
                            "NULL BAND: control vs NEUTRAL token swaps (noise floor)")
        for dv, _ in DVS:
            mags = [abs(null_store[(n, dv)]["mean"]) for n in neutrals
                    if null_store.get((n, dv))]
            null_band[dv] = max(mags) if mags else None
        print("\n  per-DV noise floor (max |control-neutral| mean): "
              + ", ".join(f"{lbl}={null_band[dv]:.4f}" if null_band[dv] is not None
                          else f"{lbl}=na" for dv, lbl in DVS))
    else:
        print("\n[no neutral conditions found -> NO noise floor; effects UNCALIBRATED]")
        null_band = None

    # --- TOTAL EFFECT vs control (flagged against null band) ----------------
    ctrl_store = matrix(by_pair, CONTROL_CONDITION, lgbtq_variants,
                        "TOTAL EFFECT vs control", null_band=null_band)
    # --- DOMAIN MODULATION vs generic_lgbtq ---------------------------------
    gen_store = matrix(by_pair, GENERIC_LGBTQ_CONDITION, domain_variants,
                       "DOMAIN MODULATION vs generic_lgbtq")

    # --- verdict per DV: does generic_lgbtq clear the noise floor? ----------
    if null_band:
        print("\n### NOISE-FLOOR VERDICT (generic_lgbtq vs control)")
        for dv, lbl in DVS:
            res = ctrl_store.get((GENERIC_LGBTQ_CONDITION, dv))
            nb = null_band.get(dv)
            if not res or nb is None:
                continue
            verdict = "OUTSIDE null (credible identity effect)" if abs(res["mean"]) > nb \
                else "WITHIN null (indistinguishable from a generic token swap)"
            print(f"  {lbl:<16} control-generic={res['mean']:+.4f}  floor={nb:.4f}  -> {verdict}")

    # --- detail -------------------------------------------------------------
    print("\n=== DETAIL (mean [95% CI] dz favor_ref/n) ===")

    def detail(ref, variants):
        print(f"\n-- reference: {ref} --")
        for v in variants:
            print(f"  [{v}]")
            for dv, lbl in DVS:
                res = stats_block(paired_deltas(by_pair, ref, v, dv))
                if not res:
                    continue
                print(f"     {lbl:<16} {res['mean']:+.4f} "
                      f"[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] dz={res['cohens_dz']:+.2f} "
                      f"favor_ref={res['favor_control']}/{res['n']}")

    if neutrals:
        detail(CONTROL_CONDITION, neutrals)
    detail(CONTROL_CONDITION, lgbtq_variants)
    detail(GENERIC_LGBTQ_CONDITION, domain_variants)

    if args.by_gender:
        print("\n=== overall_fit vs control, by perceived gender ===")
        for v in lgbtq_variants:
            diffs = paired_deltas(by_pair, CONTROL_CONDITION, v, "overall_fit")
            line = f"  {v:<20}"
            for gd in ("male", "female", "ambiguous"):
                sub = [(p, gg, d) for (p, gg, d) in diffs if gg == gd]
                res = stats_block(sub)
                line += f"  {gd[:4]}={res['mean']:+.4f}" if res else f"  {gd[:4]}=  .  "
            print(line)

    def pack(store):
        return {f"{v}|{dv}": store[(v, dv)] for (v, dv) in store}
    summary = {"n_pairs": npairs, "noise_floor": null_band,
               "vs_control": pack(ctrl_store), "vs_generic_lgbtq": pack(gen_store)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
