#!/usr/bin/env python3
"""Prompt-condition robustness analysis.

Loads the new single/binary prompt-condition scores (scores_prompts.jsonl) AND
the prior multi_dim_rubric scores (scores_conditions.jsonl + scores_neutral.jsonl,
same batch=1 code path) and asks: do the LGBTQ effects survive across prompting
interfaces? Each prompt condition is calibrated against ITS OWN neutral noise
floor (control vs neutral_* under that same prompt), so cells are comparable.

For each prompt condition x dependent variable:
  delta = control - variant   (+ = variant scored LOWER)
  floor = max |control - neutral_*|
  a queer effect is credible only if |delta| > floor.

Read MAGNITUDES, not p (deterministic + paired => p is meaningless here).

Example:
  python -m pilot.analyze_prompts
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

from .analyze import get_dv, stats_block
from .prompts import PROMPT_CONDITIONS, MULTI_DIM_MATRIX
from .signals import (CONDITIONS, NEUTRAL_CONDITIONS, CONTROL_CONDITION,
                      GENERIC_LGBTQ_CONDITION)

MATRIX_COLS = ["overall_fit", "technical", "experience", "communication", "decision"]
PROMPT_ORDER = (["multi_dim_rubric"] + [p["name"] for p in PROMPT_CONDITIONS])

# {prompt_condition: {matrix_col: dv_key}}
PROMPT_DV_MAP = {"multi_dim_rubric": dict(MULTI_DIM_MATRIX)}
for p in PROMPT_CONDITIONS:
    PROMPT_DV_MAP.setdefault(p["name"], {})[p["matrix_col"]] = p["dv_key"]

LGBTQ_VARIANTS = [c["condition_name"] for c in CONDITIONS
                  if c["condition_name"] != CONTROL_CONDITION]
DOMAIN_VARIANTS = [c["condition_name"] for c in CONDITIONS
                   if c["condition_name"] not in (CONTROL_CONDITION, GENERIC_LGBTQ_CONDITION)]
NEUTRAL_NAMES = [c["condition_name"] for c in NEUTRAL_CONDITIONS]


def load(scores_prompts, multidim_files):
    """data[prompt_condition][pair_id][resume_condition] = row."""
    data = {}
    # new prompt-condition rows
    if os.path.exists(scores_prompts):
        with open(scores_prompts) as f:
            for line in f:
                r = json.loads(line)
                pc = r["prompt_condition"]
                data.setdefault(pc, {}).setdefault(r["pair_id"], {})[r["resume_condition"]] = r
    # prior multi_dim rows (use condition_name as resume_condition)
    for path in multidim_files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rc = r.get("resume_condition", r.get("condition_name"))
                data.setdefault("multi_dim_rubric", {}).setdefault(
                    r["pair_id"], {})[rc] = r
    return data


def deltas(by_pair, ref, var, dv):
    out = []
    for pid, d in by_pair.items():
        if ref not in d or var not in d:
            continue
        rv, vv = get_dv(d[ref], dv), get_dv(d[var], dv)
        if rv is None or vv is None:
            continue
        out.append((pid, d[var].get("gender"), rv - vv))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-prompts",
                    default=os.path.join(ROOT, "results", "scores_prompts.jsonl"))
    ap.add_argument("--multidim", nargs="+",
                    default=[os.path.join(ROOT, "results", "scores_conditions.jsonl"),
                             os.path.join(ROOT, "results", "scores_neutral.jsonl")])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "summary_prompts.json"))
    args = ap.parse_args()

    data = load(args.scores_prompts, args.multidim)
    present_pc = [pc for pc in PROMPT_ORDER if pc in data]
    print(f"Prompt conditions loaded: {present_pc}\n")

    # compute floors + effects, keyed by (pc, col)
    floor = {}        # (pc,col) -> noise floor
    ctrl_eff = {}     # (pc,col,variant) -> stats
    gen_eff = {}      # (pc,col,variant) -> stats vs generic
    for pc in present_pc:
        by_pair = data[pc]
        for col, dv in PROMPT_DV_MAP[pc].items():
            neut = [abs(stats_block(deltas(by_pair, CONTROL_CONDITION, n, dv))["mean"])
                    for n in NEUTRAL_NAMES
                    if deltas(by_pair, CONTROL_CONDITION, n, dv)]
            floor[(pc, col)] = max(neut) if neut else None
            for v in LGBTQ_VARIANTS:
                r = stats_block(deltas(by_pair, CONTROL_CONDITION, v, dv))
                if r:
                    ctrl_eff[(pc, col, v)] = r
            for v in DOMAIN_VARIANTS:
                r = stats_block(deltas(by_pair, GENERIC_LGBTQ_CONDITION, v, dv))
                if r:
                    gen_eff[(pc, col, v)] = r

    # ===== ROBUSTNESS MATRIX (generic_lgbtq vs control) =====================
    print("=== ROBUSTNESS MATRIX: generic_lgbtq vs control ===")
    print("    cell = delta (OUT = outside neutral noise floor, in = within); + = LGBTQ scored LOWER")
    print(f"{'prompt_condition':<22}" + "".join(f"{c:>16}" for c in MATRIX_COLS))
    for pc in present_pc:
        cells = []
        for col in MATRIX_COLS:
            r = ctrl_eff.get((pc, col, GENERIC_LGBTQ_CONDITION))
            fl = floor.get((pc, col))
            if not r or fl is None:
                cells.append("       .       ")
            else:
                tag = "OUT" if abs(r["mean"]) > fl else " in"
                cells.append(f"{tag} {r['mean']:+.4f}")
        print(f"{pc:<22}" + "".join(f"{c:>16}" for c in cells))

    # ===== PER-PROMPT TOTAL-EFFECT TABLES (all variants, * outside floor) ====
    for pc in present_pc:
        cols = [c for c in MATRIX_COLS if c in PROMPT_DV_MAP[pc]]
        print(f"\n### {pc}: control - variant  (* = outside floor)  floor=("
              + ", ".join(f"{c}:{floor[(pc,c)]:.4f}" if floor.get((pc,c)) is not None
                          else f"{c}:na" for c in cols) + ")")
        print(f"{'variant':<20}" + "".join(f"{c:>16}" for c in cols))
        for v in LGBTQ_VARIANTS:
            row = []
            for c in cols:
                r = ctrl_eff.get((pc, c, v)); fl = floor.get((pc, c))
                if not r:
                    row.append("       .       ")
                else:
                    s = "*" if (fl is not None and abs(r["mean"]) > fl) else " "
                    row.append(f"{r['mean']:+.4f}{s}")
            print(f"{v:<20}" + "".join(f"{x:>16}" for x in row))

    # ===== DOMAIN MODULATION (vs generic_lgbtq), decision + technical ========
    print("\n### DOMAIN MODULATION: generic_lgbtq - variant (+ = variant LOWER than generic)")
    for pc in present_pc:
        cols = [c for c in MATRIX_COLS if c in PROMPT_DV_MAP[pc]]
        for c in cols:
            parts = []
            for v in DOMAIN_VARIANTS:
                r = gen_eff.get((pc, c, v))
                if r:
                    parts.append(f"{v.replace('lgbtq_','')}={r['mean']:+.4f}")
            if parts:
                print(f"  {pc:<22} [{c}] " + "  ".join(parts))

    # ===== DETAIL: generic_lgbtq vs control, CIs =============================
    print("\n=== DETAIL: generic_lgbtq vs control (mean [95% CI] dz favor/n; floor) ===")
    for pc in present_pc:
        for col in [c for c in MATRIX_COLS if c in PROMPT_DV_MAP[pc]]:
            r = ctrl_eff.get((pc, col, GENERIC_LGBTQ_CONDITION)); fl = floor.get((pc, col))
            if not r:
                continue
            fls = f"{fl:.4f}" if fl is not None else "na"
            verdict = "  OUT" if (fl is not None and abs(r['mean']) > fl) else "  in"
            print(f"  {pc:<22} {col:<13} {r['mean']:+.4f} "
                  f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] dz={r['cohens_dz']:+.2f} "
                  f"{r['favor_control']}/{r['n']}  floor={fls}{verdict}")

    # serialize
    def k(t): return "|".join(map(str, t))
    summary = {"floor": {k(x): floor[x] for x in floor},
               "ctrl_effect": {k(x): ctrl_eff[x] for x in ctrl_eff},
               "gen_effect": {k(x): gen_eff[x] for x in gen_eff}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
