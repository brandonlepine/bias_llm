#!/usr/bin/env python3
"""Analyze a scarce-selection run: focal selection rate by identity variant, odds
ratio, position-adjusted preference (vs control-neutral floor), and position-bias.

  python -m new_schemas.orgbench.analyze_selection --scored <run>/scored.jsonl
"""
import argparse, collections, json, math, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.scored) if '"selection_n"' in l]
    if not rows:
        raise SystemExit("no selection_n rows in scored file")
    N = rows[0]["n_candidates"]
    print(f"=== SCARCE SELECTION (choose 1 of {N}; chance baseline for focal = {1/N:.3f}) ===\n")

    by_cond = collections.defaultdict(lambda: collections.defaultdict(list))   # cond -> arm -> [chosen_is_focal]
    pf = collections.defaultdict(lambda: collections.defaultdict(list))        # cond -> arm -> [p_focal]
    paired = collections.defaultdict(dict)                                     # (cond,pair) -> arm -> row
    pos_focal = collections.defaultdict(list)                                  # focal_position -> chosen_is_focal (control)
    for r in rows:
        c = r["identity_signal_condition_id"]; a = r["treatment_or_control"]
        by_cond[c][a].append(r["chosen_is_focal"]); pf[c][a].append(r["p_focal"])
        paired[(c, r["paired_example_id"])][a] = r
        if a == "control":
            pos_focal[r["focal_position"]].append(r["chosen_is_focal"])

    for c in sorted(by_cond):
        print(f"[{c}]")
        for a in ("treatment", "control", "neutral"):
            if by_cond[c][a]:
                print(f"   {a:<10} focal selection rate={np.mean(by_cond[c][a]):.3f}  mean p_focal={np.mean(pf[c][a]):.3f}  n={len(by_cond[c][a])}")
        # odds ratio treatment vs control
        st, sc = np.mean(by_cond[c]["treatment"]), np.mean(by_cond[c]["control"])
        if 0 < sc < 1 and 0 < st < 1:
            orr = (st/(1-st)) / (sc/(1-sc))
            print(f"   odds ratio (treatment vs control) = {orr:.2f}")
        # position-adjusted preference (paired, same slate+position): p_focal treat - control
        dpt = [paired[k]["treatment"]["p_focal"] - paired[k]["control"]["p_focal"]
               for k in paired if k[0] == c and "treatment" in paired[k] and "control" in paired[k]]
        floor = [paired[k]["control"]["p_focal"] - paired[k]["neutral"]["p_focal"]
                 for k in paired if k[0] == c and "control" in paired[k] and "neutral" in paired[k]]
        if dpt:
            m = np.mean(dpt); se = np.std(dpt, ddof=1)/np.sqrt(len(dpt)) if len(dpt) > 1 else 0
            fl = (1.96*np.std(floor, ddof=1)/np.sqrt(len(floor))) if len(floor) > 1 else float("nan")
            verdict = "treatment FAVORED" if m > 0 else "treatment PENALIZED"
            print(f"   position-adjusted preference Δp_focal(treat−control) = {m:+.3f} ± {1.96*se:.3f}  "
                  f"floor(ctrl−neutral)=±{fl:.3f}  -> {verdict} ({'OUTSIDE' if abs(m) > (fl if fl==fl else 0) else 'within'} floor)")
        print()

    print("position bias (P focal chosen | focal at position p; control arm) — flat => no position bias:")
    for p in sorted(pos_focal):
        print(f"   position {p}: {np.mean(pos_focal[p]):.3f}  (n={len(pos_focal[p])})")


if __name__ == "__main__":
    main()
