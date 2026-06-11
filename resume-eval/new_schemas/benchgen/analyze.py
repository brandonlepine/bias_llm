#!/usr/bin/env python3
"""Analyze a scored benchmark run: paired identity effects vs a neutral noise floor,
across prompt conditions, identity loads, qualification profiles, and signal channels,
with labeled visualizations.

Orientation: delta = control_score - treatment_score.
  POSITIVE  => the identity-signaled (treatment) candidate scored LOWER  (penalized).
  NEGATIVE  => the identity candidate scored HIGHER (favored).
For salary/bonus the delta is in USD; for the yes/no bonus it is in p(Yes).

Noise floor = control vs NEUTRAL (a different non-identity signal). The identity
effect is only credible where |control-treatment| exceeds that floor; figures shade
the ±floor band and mark bars/cells that clear it.

Figures (matplotlib, Agg; NO tight_layout -- explicit subplots_adjust):
  figures/effect_<prompt_condition>.png  (by load / by qualification / by channel)
  figures/pairwise_choice.png            (P(treatment chosen), both orders + order effect)
  figures/summary_heatmap.png            (effect in floor-units across conditions)

Run from resume-eval/:
  python -m new_schemas.benchgen.analyze --scored new_schemas/runs/<...>/scored.jsonl
"""
import argparse
import collections
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot.analyze import stats_block  # reuse validated bootstrap/dz/CI


def dv(row):
    # salary/bonus bias is on the REALISTIC $5k-increment offer (modal), not the interpolated EV
    if row.get("output_type") in ("salary_increment", "bonus_increment"):
        return row.get("modal_offer_usd")
    return row.get("parsed_score")


def paired_records(rows, prompt_condition):
    """Per paired group: control-treatment and control-neutral deltas + covariates."""
    groups = collections.defaultdict(dict)
    for r in rows:
        if r["prompt_condition_id"] != prompt_condition:
            continue
        groups[r["paired_example_id"]][r["treatment_or_control"]] = r
    recs = []
    for pid, arms in groups.items():
        if "control" not in arms or "treatment" not in arms:
            continue
        c, t = dv(arms["control"]), dv(arms["treatment"])
        if c is None or t is None:
            continue
        n = dv(arms["neutral"]) if "neutral" in arms else None
        meta = arms["control"]
        recs.append({"pair_id": pid, "gender": meta.get("perceived_gender"),
                     "qual": meta["qualification_profile_id"], "load": meta["identity_load"],
                     "condition": meta["identity_signal_condition_id"],
                     "channel": (meta.get("signal_channels") or ["none"])[0],
                     "ct": c - t, "cn": (c - n) if n is not None else None})
    return recs


def agg(recs, key, value="ct"):
    """{level: stats_block} grouping recs[value] by recs[key]."""
    by = collections.defaultdict(list)
    for r in recs:
        if r[value] is not None:
            by[r[key]].append((r["pair_id"], r["gender"], r[value]))
    return {lvl: stats_block(v) for lvl, v in by.items() if v}


def floor_value(recs):
    """Noise-floor band from control-vs-neutral: max(|CI bounds|) of its mean."""
    vals = [(r["pair_id"], r["gender"], r["cn"]) for r in recs if r["cn"] is not None]
    if not vals:
        return None
    s = stats_block(vals)
    return max(abs(s["ci_lo"]), abs(s["ci_hi"]))


def _ci_err(st):
    return [[st["mean"] - st["ci_lo"]], [st["ci_hi"] - st["mean"]]]


def _panel(ax, statmap, floor, title, ylabel, order=None):
    levels = order or sorted(statmap.keys(), key=lambda k: str(k))
    xs = range(len(levels))
    means = [statmap[l]["mean"] for l in levels]
    out = [floor is not None and abs(m) > floor for m in means]
    colors = ["#2ca02c" if o else "#bbbbbb" for o in out]
    bars = ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    for i, l in enumerate(levels):
        st = statmap[l]
        ax.errorbar(i, st["mean"], yerr=_ci_err(st), fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3, zorder=4)
    if floor is not None:
        ax.axhspan(-floor, floor, color="#999999", alpha=0.22, zorder=1,
                   label=f"noise floor (±{floor:.3g})")
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    for i, o in enumerate(out):
        if o:
            y = means[i]
            ax.annotate("*", (i, y), ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=14, fontweight="bold", zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(l).replace("QP_", "").replace("load", "L") for l in levels],
                       rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7, loc="best")


def figure_for_condition(pc, recs, units, fig_dir):
    floor = floor_value(recs)
    by_load = agg(recs, "load")
    by_qual = agg(recs, "qual")
    l1 = [r for r in recs if r["load"] == 1]
    by_chan = agg(l1, "channel") if l1 else {}
    panels = [("by identity load (= channel COUNT, NOT a dose)", by_load, sorted(by_load, key=lambda k: (isinstance(k, str), k))),
              ("by qualification profile", by_qual, None)]
    if by_chan:
        panels.append(("by signal channel (load 1)", by_chan, None))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle(f"{pc}: control − treatment   (positive = identity-signaled candidate penalized)\n"
                 f"bars clearing the noise floor are green and marked *", fontsize=11)
    for ax, (title, statmap, order) in zip(axes, panels):
        if statmap:
            _panel(ax, statmap, floor, title, f"mean Δ ({units})", order=order)
            ax.set_xlabel("condition level", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.26, wspace=0.30)  # NOT tight_layout
    path = os.path.join(fig_dir, f"effect_{pc}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return floor, by_load


def pairwise_figure(rows, fig_dir):
    """Position-DEBIASED pairwise preference. The raw A/B logit is saturated by a
    position-A primacy bias (the model nearly always picks A), so the discrete choice
    is uninformative; the identity preference is T = (logit_AB|treat=A - logit_AB|treat=B)/2,
    which differences out the constant position bias. + = treatment (LGBTQ) favored."""
    pw = [r for r in rows if r.get("output_type") == "pairwise_AB"]
    if not pw:
        return None
    frac_A = sum(r["chosen_candidate"] == "A" for r in pw) / len(pw)
    g = collections.defaultdict(dict); cond = {}
    for r in pw:
        g[r["paired_example_id"]][r["candidate_a_variant_type"]] = r["logit_A_minus_logit_B"]
        cond[r["paired_example_id"]] = r["identity_signal_condition_id"]
    byc = collections.defaultdict(list)
    for pid, d in g.items():
        if "treatment" in d and "control" in d:
            byc[cond[pid]].append((d["treatment"] - d["control"]) / 2)
    conds = sorted(byc)
    means = [float(np.mean(byc[c])) for c in conds]
    err = [1.96 * float(np.std(byc[c], ddof=1)) / np.sqrt(len(byc[c])) for c in conds]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.suptitle("Pairwise: position-DEBIASED treatment preference  (+ = LGBTQ candidate favored head-to-head)\n"
                 f"[raw A/B choice is {frac_A:.0%} position-A -> discrete choice uninformative; using logit debiasing]",
                 fontsize=10)
    x = range(len(conds))
    ax.bar(x, means, yerr=err, color=["#2ca02c" if m > 0 else "#d62728" for m in means],
           edgecolor="black", capsize=3, zorder=3)
    ax.axhline(0, color="black", linewidth=0.9, label="no preference (0)", zorder=2)
    ax.set_xticks(list(x)); ax.set_xticklabels([c.replace("load", "L") for c in conds], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("debiased preference T (logit; + favors treatment)", fontsize=9)
    ax.set_xlabel("identity condition", fontsize=9); ax.legend(fontsize=8, loc="best")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.26)  # NOT tight_layout
    path = os.path.join(fig_dir, "pairwise_choice.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path

def summary_heatmap(per_cond, fig_dir):
    """Heatmap of effect in FLOOR UNITS (mean Δ / floor) — comparable across DVs."""
    pcs = [pc for pc, (floor, bl) in per_cond.items() if floor]
    loads = sorted({l for pc in pcs for l in per_cond[pc][1]}, key=lambda k: (isinstance(k, str), k))
    if not pcs or not loads:
        return None
    import numpy as np
    M = np.full((len(pcs), len(loads)), np.nan)
    for i, pc in enumerate(pcs):
        floor, bl = per_cond[pc]
        for j, l in enumerate(loads):
            if l in bl and floor:
                M[i, j] = bl[l]["mean"] / floor
    fig, ax = plt.subplots(figsize=(1.3 * len(loads) + 3, 0.6 * len(pcs) + 2.5))
    vmax = max(2.0, np.nanmax(np.abs(M)) if np.isfinite(M).any() else 2.0)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(loads))); ax.set_xticklabels([f"L{l}" for l in loads], fontsize=8)
    ax.set_yticks(range(len(pcs))); ax.set_yticklabels(pcs, fontsize=8)
    for i in range(len(pcs)):
        for j in range(len(loads)):
            if math.isfinite(M[i, j]):
                out = abs(M[i, j]) > 1.0
                ax.text(j, i, f"{M[i,j]:+.1f}" + ("*" if out else ""), ha="center", va="center",
                        fontsize=8, fontweight="bold" if out else "normal",
                        color="white" if abs(M[i, j]) > vmax * 0.6 else "black")
    ax.set_title("Identity effect in NOISE-FLOOR UNITS  (mean Δ / floor; |value|>1 and * = exceeds floor)", fontsize=10)
    ax.set_xlabel("identity load", fontsize=9); ax.set_ylabel("prompt condition", fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label("control − treatment, in floor units", fontsize=8)
    fig.subplots_adjust(left=0.30, right=1.02, top=0.90, bottom=0.12)  # NOT tight_layout
    path = os.path.join(fig_dir, "summary_heatmap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path




FACTOR_KEY = {"explicitness": "identity_description_mode", "salience": "signal_salience_level", "location": "resume_location_level"}
DOSE_ORDER = {"explicitness": ["organization_name_only", "identity_description_subtle", "identity_description_explicit", "strongly_explicit_identity"],
              "salience": ["low", "moderate", "high", "leadership"],
              "location": ["bottom_section", "mid_resume", "leadership_section", "experience_embedded"]}


def _money(r):
    return r.get("modal_offer_usd") if r.get("output_type") in ("salary_increment", "bonus_increment") else r.get("parsed_score")


def dose_figure(rows, factor, fig_dir):
    """Effect by a single dose factor (channel held fixed) per outcome, with floor band
    and a monotonicity flag. Salary/bonus use the realistic $5k modal offer."""
    key = FACTOR_KEY[factor]
    pcs = sorted({r["prompt_condition_id"] for r in rows if r.get("output_type") != "pairwise_AB"})
    if not pcs:
        return None
    fig, axes = plt.subplots(1, len(pcs), figsize=(4.2 * len(pcs), 4.6))
    if len(pcs) == 1:
        axes = [axes]
    for ax, pc in zip(axes, pcs):
        groups = collections.defaultdict(dict)
        for r in rows:
            if r["prompt_condition_id"] != pc or r.get("output_type") == "pairwise_AB":
                continue
            groups[r["paired_example_id"]][r["treatment_or_control"]] = r
        bylevel, floorlevel = collections.defaultdict(list), collections.defaultdict(list)
        for arms in groups.values():
            if "control" not in arms or "treatment" not in arms:
                continue
            lv = arms["control"].get(key)
            cv, tv = _money(arms["control"]), _money(arms["treatment"])
            if cv is not None and tv is not None:
                bylevel[lv].append(cv - tv)
            if "neutral" in arms and _money(arms["neutral"]) is not None:
                floorlevel[lv].append(cv - _money(arms["neutral"]))
        levels = [l for l in DOSE_ORDER.get(factor, []) if l in bylevel] or sorted(bylevel, key=str)
        means = [float(np.mean(bylevel[l])) for l in levels]
        err = [1.96 * float(np.std(bylevel[l], ddof=1)) / np.sqrt(len(bylevel[l])) if len(bylevel[l]) > 1 else 0 for l in levels]
        floor = float(np.mean([abs(np.mean(floorlevel[l])) for l in levels if floorlevel[l]])) if any(floorlevel.values()) else None
        ax.bar(range(len(levels)), means, yerr=err, color="#1f77b4", edgecolor="black", capsize=3, zorder=3)
        if floor:
            ax.axhspan(-floor, floor, color="#999999", alpha=0.22, zorder=1, label=f"floor ±{floor:.3g}")
        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.set_title(pc, fontsize=8)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l).replace("identity_description_", "").replace("_", " ") for l in levels], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("control − treatment", fontsize=8)
        mono = len(levels) >= 3 and (all(x <= y for x, y in zip(means, means[1:])) or all(x >= y for x, y in zip(means, means[1:])))
        ax.legend(fontsize=6, loc="best", title=("MONOTONIC dose-response" if mono else None))
    fig.suptitle(f"{factor.upper()} dose (single affiliation signal; + = identity-signaled candidate penalized)", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.30, wspace=0.35)  # NOT tight_layout
    path = os.path.join(fig_dir, f"dose_{factor}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = os.path.dirname(os.path.abspath(args.scored))
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.scored)]
    UNITS = {"score_0_100": "0-100", "salary_usd": "USD", "bonus_usd": "USD",
             "salary_increment": "USD", "bonus_increment": "USD", "binary_yes_no": "p(Yes)"}
    single_pcs = sorted({r["prompt_condition_id"] for r in rows if r.get("output_type") != "pairwise_AB"})

    print("=== IDENTITY EFFECT vs NOISE FLOOR (control − treatment; + = identity penalized) ===\n")
    per_cond, summary = {}, {}
    for pc in single_pcs:
        recs = paired_records(rows, pc)
        if not recs:
            continue
        units = UNITS.get(recs and rows and next(r["output_type"] for r in rows if r["prompt_condition_id"] == pc), "")
        floor, by_load = figure_for_condition(pc, recs, units, fig_dir)
        per_cond[pc] = (floor, by_load)
        overall = stats_block([(r["pair_id"], r["gender"], r["ct"]) for r in recs])
        verdict = "OUTSIDE floor" if (floor and abs(overall["mean"]) > floor) else "within floor"
        summary[pc] = {"overall_mean": overall["mean"], "ci": [overall["ci_lo"], overall["ci_hi"]],
                       "floor": floor, "verdict": verdict, "n_pairs": overall["n"]}
        print(f"[{pc}] ({units})  mean Δ={overall['mean']:+.4g} "
              f"[{overall['ci_lo']:+.4g},{overall['ci_hi']:+.4g}] dz={overall['cohens_dz']:+.2f} "
              f"floor=±{floor:.4g} -> {verdict}  (n={overall['n']})" if floor else
              f"[{pc}] ({units})  mean Δ={overall['mean']:+.4g}  (no neutral floor available)")
        for l in sorted(by_load, key=lambda k: (isinstance(k, str), k)):
            st = by_load[l]
            mark = "*OUT" if (floor and abs(st["mean"]) > floor) else ""
            print(f"     load {l}: Δ={st['mean']:+.4g} [{st['ci_lo']:+.4g},{st['ci_hi']:+.4g}] n={st['n']} {mark}")

    pw_path = pairwise_figure(rows, fig_dir)
    for _f in ('explicitness', 'salience', 'location'):
        if len({r.get(FACTOR_KEY[_f]) for r in rows if r.get('output_type') != 'pairwise_AB'}) > 1:
            p_ = dose_figure(rows, _f, fig_dir)
            if p_: print(f'dose figure ({_f}) -> {p_}')
    hm_path = summary_heatmap(per_cond, fig_dir)
    out = args.out or os.path.join(run_dir, "analysis_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nfigures -> {fig_dir}/  (effect_*.png" + (", pairwise_choice.png" if pw_path else "")
          + (", summary_heatmap.png" if hm_path else "") + ")")
    print(f"summary -> {out}")
    print("NOTE: read magnitudes vs the noise floor, not p-values (deterministic+paired).")


if __name__ == "__main__":
    main()
