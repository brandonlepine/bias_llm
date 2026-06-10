#!/usr/bin/env python3
"""Paired analysis of the queer-vs-control hiring pilot.

For each bootstrap pair we compute a WITHIN-PAIR difference (control - queer) on
each dependent variable. Orientation: a POSITIVE difference means the control
(non-queer) resume scored higher == evidence of bias AGAINST the queer-coded
resume. Because the two resumes in a pair are byte-identical except the signal,
the paired difference isolates the causal effect of the queer signal (modulo the
name held fixed within the pair).

Reports per DV: N pairs, mean paired difference, bootstrap 95% CI, paired t-test
and Wilcoxon signed-rank p-values, Cohen's dz, and the share of pairs favoring
control. Also breaks the effect down by the candidate's perceived gender.

Example:
  python -m pilot.analyze
"""
import argparse
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# DVs to test if present. (key, higher_is_more_hireable)
DV_KEYS = [
    ("logit_diff", True),
    ("p_yes", True),
    ("overall_fit", True),
    ("technical_qualifications", True),
    ("relevant_experience", True),
    ("communication_collaboration", True),
    ("hire_probability", True),
]


def load_scores(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_dv(rec, key):
    if key in rec and rec[key] is not None:
        return rec[key]
    r = rec.get("rubric")
    if isinstance(r, dict) and r.get(key) is not None:
        return r[key]
    return None


def paired_diffs(rows, key):
    """Return list of (pair_id, gender, control_minus_queer) for pairs with both."""
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["condition"]] = r
    out = []
    for pid, d in by_pair.items():
        if "queer" not in d or "control" not in d:
            continue
        qv, cv = get_dv(d["queer"], key), get_dv(d["control"], key)
        if qv is None or cv is None:
            continue
        out.append((pid, d["control"].get("gender"), cv - qv))
    return out


def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=0):
    import random
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def stats_block(diffs):
    vals = [d for _, _, d in diffs]
    n = len(vals)
    if n == 0:
        return None
    mean = statistics.mean(vals)
    sd = statistics.pstdev(vals) if n == 1 else statistics.stdev(vals)
    dz = mean / sd if sd > 0 else float("nan")
    lo, hi = bootstrap_ci(vals)
    favor_control = sum(1 for v in vals if v > 0)
    favor_queer = sum(1 for v in vals if v < 0)
    ties = sum(1 for v in vals if v == 0)

    res = dict(n=n, mean=mean, sd=sd, ci_lo=lo, ci_hi=hi, cohens_dz=dz,
               favor_control=favor_control, favor_queer=favor_queer, ties=ties)

    # Inferential tests via scipy if available; else paired-t by hand.
    try:
        from scipy import stats as ss
        res["t_p"] = float(ss.ttest_1samp(vals, 0.0).pvalue) if n > 1 else float("nan")
        if n > 1 and len(set(vals)) > 1:
            res["wilcoxon_p"] = float(ss.wilcoxon(vals).pvalue)
        else:
            res["wilcoxon_p"] = float("nan")
    except Exception:
        if n > 1 and sd > 0:
            t = mean / (sd / math.sqrt(n))
            res["t_stat"] = t
        res["t_p"] = None
        res["wilcoxon_p"] = None
    return res


def fmt(res):
    def g(x): return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    line = (f"n={res['n']:3d}  mean(ctrl-queer)={g(res['mean'])}  "
            f"95%CI=[{g(res['ci_lo'])}, {g(res['ci_hi'])}]  dz={g(res['cohens_dz'])}  "
            f"favor_ctrl={res['favor_control']}/{res['n']}")
    if res.get("t_p") is not None:
        line += f"  t_p={g(res['t_p'])}"
    if res.get("wilcoxon_p") is not None:
        line += f"  wilcoxon_p={g(res['wilcoxon_p'])}"
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=os.path.join(ROOT, "results", "scores.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "summary.json"))
    ap.add_argument("--by-gender", action="store_true", help="also break down by gender")
    args = ap.parse_args()

    rows = load_scores(args.scores)
    print(f"Loaded {len(rows)} resume scores "
          f"({len(set(r['pair_id'] for r in rows))} pairs)\n")
    print("Positive mean => control (non-queer) favored => bias AGAINST queer signal.\n")

    summary = {}
    for key, _ in DV_KEYS:
        diffs = paired_diffs(rows, key)
        if not diffs:
            continue
        res = stats_block(diffs)
        summary[key] = res
        print(f"[{key}]")
        print("  ALL        " + fmt(res))
        if args.by_gender:
            genders = sorted(set(g for _, g, _ in diffs if g))
            for gd in genders:
                sub = [(p, g, v) for (p, g, v) in diffs if g == gd]
                rsub = stats_block(sub)
                summary.setdefault(key + "_by_gender", {})[gd] = rsub
                print(f"  {gd:<10} " + fmt(rsub))
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
