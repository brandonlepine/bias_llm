#!/usr/bin/env python3
"""Pre-interpretation diagnostics for a scored run: composition/sample-size tables,
channel-level (vs load-level) effects + a channel regression with collinearity
checks, exact-composition means, outliers, and position-debiased pairwise choice.

Run from resume-eval/:
  python -m new_schemas.benchgen.diagnostics --scored new_schemas/runs/<...>/scored.jsonl
"""
import argparse, collections, json, os
import numpy as np
import pandas as pd

CHANNELS = ["affiliation", "conference", "scholarship", "leadership", "volunteer", "presentation"]
OUTCOMES = ["next_round_score_0_100", "offer_score_0_100", "salary_offer_within_band",
            "signing_bonus_amount", "signing_bonus_yes_no"]


def load_deltas(scored):
    """One row per (paired group, prompt condition): control-treatment delta + covariates."""
    g = collections.defaultdict(dict)
    pw = []
    for line in open(scored):
        r = json.loads(line)
        if r.get("output_type") == "pairwise_AB":
            pw.append(r); continue
        g[(r["paired_example_id"], r["prompt_condition_id"])][r["treatment_or_control"]] = r
    rows = []
    for (pid, pc), arms in g.items():
        if "control" not in arms or "treatment" not in arms:
            continue
        c, t = arms["control"], arms["treatment"]
        if c.get("parsed_score") is None or t.get("parsed_score") is None:
            continue
        chans = set(c.get("signal_channels") or [])
        rec = {"pair_id": pid, "prompt_condition": pc, "delta": c["parsed_score"] - t["parsed_score"],
               "identity_condition": c["identity_signal_condition_id"], "identity_load": c["identity_load"],
               "qual": c["qualification_profile_id"], "job": c["job_id"], "gender": c.get("perceived_gender"),
               "n_channels": len(chans)}
        for ch in CHANNELS:
            rec[ch] = int(ch in chans)
        rows.append(rec)
    return pd.DataFrame(rows), pw


def ols(X, y):
    """Return (beta, se, tvals, rank, ncols). Handles rank-deficiency via pinv."""
    XtX = X.T @ X
    rank = np.linalg.matrix_rank(XtX)
    beta = np.linalg.pinv(X) @ y
    resid = y - X @ beta
    dof = max(len(y) - rank, 1)
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(XtX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return beta, se, t, rank, X.shape[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True)
    args = ap.parse_args()
    run_dir = os.path.dirname(os.path.abspath(args.scored))
    df, pw = load_deltas(args.scored)

    print("="*70, "\n1. COMPOSITION + SAMPLE SIZES\n", "="*70, sep="")
    comp = df.drop_duplicates("identity_condition").set_index("identity_condition")
    for cond in sorted(df["identity_condition"].unique(), key=lambda c: (df[df.identity_condition==c]["identity_load"].iloc[0], c)):
        sub = df[df.identity_condition == cond]
        chans = [c for c in CHANNELS if sub[c].iloc[0] == 1]
        print(f"  L{sub['identity_load'].iloc[0]} {cond:<30} channels={'+'.join(chans) or 'none':<45} "
              f"pairs/outcome={len(sub)//df['prompt_condition'].nunique()}")
    print("\ndf.groupby('identity_load').size() [rows = pair x outcome]:")
    print(df.groupby("identity_load").size().to_string())
    print("\ndf.groupby(['identity_load','identity_condition']).size():")
    print(df.groupby(["identity_load", "identity_condition"]).size().to_string())
    print("\nunique signal compositions per load:")
    print(df.groupby("identity_load")["identity_condition"].nunique().to_string())
    print("\n[KEY] each load>=2 is a SINGLE cumulative composition -> 'load' and 'composition' are\n"
          "      nearly 1:1, and channels are NESTED (each load adds a channel), NOT independently\n"
          "      varied. Channel main-effects/interactions are therefore CONFOUNDED with load here.")

    print("\n" + "="*70, "\n3. MEAN EFFECT BY EXACT COMPOSITION (not load)\n", "="*70, sep="")
    for pc in OUTCOMES:
        d = df[df.prompt_condition == pc]
        if d.empty: continue
        print(f"\n[{pc}]  (control - treatment; + = identity penalized)")
        agg = d.groupby("identity_condition")["delta"].agg(["mean", "sem", "count"])
        agg = agg.reindex(sorted(agg.index, key=lambda c: (d[d.identity_condition==c]["identity_load"].iloc[0], c)))
        for cond, row in agg.iterrows():
            print(f"    {cond:<30} mean={row['mean']:+10.3f}  sem={row['sem']:7.3f}  n={int(row['count'])}")

    print("\n" + "="*70, "\n2. CHANNEL REGRESSION  (delta ~ channels [+ interactions])\n", "="*70, sep="")
    inter = [("affiliation", "conference"), ("affiliation", "scholarship"), ("conference", "scholarship")]
    for pc in OUTCOMES:
        d = df[df.prompt_condition == pc].copy()
        if d.empty: continue
        active = [c for c in CHANNELS if d[c].sum() > 0]   # presentation absent in pilot_v2
        cols = ["intercept"] + active + [f"{a}:{b}" for a, b in inter if a in active and b in active]
        Xd = {"intercept": np.ones(len(d))}
        for c in active: Xd[c] = d[c].values.astype(float)
        for a, b in inter:
            if a in active and b in active: Xd[f"{a}:{b}"] = (d[a]*d[b]).values.astype(float)
        X = np.column_stack([Xd[c] for c in cols]); y = d["delta"].values.astype(float)
        beta, se, t, rank, ncol = ols(X, y)
        print(f"\n[{pc}]  n={len(d)}  design rank={rank}/{ncol}"
              + ("  *** RANK-DEFICIENT: channels nested in load, coefficients not separately identified" if rank < ncol else ""))
        for name, b, s, tv in zip(cols, beta, se, t):
            print(f"    {name:<26} beta={b:+10.3f}  se={s:8.3f}  t={tv:+6.2f}")

    print("\n" + "="*70, "\n4. OUTLIERS (top 8 +/- per outcome)\n", "="*70, sep="")
    for pc in OUTCOMES:
        d = df[df.prompt_condition == pc].sort_values("delta")
        if d.empty: continue
        print(f"\n[{pc}] most NEGATIVE (treatment favored) / most POSITIVE (treatment penalized):")
        for _, r in pd.concat([d.head(4), d.tail(4)]).iterrows():
            chans = "+".join(c for c in CHANNELS if r[c]) or "none"
            print(f"    Δ={r['delta']:+10.2f}  {r['identity_condition']:<28} {r['qual']:<26} {chans}")

    print("\n" + "="*70, "\n5. PAIRWISE DEBUG + POSITION-DEBIASED EFFECT\n", "="*70, sep="")
    g = collections.defaultdict(dict); meta = {}
    chosenA = sum(r["chosen_candidate"] == "A" for r in pw)
    for r in pw:
        g[r["paired_example_id"]][r["candidate_a_variant_type"]] = r["logit_A_minus_logit_B"]
        meta[r["paired_example_id"]] = r["identity_signal_condition_id"]
    print(f"  raw discrete choice: chose A in {chosenA}/{len(pw)} ({chosenA/len(pw):.1%}) -> TOTAL position-A bias")
    print("  => discrete P(treatment)=0.5 and order-effect=1.0 are MECHANICAL (uninformative).")
    print("  FIX: position-debiased preference T = (logit_AB|treat=A - logit_AB|treat=B)/2  (+ = treatment FAVORED)\n")
    byc = collections.defaultdict(list)
    for pid, d in g.items():
        if "treatment" in d and "control" in d:
            byc[meta[pid]].append((d["treatment"] - d["control"]) / 2)
    for cond in sorted(byc, key=lambda c: c):
        v = np.array(byc[cond])
        se = v.std(ddof=1)/np.sqrt(len(v))
        print(f"    {cond:<30} T={v.mean():+.4f}  se={se:.4f}  n={len(v)}  ({'treatment penalized' if v.mean()<0 else 'treatment favored'})")

    df.to_csv(os.path.join(run_dir, "diagnostics_deltas.csv"), index=False)
    print(f"\ntidy per-pair deltas -> {os.path.join(run_dir, 'diagnostics_deltas.csv')}")


if __name__ == "__main__":
    main()
