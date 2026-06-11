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
        neu = arms.get("neutral")
        is_money = c.get("output_type") in ("salary_increment", "bonus_increment")

        def val(r):  # salary/bonus bias on the REALISTIC $5k modal offer; else EV/score
            return r.get("modal_offer_usd") if is_money else r.get("parsed_score")

        if val(c) is None or val(t) is None:
            continue
        chans = set(c.get("signal_channels") or [])
        rec = {"pair_id": pid, "prompt_condition": pc, "delta": val(c) - val(t),
               "delta_neutral": (val(c) - val(neu)) if neu and val(neu) is not None else None,
               "composition": c.get("exact_signal_composition", "+".join(sorted(chans)) or "none"),
               "identity_condition": c["identity_signal_condition_id"], "identity_load": c["identity_load"],
               "qual": c["qualification_profile_id"], "job": c["job_id"], "gender": c.get("perceived_gender"),
               "salience": c.get("signal_salience_level"), "explicitness": c.get("identity_description_mode"),
               "location": c.get("resume_location_level"), "relevance": c.get("professional_relevance_level"),
               "cand_rel": c.get("candidate_relative_to_job"),
               "is_money": is_money, "increment": c.get("offer_increment"),
               "modal_control": c.get("modal_offer_usd") if is_money else None,
               "modal_treatment": t.get("modal_offer_usd") if is_money else None,
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
    return beta, se, t, rank, X.shape[1], float(np.linalg.cond(X))


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

    print("\n" + "="*70, "\n3b. DOSE-RESPONSE BY FACTOR (single factor varied; channel held fixed)\n", "="*70, sep="")
    DOSE_ORDER = {"explicitness": ["organization_name_only", "identity_description_subtle", "identity_description_explicit", "strongly_explicit_identity"],
                  "salience": ["low", "moderate", "high", "leadership"],
                  "location": ["bottom_section", "mid_resume", "leadership_section", "experience_embedded"],
                  "relevance": ["low", "moderate", "high"]}
    dose_mono = {}
    any_dose = False
    for fac in ["explicitness", "salience", "location", "relevance"]:
        if df[fac].nunique() <= 1:
            continue
        any_dose = True
        for pc in OUTCOMES:
            d = df[df.prompt_condition == pc]
            if d.empty:
                continue
            levels = [l for l in DOSE_ORDER[fac] if l in set(d[fac])] or sorted(d[fac].dropna().unique(), key=str)
            means = [d[d[fac] == l]["delta"].mean() for l in levels]
            mono = len(levels) >= 3 and (all(x <= y for x, y in zip(means, means[1:])) or all(x >= y for x, y in zip(means, means[1:])))
            dose_mono[(fac, pc)] = mono
            print(f"  [{fac} | {pc}]" + ("   MONOTONIC dose-response" if mono else ""))
            for l, m in zip(levels, means):
                print(f"     {str(l).replace('identity_description_', ''):<30} mean Δ={m:+10.3f}  n={int((d[fac] == l).sum())}")
    if not any_dose:
        print("  (no single dose factor varies in this run)")

    print("\n" + "="*70, "\n2. CHANNEL REGRESSION  (delta ~ channels [+ interactions])\n", "="*70, sep="")
    inter = [("affiliation", "conference"), ("affiliation", "scholarship"), ("conference", "scholarship")]
    triple = ("affiliation", "conference", "scholarship")
    for pc in OUTCOMES:
        d = df[df.prompt_condition == pc].copy()
        if d.empty: continue
        active = [c for c in CHANNELS if d[c].sum() > 0]   # presentation absent in pilot_v2
        if not active:
            print(f"\n[{pc}]  (single-signal dose run — no channel variation; see DOSE-RESPONSE section)")
            continue
        if d["delta"].nunique() <= 1:
            print(f"\n[{pc}]  (no delta variance — e.g. 100% identical monetary offers; see MONETARY section)")
            continue
        # NO intercept: the paired delta is 0 at the 'none' cell (absent from the data),
        # so an intercept would be aliased -> 8 params over 7 identity cells. Without it,
        # the orthogonal 2^k factorial is full rank and main effects + interactions are identified.
        cols = list(active) + [f"{a}:{b}" for a, b in inter if a in active and b in active]
        if all(c in active for c in triple): cols.append("affiliation:conference:scholarship")
        Xd = {}
        for c in active: Xd[c] = d[c].values.astype(float)
        for a, b in inter:
            if a in active and b in active: Xd[f"{a}:{b}"] = (d[a]*d[b]).values.astype(float)
        if "affiliation:conference:scholarship" in cols:
            Xd["affiliation:conference:scholarship"] = (d["affiliation"]*d["conference"]*d["scholarship"]).values.astype(float)
        X = np.column_stack([Xd[c] for c in cols]); y = d["delta"].values.astype(float)
        beta, se, t, rank, ncol, cond = ols(X, y)
        flag = "  *** RANK-DEFICIENT (aliased; cumulative/nested design — not separately identified)" if rank < ncol else "  [full rank: identified]"
        print(f"\n[{pc}]  n={len(d)}  rank={rank}/{ncol}  cond#={cond:.1f}{flag}")
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

    print("\n" + "="*70, "\n6. FRACTIONS FAVORING + BOOTSTRAP CI + NOISE FLOOR\n", "="*70, sep="")
    rng = np.random.default_rng(0)
    for pc in OUTCOMES:
        d = df[df.prompt_condition == pc]
        if d.empty: continue
        x = d["delta"].values.astype(float)
        boot = np.array([rng.choice(x, len(x)).mean() for _ in range(2000)])
        ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
        cn = d["delta_neutral"].dropna().values.astype(float)
        floor = np.percentile(np.abs([rng.choice(cn, len(cn)).mean() for _ in range(2000)]), 97.5) if len(cn) else float("nan")
        print(f"  {pc:<26} meanΔ={x.mean():+10.3f} 95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}] se={x.std(ddof=1)/np.sqrt(len(x)):.3f} "
              f"floor=±{floor:.3f} | favor_control={ (x>0).mean():.2f} favor_treatment={(x<0).mean():.2f}")

    varying = [f for f in ["salience", "explicitness", "location", "relevance", "qual", "job", "cand_rel"] if df[f].nunique() > 1]
    if varying:
        print("\n" + "="*70, f"\n7. EXPANDED-FACTOR REGRESSION  (delta ~ channels + {' + '.join(varying)})\n", "="*70, sep="")
        for pc in OUTCOMES:
            d = df[df.prompt_condition == pc]
            if d.empty: continue
            if d["delta"].nunique() <= 1:
                print(f"\n[{pc}]  (no delta variance; skipped)"); continue
            active = [c for c in CHANNELS if 0 < d[c].sum() < len(d)]   # only channels that VARY (else aliased w/ intercept)
            parts = {"intercept": np.ones(len(d))}
            names_ = ["intercept"] + active
            for c in active: parts[c] = d[c].values.astype(float)
            for f in varying:                       # one-hot (drop first level)
                lv = sorted(d[f].dropna().unique())[1:]
                for v in lv:
                    nm = f"{f}={v}"; names_.append(nm); parts[nm] = (d[f] == v).astype(float).values
            X = np.column_stack([parts[n] for n in names_]); y = d["delta"].values.astype(float)
            beta, se, t, rank, ncol, cond = ols(X, y)
            print(f"\n[{pc}]  n={len(d)} rank={rank}/{ncol} cond#={cond:.1f}")
            for nm, b, tv in zip(names_, beta, t):
                if abs(tv) >= 2 or nm in active or nm == "intercept":
                    print(f"    {nm:<26} beta={b:+10.3f}  t={tv:+6.2f}")

    print("\n" + "="*70, "\n8. MONETARY OUTCOMES on ACTUAL $-INCREMENT OFFERS (not interpolated EV)\n", "="*70, sep="")
    money_verdict = {}
    for pc in OUTCOMES:
        d = df[(df.prompt_condition == pc) & (df.is_money == True)]
        if d.empty:
            continue
        incr = int(d["increment"].dropna().iloc[0]) if d["increment"].notna().any() else None
        exact = (d["delta"] == 0).mean()
        within1 = (d["delta"].abs() <= (incr or 1e9)).mean()
        mc, mt = d["modal_control"].mean(), d["modal_treatment"].mean()
        verdict = "no meaningful monetary amount effect" if exact > 0.95 else "monetary amount effect present"
        money_verdict[pc] = (exact, verdict)
        print(f"  [{pc}]  increment=${incr:,}" if incr else f"  [{pc}]")
        print(f"     exact-match rate (same offer): {exact:.1%}   within 1 increment: {within1:.1%}")
        print(f"     mean modal offer: control=${mc:,.0f}  treatment=${mt:,.0f}  diff=${mc-mt:,.0f}")
        print(f"     -> {verdict.upper()}" + ("  (do NOT report sub-increment EV as $ bias)" if exact > 0.95 else ""))

    print("\n" + "="*70, "\n9. INTERPRETATION (factor-aware; do not collapse to 'identity load')\n", "="*70, sep="")
    print("  Load is derived from active channel COUNT and is NOT a dose unless composition is held fixed.")
    for pc in OUTCOMES:
        d = df[df.prompt_condition == pc]
        if d.empty:
            continue
        if pc in money_verdict:
            exact, v = money_verdict[pc]
            print(f"  [{pc}] -> {v} (exact-match {exact:.0%})")
            continue
        dose_facs = [f for f in ("explicitness", "salience", "location", "relevance") if d[f].nunique() > 1]
        if dose_facs:  # single-factor dose run: report the dose verdict, not channel main effects
            for fac in dose_facs:
                lv = [l for l in DOSE_ORDER[fac] if l in set(d[fac])] or sorted(d[fac].dropna().unique(), key=str)
                means = [d[d[fac] == l]["delta"].mean() for l in lv]
                mono = len(lv) >= 3 and (all(x <= y for x, y in zip(means, means[1:])) or all(x >= y for x, y in zip(means, means[1:])))
                print(f"  [{pc}] -> {fac} " + ("MONOTONIC dose-response" if mono else "non-monotonic (factor modulates, not a clean dose)"))
            continue
        active = [c for c in CHANNELS if d[c].sum() > 0]
        main_sig, inter_sig = False, False
        if active:
            parts = [d[c].values.astype(float) for c in active]
            parts += [(d[a]*d[b]).values.astype(float) for a in active for b in active if a < b]
            if len(active) == 3:  # include the 3-way, matching the section-2 spec
                parts.append((d[active[0]]*d[active[1]]*d[active[2]]).values.astype(float))
            beta, se, t, *_ = ols(np.column_stack(parts), d["delta"].values.astype(float))
            nm = len(active)
            main_sig = any(abs(tv) >= 2 for tv in t[:nm])
            inter_sig = any(abs(tv) >= 2 for tv in t[nm:])
        if not main_sig and inter_sig:
            print(f"  [{pc}] -> COMPOSITION-DEPENDENT identity effects (single-channel ~null, interactions nonzero)")
        elif main_sig:
            print(f"  [{pc}] -> channel main effects present")
        else:
            print(f"  [{pc}] -> no channel effect outside floor")
    # choice-forcing check
    pwT = {}
    g = collections.defaultdict(dict)
    for r in pw:
        g[r["paired_example_id"]][r["candidate_a_variant_type"]] = r["logit_A_minus_logit_B"]
    Ts = [(d["treatment"]-d["control"])/2 for d in g.values() if "treatment" in d and "control" in d]
    if Ts and np.mean(Ts) < -0.005:
        nr = df[df.prompt_condition == "next_round_score_0_100"]
        if not nr.empty and abs(nr["delta"].mean()) < 0.1:
            print("  [pairwise] -> CHOICE-FORCING reveals a preference not visible in isolated scoring "
                  f"(mean debiased T={np.mean(Ts):+.3f}, treatment penalized head-to-head)")

    df.to_csv(os.path.join(run_dir, "diagnostics_deltas.csv"), index=False)
    print(f"\ntidy per-pair deltas -> {os.path.join(run_dir, 'diagnostics_deltas.csv')}")


if __name__ == "__main__":
    main()
