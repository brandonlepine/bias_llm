#!/usr/bin/env python3
"""Predicate-vs-identity decomposition of the WinoQueer head circuit (CPU; post-processing).

Tests the hypothesis: is the BIAS CONTENT (predicate category, e.g. SEXUAL_MORAL_THREAT) written by
a shared set of heads regardless of WHICH identity it targets, while the identity signal lives
elsewhere?  If so, a head's WRITE effect across the identity×predicate cells should vary more with
the predicate than with the identity.

Consumes the already-saved `segmented_head_stats__idpred.csv` (group = "identity|predicate", with
per-(layer,head) write/read/nec/n/qualifies) and the pooled head ranking for the magnitude gate.

Figures (publication-styled):
  1. factor-tuning scatter — per gated head, eta^2(predicate) vs eta^2(identity): is each head
     tuned to the stereotype TYPE, the identity, or both?
  2. predicate x predicate head-Jaccard — do different stereotype types use distinct WRITE heads?
  3. within-predicate cross-identity sharing vs the pooled baseline — does fixing the predicate make
     identities share more heads (⇒ predicate is the shared organizing axis)?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winoqueer_identity_taxonomy import (  # noqa: E402
    MIN_CELL, SPECIFIC_IDENTITIES, PREDICATE_CATEGORIES, pub_style, jaccard, bootstrap_diff_ci,
)


def load_idpred(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path)
    s[["identity", "predicate"]] = s["group"].str.split("|", n=1, expand=True)
    return s


def weighted_eta2(values: np.ndarray, ns: np.ndarray, levels: np.ndarray) -> float:
    """One-way weighted eta^2 = SS_between(levels) / SS_total. NaN if undefined / <2 levels."""
    w = ns.astype(float)
    if w.sum() <= 0 or len(np.unique(levels)) < 2:
        return float("nan")
    gm = float((values * w).sum() / w.sum())
    sst = float((w * (values - gm) ** 2).sum())
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for lv in np.unique(levels):
        m = levels == lv
        nl = w[m].sum()
        ml = (values[m] * w[m]).sum() / nl
        ssb += nl * (ml - gm) ** 2
    return float(ssb / sst)


def factor_tuning(qs: pd.DataFrame, gate: set, min_cells: int = 6) -> pd.DataFrame:
    """Per gated head: eta^2 of WRITE explained by predicate vs by identity, over its qualifying
    specific-identity cells (needs >=2 identities AND >=2 predicates, >=min_cells cells)."""
    spec = qs[qs["identity"].isin(SPECIFIC_IDENTITIES)]
    rows = []
    for (L, H), g in spec.groupby(["layer", "head"]):
        if gate and (int(L), int(H)) not in gate:
            continue
        if len(g) < min_cells or g["identity"].nunique() < 2 or g["predicate"].nunique() < 2:
            continue
        v, n = g["write"].to_numpy(), g["n"].to_numpy()
        rows.append({
            "layer": int(L), "head": int(H), "n_cells": len(g),
            "eta2_predicate": weighted_eta2(v, n, g["predicate"].to_numpy()),
            "eta2_identity": weighted_eta2(v, n, g["identity"].to_numpy()),
            "mean_abs_write": float(np.average(np.abs(v), weights=n)),
        })
    return pd.DataFrame(rows)


def plot_factor_tuning(ft: pd.DataFrame, out_dir: Path):
    if ft.empty:
        return
    d = ft.dropna(subset=["eta2_predicate", "eta2_identity"])
    pub_style()
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    sizes = 25 + 320 * (d["mean_abs_write"] / (d["mean_abs_write"].max() + 1e-9))
    sc = ax.scatter(d["eta2_identity"], d["eta2_predicate"], s=sizes, c=d["layer"], cmap="viridis",
                    alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    lim = max(float(d[["eta2_identity", "eta2_predicate"]].to_numpy().max()), 0.1) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#999", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#2c7fb8", alpha=0.05)
    ax.fill_between([0, lim], [0, 0], [0, lim], color="#c0392b", alpha=0.05)
    ax.text(0.04 * lim, 0.95 * lim, "predicate-tuned\n(bias CONTENT drives the head)", fontsize=9, va="top", color="#1a5276", fontweight="bold")
    ax.text(0.62 * lim, 0.06 * lim, "identity-tuned\n(WHO drives the head)", fontsize=9, color="#922b21", fontweight="bold")
    for _, r in d.sort_values("mean_abs_write", ascending=False).head(12).iterrows():
        ax.annotate(f"L{int(r['layer'])}H{int(r['head'])}", (r["eta2_identity"], r["eta2_predicate"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=7, fontweight="bold")
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("layer")
    ax.set_xlabel("η²(identity)  →  variance of the head's WRITE effect explained by WHICH identity")
    ax.set_ylabel("η²(predicate)  →  variance explained by the STEREOTYPE TYPE")
    ax.set_title("Is each head tuned to the bias content or to the identity?\n"
                 "point size = head magnitude · above diagonal = predicate-dominant")
    fig.tight_layout(); fig.savefig(out_dir / "segmented_predicate_factor_tuning.png"); plt.close(fig)


def predicate_circuit_jaccard(qs: pd.DataFrame, top_k: int, out_dir: Path):
    """Pool WRITE across specific identities within each predicate -> per-predicate top heads ->
    predicate x predicate Jaccard (do stereotype types use distinct heads?)."""
    spec = qs[qs["identity"].isin(SPECIFIC_IDENTITIES)].copy()
    spec["wn"] = spec["write"] * spec["n"]
    pooled = spec.groupby(["predicate", "layer", "head"]).agg(wn=("wn", "sum"), n=("n", "sum")).reset_index()
    pooled["write"] = pooled["wn"] / pooled["n"]
    preds = [p for p in PREDICATE_CATEGORIES if p in set(pooled["predicate"])]
    topsets = {}
    for p in preds:
        d = pooled[pooled["predicate"] == p].sort_values("write", ascending=False).head(top_k)
        topsets[p] = set((int(r.layer), int(r.head)) for r in d.itertuples())
    preds = [p for p in preds if topsets.get(p)]
    if len(preds) < 2:
        return None
    J = pd.DataFrame(index=preds, columns=preds, dtype=float)
    for a in preds:
        for b in preds:
            J.loc[a, b] = jaccard(topsets[a], topsets[b])
    J.to_csv(out_dir / "segmented_predicate_circuit_jaccard.csv")
    M = J.to_numpy(float); off = M.copy(); np.fill_diagonal(off, np.nan)
    short = [p.replace("_", " ").title()[:22] for p in preds]
    pub_style()
    fig, ax = plt.subplots(figsize=(1.0 * len(preds) + 4, 1.0 * len(preds) + 3.2))
    im = ax.imshow(off, cmap="magma", vmin=0, vmax=max(float(np.nanmax(off)), 0.05))
    for i in range(len(preds)):
        for j in range(len(preds)):
            ax.text(j, i, "•" if i == j else f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="#bbb" if i == j else
                    ("white" if M[i, j] < 0.6 * max(np.nanmax(off), 0.05) else "#111"))
    ax.set_xticks(range(len(preds))); ax.set_xticklabels(short, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(preds))); ax.set_yticklabels(short, fontsize=8)
    ax.set_title(f"Do different stereotype TYPES use the same heads?\n"
                 f"Jaccard of each predicate's top-{top_k} WRITE heads (pooled over identities)", fontsize=11.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("Jaccard overlap", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_predicate_circuit_jaccard.png"); plt.close(fig)
    return J


def within_predicate_sharing(qs: pd.DataFrame, top_k: int, pooled_baseline: float, out_dir: Path):
    """For each predicate, mean cross-identity Jaccard of top WRITE heads (specific identities with a
    qualifying cell). If > the identity-pooled baseline, fixing the predicate makes identities share more."""
    spec = qs[qs["identity"].isin(SPECIFIC_IDENTITIES)]
    rows = []
    for p in [p for p in PREDICATE_CATEGORIES if p in set(spec["predicate"])]:
        dp = spec[spec["predicate"] == p]
        ids = sorted(dp["identity"].unique())
        if len(ids) < 2:
            continue
        sets = {i: set((int(r.layer), int(r.head)) for r in
                       dp[dp["identity"] == i].sort_values("write", ascending=False).head(top_k).itertuples()) for i in ids}
        js = [jaccard(sets[a], sets[b]) for k, a in enumerate(ids) for b in ids[k + 1:]]
        rows.append({"predicate": p, "n_identities": len(ids), "mean_cross_identity_jaccard": float(np.nanmean(js))})
    out = pd.DataFrame(rows).sort_values("mean_cross_identity_jaccard", ascending=False)
    out.to_csv(out_dir / "segmented_within_predicate_sharing.csv", index=False)
    if out.empty:
        return out
    pub_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    short = [p.replace("_", " ").title() for p in out["predicate"]]
    ax.bar(range(len(out)), out["mean_cross_identity_jaccard"], color="#2c7fb8", alpha=0.88)
    ax.axhline(pooled_baseline, color="#c0392b", ls="--", lw=1.6,
               label=f"pooled-across-predicates baseline = {pooled_baseline:.3f}")
    ax.set_xticks(range(len(out))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(f"mean cross-identity Jaccard of top-{top_k} WRITE heads")
    ax.set_title("With the stereotype type held fixed, do identities share more heads?\n"
                 "bars above the red line ⇒ that predicate is written by a shared, identity-general circuit")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_within_predicate_sharing.png"); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Predicate-vs-identity decomposition of the WinoQueer head circuit.")
    ap.add_argument("--idpred_stats", type=Path, required=True, help="segmented_head_stats__idpred.csv")
    ap.add_argument("--pooled_ranking", type=Path, required=True, help="segmented_pooled_head_ranking.csv (for the gate)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--gate_frac", type=float, default=0.10)
    ap.add_argument("--gate_min_k", type=int, default=20)
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--identity_baseline", type=float, default=None,
                    help="pooled cross-identity Jaccard from the identity-level analysis (for the reference line)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    s = load_idpred(args.idpred_stats)
    qs = s[s["qualifies"]].copy()
    pooled = pd.read_csv(args.pooled_ranking)
    k = max(args.gate_min_k, int(round(len(pooled) * args.gate_frac)))
    gate = set((int(r.layer), int(r.head)) for r in
               pooled.reindex(pooled["write"].abs().sort_values(ascending=False).index).head(k).itertuples())
    print(f"Qualifying idpred head-cells: {len(qs)} | gated heads: {len(gate)} | "
          f"identities: {sorted(qs['identity'].unique())}")

    ft = factor_tuning(qs, gate)
    ft.to_csv(args.out_dir / "segmented_predicate_factor_tuning.csv", index=False)
    plot_factor_tuning(ft, args.out_dir)
    predicate_circuit_jaccard(qs, args.top_k, args.out_dir)

    # baseline: the pooled cross-identity Jaccard (computed across all predicates). If not given,
    # estimate it from the identity-pooled top heads within this gate.
    baseline = args.identity_baseline
    if baseline is None:
        spec = qs[qs["identity"].isin(SPECIFIC_IDENTITIES)].copy()
        spec["wn"] = spec["write"] * spec["n"]
        idp = spec.groupby(["identity", "layer", "head"]).agg(wn=("wn", "sum"), n=("n", "sum")).reset_index()
        idp["write"] = idp["wn"] / idp["n"]
        ids = sorted(idp["identity"].unique())
        sets = {i: set((int(r.layer), int(r.head)) for r in
                       idp[idp["identity"] == i].sort_values("write", ascending=False).head(args.top_k).itertuples()) for i in ids}
        js = [jaccard(sets[a], sets[b]) for k2, a in enumerate(ids) for b in ids[k2 + 1:]]
        baseline = float(np.nanmean(js))
    wp = within_predicate_sharing(qs, args.top_k, baseline, args.out_dir)

    print("\n=== Predicate vs identity tuning (gated heads) ===")
    if not ft.empty:
        d = ft.dropna(subset=["eta2_predicate", "eta2_identity"])
        diff = bootstrap_diff_ci(d["eta2_predicate"].to_numpy(), d["eta2_identity"].to_numpy())
        print(f"heads analyzed: {len(d)} | mean η²(predicate)={d['eta2_predicate'].mean():.3f}  "
              f"η²(identity)={d['eta2_identity'].mean():.3f}")
        print(f"η²(predicate) − η²(identity) = {diff[0]:+.3f}  CI[{diff[1]:+.3f}, {diff[2]:+.3f}]  "
              f"({'predicate-dominant' if diff[0] > 0 else 'identity-dominant'})")
        print(f"predicate-dominant heads: {int((d['eta2_predicate'] > d['eta2_identity']).sum())}/{len(d)}")
    if wp is not None and not wp.empty:
        print(f"\nWithin-predicate cross-identity sharing (baseline pooled = {baseline:.3f}):")
        print(wp.to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
