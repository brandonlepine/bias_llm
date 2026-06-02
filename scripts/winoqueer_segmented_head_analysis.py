#!/usr/bin/env python3
"""Identity-segmented analysis of the WinoQueer attention-head circuit (CPU; post-processing).

Joins the per-pair head-PATCHING raw (WRITE = normalized bias_effect; READ = readout→identity
attention, normalized by identity-token count) and head-ABLATION raw (NEC = frac_bias_removed) to
the frozen cohort on `row_id` (the stable key — `pair_id` is positional and unsafe), then segments
by axis / identity / axis×predicate / identity×predicate.

Core questions:
  - Which heads are IDENTITY-SPECIFIC vs SHARED?  (EB-shrunk entropy selectivity over 7 identities)
  - Is identity-location ≠ bias-location?          (READ vs WRITE per identity; Spearman + Jaccard)
  - Are WRITE heads shared while READ heads are specific?  (selectivity(READ) vs selectivity(WRITE))
  - Is the umbrella circuit the union of the specific circuits?  (Jaccard with union-of-specifics)

Rigor: compare NORMALIZED frac (not raw effect) across groups; magnitude-gate to the top heads
before per-group claims; sign-consistency + bootstrap CIs; cross-identity overlap vs a random null.
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
    MIN_CELL, LEVELS, SPECIFIC_IDENTITIES, UMBRELLA, IDENTITY_AXIS, annotate,
    selectivity, jaccard, rbo, bootstrap_ci, bootstrap_diff_ci, random_jaccard_null,
)

CELL_COL = "predicate_label_provisional"

# Publication styling + axis-ordered identities (orientation | gender | umbrella).
IDENTITY_ORDER = ["Asexual", "Bisexual", "Gay", "Lesbian", "Pansexual", "Transgender", "NB", "Queer", "LGBTQ"]
AXIS_COLORS = {"sexual_orientation": "#1f78b4", "gender_identity": "#e31a1c", "umbrella": "#6a3d9a"}
AXIS_SHORT = {"sexual_orientation": "sexual orientation", "gender_identity": "gender identity", "umbrella": "umbrella"}


def _pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444", "axes.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    })


def _ordered_present(values) -> list[str]:
    s = set(values)
    return [i for i in IDENTITY_ORDER if i in s]


def _axis_separators(ids: list[str]) -> list[int]:
    a = [IDENTITY_AXIS[i] for i in ids]
    return [i for i in range(1, len(ids)) if a[i] != a[i - 1]]


def _color_ticklabels(ax, ids, axis="x"):
    labs = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for t, c in zip(labs, ids):
        t.set_color(AXIS_COLORS[IDENTITY_AXIS[c]])
        t.set_fontweight("bold")


# ----------------------------------------------------------------------------- data
def load_long(patching_raw: Path, ablation_raw: Path, cohort: Path) -> pd.DataFrame:
    """Per (row_id, layer, head) long table with WRITE / READ / NEC + axis/identity/predicate."""
    pat = pd.read_csv(patching_raw)
    abl = pd.read_csv(ablation_raw)
    coh = annotate(pd.read_csv(cohort))[["row_id", "axis", "identity", "is_umbrella", CELL_COL]]

    # WRITE = normalized restoration (bias_effect / denom); READ = attn normalized by identity tokens.
    nidt = pat["n_identity_tokens"].clip(lower=1) if "n_identity_tokens" in pat.columns else 1
    pat = pat.assign(write=pat["normalized_restoration"], read=pat["attn_readout_to_identity"] / nidt)
    pat = pat[["row_id", "layer", "head", "write", "read"]]
    abl = abl.assign(nec=abl["frac_bias_removed"])[["row_id", "layer", "head", "nec"]]

    long = pat.merge(abl, on=["row_id", "layer", "head"], how="inner").merge(coh, on="row_id", how="inner")
    return long


def level_group_col(df: pd.DataFrame, level: str) -> pd.Series:
    """Bare group label per row for a segmentation level."""
    if level == "axis":
        return df["axis"]
    if level == "identity":
        return df["identity"]
    if level == "axispred":
        return df["axis"].astype(str) + "|" + df[CELL_COL].astype(str)
    if level == "idpred":
        return df["identity"].astype(str) + "|" + df[CELL_COL].astype(str)
    raise ValueError(level)


def head_group_stats(long: pd.DataFrame, level: str) -> pd.DataFrame:
    """Per (group, layer, head): mean WRITE/READ/NEC, sign-consistency, n, qualifies(n>=MIN_CELL)."""
    df = long.copy()
    df["group"] = level_group_col(df, level)
    g = df.groupby(["group", "layer", "head"], sort=False)
    s = g.agg(
        write=("write", "mean"), read=("read", "mean"), nec=("nec", "mean"),
        write_sign=("write", lambda x: float((x > 0).mean())),
        nec_sign=("nec", lambda x: float((x > 0).mean())),
        n=("write", "size"),
    ).reset_index()
    s["qualifies"] = s["n"] >= MIN_CELL
    return s


def pooled_head_ranking(long: pd.DataFrame) -> pd.DataFrame:
    s = long.groupby(["layer", "head"], sort=False).agg(
        write=("write", "mean"), read=("read", "mean"), nec=("nec", "mean"),
        write_sign=("write", lambda x: float((x > 0).mean())), n=("write", "size"),
    ).reset_index()
    return s.sort_values("write", ascending=False).reset_index(drop=True)


def top_heads(pooled: pd.DataFrame, col: str, frac: float, min_k: int) -> set[tuple[int, int]]:
    """Magnitude gate: the top fraction of heads by |col| (at least min_k)."""
    k = max(min_k, int(round(len(pooled) * frac)))
    top = pooled.reindex(pooled[col].abs().sort_values(ascending=False).index).head(k)
    return set((int(r.layer), int(r.head)) for r in top.itertuples())


# ----------------------------------------------------------------------------- specificity
def _aligned(gm: pd.DataFrame, col: str) -> tuple[list[float], list[float]]:
    vals = [float(gm.loc[i, col]) if i in gm.index else 0.0 for i in SPECIFIC_IDENTITIES]
    ns = [float(gm.loc[i, "n"]) if i in gm.index else 0.0 for i in SPECIFIC_IDENTITIES]
    return vals, ns


def selectivity_table(stats_identity: pd.DataFrame, gate: set) -> pd.DataFrame:
    """Per gated head: EB-shrunk entropy selectivity over the 7 specific identities, WRITE and READ."""
    spec = stats_identity[stats_identity["group"].isin(SPECIFIC_IDENTITIES)]
    rows = []
    for (L, H), g in spec.groupby(["layer", "head"]):
        if gate and (int(L), int(H)) not in gate:
            continue
        gm = g.set_index("group")
        w, nw = _aligned(gm, "write")
        r, nr = _aligned(gm, "read")
        rows.append({
            "layer": int(L), "head": int(H),
            "selectivity_write": selectivity(w, nw), "selectivity_read": selectivity(r, nr),
            "argmax_write": SPECIFIC_IDENTITIES[int(np.argmax(w))],
            "argmax_read": SPECIFIC_IDENTITIES[int(np.argmax(r))],
            "max_write": float(max(w)), "max_read": float(max(r)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("selectivity_write", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- plots
def plot_heads_groups_matrix(stats_identity: pd.DataFrame, gate: set, out_dir: Path, top_n: int = 30):
    cols = _ordered_present(stats_identity["group"])
    df = stats_identity[stats_identity["group"].isin(cols)]
    if df.empty:
        return
    piv = df.pivot_table(index=["layer", "head"], columns="group", values="write").reindex(columns=cols)
    if gate:
        piv = piv[[idx in gate for idx in piv.index]]
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index).head(top_n)
    if piv.empty:
        return
    labels = [f"L{int(l)}H{int(h)}" for l, h in piv.index]
    raw = piv.to_numpy(dtype=float)
    pos = np.clip(raw, 0, None)
    share = pos / np.where(pos.sum(1, keepdims=True) > 0, pos.sum(1, keepdims=True), 1.0)
    seps = _axis_separators(cols)

    _pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4 + len(cols) * 1.25, max(6.2, len(labels) * 0.30)))
    v = max(float(np.nanpercentile(np.abs(raw), 98)), 1e-9)
    smax = max(float(np.nanmax(share)), 1e-9)
    panels = [
        (axes[0], raw, "How strongly each head writes\nthe stereotype, per identity",
         dict(cmap="RdBu_r", vmin=-v, vmax=v),
         "WRITE effect: Δ logP(stereotype) when the head's\nqueer output is injected   (>0 = writes the bias)"),
        (axes[1], share, "Where each head's bias effect\nis concentrated",
         dict(cmap="magma", vmin=0, vmax=smax),
         "within-head share: fraction of a head's total\npositive WRITE effect going to each identity"),
    ]
    for ax, mat, title, kw, cbl in panels:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", **kw)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=50, ha="right")
        _color_ticklabels(ax, cols, "x")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7, family="monospace")
        for s in seps:
            ax.axvline(s - 0.5, color="white", lw=3); ax.axvline(s - 0.5, color="#222", lw=0.9)
        ax.set_title(title, fontsize=11.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label(cbl, fontsize=8); cb.ax.tick_params(labelsize=8)
    fig.suptitle("Which attention heads write each identity's stereotype\n"
                 "rows: top bias-writing heads   ·   columns by axis — "
                 "$\\bf{orientation}$ (blue) · $\\bf{gender}$ (red) · $\\bf{umbrella}$ (purple)",
                 fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_heads_groups_matrix.png"); plt.close(fig)


def plot_selectivity_scatter(seltab: pd.DataFrame, out_dir: Path):
    if seltab.empty:
        return
    _pub_style()
    fig, ax = plt.subplots(figsize=(9.2, 7))
    palette = {ident: plt.cm.tab10(k % 10) for k, ident in enumerate(IDENTITY_ORDER)}
    present = _ordered_present(seltab["argmax_write"])
    for ident in present:
        d = seltab[seltab["argmax_write"] == ident]
        ax.scatter(d["max_write"], d["selectivity_write"], s=80, alpha=0.9,
                   color=palette[ident], edgecolor="white", linewidth=0.7, label=ident, zorder=3)
    for _, r in seltab.head(12).iterrows():
        ax.annotate(f"L{int(r['layer'])}H{int(r['head'])}", (r["max_write"], r["selectivity_write"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8, fontweight="bold", color="#222")
    ax.set_xlabel("MAGNITUDE  →   strongest single-identity WRITE effect  (normalized Δ logP)")
    ax.set_ylabel("SPECIFICITY  →   WRITE selectivity across 7 identities\n(0 = shared by all, 1 = exclusive to one)")
    ax.set_title("Are strong bias-writing heads shared or identity-specific?")
    ax.text(0.985, 0.04, "top-right = strong AND identity-specific", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, style="italic", color="#555",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))
    ax.legend(title="dominant identity", ncol=2, frameon=False, loc="upper right")
    ax.margins(0.06)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_head_selectivity_scatter.png"); plt.close(fig)


def plot_jaccard_matrix(J: pd.DataFrame, metric: str, top_k: int, out_dir: Path, null: dict | None = None):
    ids = _ordered_present(J.index)
    M = J.reindex(index=ids, columns=ids).to_numpy(dtype=float)
    off = M.copy(); np.fill_diagonal(off, np.nan)
    vmax = max(float(np.nanmax(off)), 0.05)
    seps = _axis_separators(ids)
    role = "write the stereotype for" if metric == "write" else "read the identity for"

    _pub_style()
    fig, ax = plt.subplots(figsize=(1.15 * len(ids) + 3.4, 1.05 * len(ids) + 3.0))
    im = ax.imshow(off, cmap="magma", vmin=0, vmax=vmax)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                ax.text(j, i, "•", ha="center", va="center", color="#bbb", fontsize=12)
            else:
                val = M[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="white" if val < 0.62 * vmax else "#111")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=50, ha="right")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids)
    _color_ticklabels(ax, ids, "x"); _color_ticklabels(ax, ids, "y")
    for s in seps:
        ax.axvline(s - 0.5, color="white", lw=2.5); ax.axhline(s - 0.5, color="white", lw=2.5)
    nt = f"   ·   random baseline ≈ {null['null_mean']:.3f}" if null else ""
    ax.set_title(f"Do identities share the same heads that {role} them?\n"
                 f"Jaccard overlap of each identity's top-{top_k} {metric.upper()} heads{nt}", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Jaccard overlap  (fraction of shared top heads)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / f"segmented_head_jaccard_{metric}.png"); plt.close(fig)


def overlap_matrices(stats_identity: pd.DataFrame, top_k: int, out_dir: Path):
    ids = [i for i in SPECIFIC_IDENTITIES if i in set(stats_identity["group"])]
    if len(ids) < 2:
        return None
    pool = stats_identity.groupby(["layer", "head"]).ngroups
    nm, nsd = random_jaccard_null(pool, top_k)
    null = {"null_mean": nm, "null_sd": nsd}

    def topset(metric, ident):
        d = stats_identity[stats_identity["group"] == ident].sort_values(metric, ascending=False).head(top_k)
        return [(int(r.layer), int(r.head)) for r in d.itertuples()]

    for metric in ("write", "read"):
        J = pd.DataFrame(index=ids, columns=ids, dtype=float)
        R = pd.DataFrame(index=ids, columns=ids, dtype=float)
        sets = {i: topset(metric, i) for i in ids}
        for a in ids:
            for b in ids:
                J.loc[a, b] = jaccard(sets[a], sets[b])
                R.loc[a, b] = rbo(sets[a], sets[b])
        J.to_csv(out_dir / f"segmented_head_jaccard_{metric}.csv")
        R.to_csv(out_dir / f"segmented_head_rbo_{metric}.csv")
        plot_jaccard_matrix(J, metric, top_k, out_dir, null)
    return null


def read_vs_write(stats_identity: pd.DataFrame, seltab: pd.DataFrame, top_k: int, out_dir: Path):
    ids = [i for i in SPECIFIC_IDENTITIES if i in set(stats_identity["group"])]
    rows = []
    for ident in ids:
        d = stats_identity[stats_identity["group"] == ident]
        if d.empty:
            continue
        sp = d["read"].corr(d["write"], method="spearman")
        topr = set((int(r.layer), int(r.head)) for r in d.sort_values("read", ascending=False).head(top_k).itertuples())
        topw = set((int(r.layer), int(r.head)) for r in d.sort_values("write", ascending=False).head(top_k).itertuples())
        rows.append({"identity": ident, "spearman_read_write": sp, "jaccard_topread_topwrite": jaccard(topr, topw)})
    rw = pd.DataFrame(rows)
    rw.to_csv(out_dir / "segmented_read_vs_write.csv", index=False)

    diff = (float("nan"),) * 3
    if not seltab.empty:
        diff = bootstrap_diff_ci(seltab["selectivity_read"].to_numpy(), seltab["selectivity_write"].to_numpy())
    if not rw.empty:
        order = _ordered_present(rw["identity"])
        rw = rw.set_index("identity").reindex(order).reset_index()
        _pub_style()
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        x = np.arange(len(rw)); w = 0.38
        ax.bar(x - w / 2, rw["spearman_read_write"], w, color="#1f78b4", label="Spearman(READ, WRITE) over all heads")
        ax.bar(x + w / 2, rw["jaccard_topread_topwrite"], w, color="#e08214",
               label=f"Jaccard(top-{top_k} READ heads, top-{top_k} WRITE heads)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(rw["identity"], rotation=35, ha="right")
        _color_ticklabels(ax, list(rw["identity"]), "x")
        ax.set_ylabel("agreement between a head's READ and WRITE rankings")
        ax.set_title("Identity-reading heads ≠ stereotype-writing heads\n"
                     "low agreement ⇒ the model reads an identity and writes its stereotype with DIFFERENT heads")
        ax.legend(frameon=False, loc="upper right")
        if not np.isnan(diff[0]):
            ax.text(0.015, 0.97,
                    f"reading is more SHARED than writing:\nselectivity(READ) − selectivity(WRITE) = {diff[0]:+.3f}  "
                    f"[{diff[1]:+.3f}, {diff[2]:+.3f}]",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#444",
                    bbox=dict(boxstyle="round", fc="#f6f6f6", ec="#ccc"))
        fig.tight_layout(); fig.savefig(out_dir / "segmented_read_vs_write.png"); plt.close(fig)
    return rw, diff


def umbrella_union_test(stats_identity: pd.DataFrame, k: int, out_dir: Path):
    spec_union = set()
    for grp in SPECIFIC_IDENTITIES:
        d = stats_identity[stats_identity["group"] == grp].sort_values("write", ascending=False)
        spec_union |= set((int(r.layer), int(r.head)) for r in d.head(k).itertuples())
    rows = []
    for grp in sorted(UMBRELLA):
        d = stats_identity[stats_identity["group"] == grp].sort_values("write", ascending=False)
        uset = set((int(r.layer), int(r.head)) for r in d.head(k).itertuples())
        rows.append({"umbrella": grp, "jaccard_with_union_of_specifics": jaccard(uset, spec_union),
                     "n_specific_union_heads": len(spec_union)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "segmented_umbrella_union_test.csv", index=False)
    return out


def plot_top_heads_per_identity(stats_identity: pd.DataFrame, out_dir: Path, top: int = 8):
    ids = _ordered_present(stats_identity["group"])
    ids = [i for i in ids if i in SPECIFIC_IDENTITIES]
    if not ids:
        return
    _pub_style()
    fig, axes = plt.subplots(1, len(ids), figsize=(max(10, len(ids) * 2.5), 5.2), squeeze=False, sharex=True)
    vmax = 0.0
    for grp in ids:
        d = stats_identity[stats_identity["group"] == grp]
        vmax = max(vmax, float(d["write"].sort_values(ascending=False).head(top).max()))
    for ax, grp in zip(axes[0], ids):
        d = stats_identity[stats_identity["group"] == grp].sort_values("write", ascending=False).head(top).iloc[::-1]
        ax.barh(range(len(d)), d["write"], color=AXIS_COLORS[IDENTITY_AXIS[grp]], alpha=0.9)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"L{int(l)}H{int(h)}" for l, h in zip(d["layer"], d["head"])], fontsize=7, family="monospace")
        ax.set_title(grp, fontsize=10.5, color=AXIS_COLORS[IDENTITY_AXIS[grp]])
        ax.axvline(0, color="#888", lw=0.6); ax.set_xlim(0, vmax * 1.08)
    axes[0][0].set_xlabel("WRITE effect  (normalized Δ logP)")
    fig.suptitle("The heads that most write each identity's stereotype\n"
                 "$\\bf{orientation}$ (blue) · $\\bf{gender}$ (red)", fontsize=12.5, y=1.03)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_top_heads_per_identity.png"); plt.close(fig)


def make_all_plots(stats_identity: pd.DataFrame, seltab: pd.DataFrame, gate: set, top_k: int, out_dir: Path):
    plot_heads_groups_matrix(stats_identity, gate, out_dir)
    plot_selectivity_scatter(seltab, out_dir)
    null = overlap_matrices(stats_identity, top_k, out_dir)
    rw, sel_diff = read_vs_write(stats_identity, seltab, top_k, out_dir)
    umb = umbrella_union_test(stats_identity, top_k, out_dir)
    plot_top_heads_per_identity(stats_identity, out_dir)
    return null, rw, sel_diff, umb


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Identity-segmented WinoQueer head-circuit analysis.")
    ap.add_argument("--patching_raw", type=Path, default=None)
    ap.add_argument("--ablation_raw", type=Path, default=None)
    ap.add_argument("--cohort", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--gate_frac", type=float, default=0.05, help="Magnitude-gate to this top fraction of heads.")
    ap.add_argument("--gate_min_k", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=10, help="Top-K head-set size for overlap/RBO/union tests.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--plot_only", action="store_true",
                    help="Skip the 1 GB raw merge; rebuild the figures from the saved stats CSVs in --out_dir.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        stats_identity = pd.read_csv(args.out_dir / "segmented_head_stats__identity.csv")
        seltab = pd.read_csv(args.out_dir / "segmented_head_selectivity.csv")
        pooled = pd.read_csv(args.out_dir / "segmented_pooled_head_ranking.csv")
        gate = top_heads(pooled, "write", args.gate_frac, args.gate_min_k)
        null, rw, sel_diff, umb = make_all_plots(stats_identity, seltab, gate, args.top_k, args.out_dir)
        print(f"plot_only: regenerated figures in {args.out_dir}")
        if not rw.empty:
            print(rw.to_string(index=False))
        print(f"selectivity(READ) − selectivity(WRITE): {sel_diff[0]:.3f}  CI[{sel_diff[1]:.3f}, {sel_diff[2]:.3f}]")
        return
    if not (args.patching_raw and args.ablation_raw and args.cohort):
        ap.error("--patching_raw, --ablation_raw, --cohort are required unless --plot_only")

    long = load_long(args.patching_raw, args.ablation_raw, args.cohort)
    print(f"Joined long rows: {len(long)} | pairs: {long['row_id'].nunique()} | "
          f"heads: {long.groupby(['layer','head']).ngroups} | identities: {sorted(long['identity'].unique())}")

    pooled = pooled_head_ranking(long)
    pooled.to_csv(args.out_dir / "segmented_pooled_head_ranking.csv", index=False)
    gate = top_heads(pooled, "write", args.gate_frac, args.gate_min_k)
    print(f"Magnitude-gated to {len(gate)} heads (top {args.gate_frac:.0%}).")

    stats = {}
    for level in LEVELS:
        s = head_group_stats(long, level)
        if level in ("axis", "identity"):  # bootstrap CI on WRITE for gated heads only
            s["write_ci_lo"] = np.nan; s["write_ci_hi"] = np.nan
            df_long = long.copy(); df_long["group"] = level_group_col(df_long, level)
            for idx, r in s.iterrows():
                if (int(r["layer"]), int(r["head"])) not in gate:
                    continue
                vals = df_long[(df_long["layer"] == r["layer"]) & (df_long["head"] == r["head"]) & (df_long["group"] == r["group"])]["write"]
                _, lo, hi = bootstrap_ci(vals.to_numpy(), n_boot=args.n_boot)
                s.at[idx, "write_ci_lo"] = lo; s.at[idx, "write_ci_hi"] = hi
        s.to_csv(args.out_dir / f"segmented_head_stats__{level}.csv", index=False)
        stats[level] = s
        if level in ("axispred", "idpred"):
            print(f"  level {level}: {s['group'].nunique()} groups, {len(s)} head-group cells "
                  f"({int(s['qualifies'].sum())} qualifying n>={MIN_CELL})")
        else:
            print(f"  level {level}: {s['group'].nunique()} groups, {len(s)} head-group cells")

    seltab = selectivity_table(stats["identity"], gate)
    seltab.to_csv(args.out_dir / "segmented_head_selectivity.csv", index=False)

    null, rw, sel_diff, umb = make_all_plots(stats["identity"], seltab, gate, args.top_k, args.out_dir)

    print("\n=== Summary ===")
    if not rw.empty:
        print("READ vs WRITE per identity (Spearman | Jaccard top-read/top-write):")
        print(rw.to_string(index=False))
    print(f"\nselectivity(READ) − selectivity(WRITE) bootstrap: {sel_diff[0]:.3f}  CI[{sel_diff[1]:.3f}, {sel_diff[2]:.3f}]")
    if null:
        print(f"Cross-identity overlap computed (random Jaccard null {null['null_mean']:.3f} ± {null['null_sd']:.3f}).")
    print("\nUmbrella vs union-of-specifics:")
    print(umb.to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
