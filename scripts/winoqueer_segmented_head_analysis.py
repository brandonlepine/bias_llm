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
    MIN_CELL, LEVELS, SPECIFIC_IDENTITIES, UMBRELLA, annotate,
    selectivity, jaccard, rbo, bootstrap_ci, bootstrap_diff_ci, random_jaccard_null,
)

CELL_COL = "predicate_label_provisional"


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
    cols = [i for i in SPECIFIC_IDENTITIES if i in set(stats_identity["group"])]
    cols += [u for u in ("Queer", "LGBTQ") if u in set(stats_identity["group"])]
    df = stats_identity[stats_identity["group"].isin(cols)]
    if df.empty:
        return
    piv = df.pivot_table(index=["layer", "head"], columns="group", values="write").reindex(columns=cols)
    if gate:
        piv = piv[[idx in gate for idx in piv.index]]
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index).head(top_n)
    if piv.empty:
        return
    labels = [f"L{l}H{h}" for l, h in piv.index]
    raw = piv.to_numpy(dtype=float)
    pos = np.clip(raw, 0, None)
    rownorm = pos / np.where(pos.sum(1, keepdims=True) > 0, pos.sum(1, keepdims=True), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(2.0 + len(cols) * 1.1, max(6, len(labels) * 0.28)))
    v = max(float(np.nanmax(np.abs(raw))), 1e-9)
    for ax, mat, title, kw in [
        (axes[0], raw, "WRITE (normalized bias_effect)", dict(cmap="RdBu_r", vmin=-v, vmax=v)),
        (axes[1], rownorm, "within-head share (positive)", dict(cmap="magma", vmin=0, vmax=1)),
    ]:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", **kw)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Heads × identities — shared vs identity-specific", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_heads_groups_matrix.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_selectivity_scatter(seltab: pd.DataFrame, out_dir: Path):
    if seltab.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(seltab["max_write"], seltab["selectivity_write"], s=26, alpha=0.8, c="#c0392b", edgecolor="white", linewidth=0.3)
    for _, r in seltab.head(14).iterrows():
        ax.annotate(f"L{int(r['layer'])}H{int(r['head'])}\n{r['argmax_write']}", (r["max_write"], r["selectivity_write"]),
                    xytext=(3, 2), textcoords="offset points", fontsize=6.5, fontweight="bold")
    ax.set_xlabel("max-identity WRITE (normalized bias_effect)")
    ax.set_ylabel("WRITE selectivity (EB-shrunk, 7 identities)")
    ax.set_title("Head specificity vs magnitude — top-right = strong & identity-specific")
    fig.tight_layout(); fig.savefig(out_dir / "segmented_head_selectivity_scatter.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def overlap_matrices(stats_identity: pd.DataFrame, top_k: int, out_dir: Path):
    ids = [i for i in SPECIFIC_IDENTITIES if i in set(stats_identity["group"])]
    if len(ids) < 2:
        return None

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
        fig, ax = plt.subplots(figsize=(1.1 * len(ids) + 3, 1.0 * len(ids) + 2.5))
        im = ax.imshow(J.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids, fontsize=8)
        ax.set_title(f"Cross-identity head Jaccard — {metric.upper()} top-{top_k}")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        fig.tight_layout(); fig.savefig(out_dir / f"segmented_head_jaccard_{metric}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    pool = stats_identity.groupby(["layer", "head"]).ngroups
    nm, nsd = random_jaccard_null(pool, top_k)
    return {"null_mean": nm, "null_sd": nsd}


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
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.bar(rw["identity"], rw["spearman_read_write"], color="#2c7fb8", alpha=0.85)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_ylabel("Spearman(READ, WRITE) over heads")
        ax.set_title("READ vs WRITE per identity (low = different circuits)")
        ax.set_xticks(range(len(rw))); ax.set_xticklabels(rw["identity"], rotation=45, ha="right", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "segmented_read_vs_write.png", dpi=160, bbox_inches="tight"); plt.close(fig)
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


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Identity-segmented WinoQueer head-circuit analysis.")
    ap.add_argument("--patching_raw", type=Path, required=True)
    ap.add_argument("--ablation_raw", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--gate_frac", type=float, default=0.05, help="Magnitude-gate to this top fraction of heads.")
    ap.add_argument("--gate_min_k", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=10, help="Top-K head-set size for overlap/RBO/union tests.")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
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

    plot_heads_groups_matrix(stats["identity"], gate, args.out_dir)
    plot_selectivity_scatter(seltab, args.out_dir)
    null = overlap_matrices(stats["identity"], args.top_k, args.out_dir)
    rw, sel_diff = read_vs_write(stats["identity"], seltab, args.top_k, args.out_dir)
    umb = umbrella_union_test(stats["identity"], args.top_k, args.out_dir)

    # per-identity top-head bars (annotated implicitly by order)
    ids = [g for g in SPECIFIC_IDENTITIES if g in set(stats["identity"]["group"])]
    if ids:
        fig, axes = plt.subplots(1, len(ids), figsize=(max(10, len(ids) * 2.6), 5), squeeze=False)
        for ax, grp in zip(axes[0], ids):
            d = stats["identity"][stats["identity"]["group"] == grp].sort_values("write", ascending=False).head(8).iloc[::-1]
            ax.barh(range(len(d)), d["write"], color="#c0392b", alpha=0.85)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels([f"L{int(l)}H{int(h)}" for l, h in zip(d["layer"], d["head"])], fontsize=7)
            ax.set_title(grp, fontsize=10); ax.axvline(0, color="#888", lw=0.6)
        fig.suptitle("Top WRITE heads per identity", fontsize=12)
        fig.tight_layout(); fig.savefig(args.out_dir / "segmented_top_heads_per_identity.png", dpi=160, bbox_inches="tight"); plt.close(fig)

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
