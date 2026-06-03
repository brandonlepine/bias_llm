#!/usr/bin/env python3
"""Dataset-agnostic segmented circuit analysis (WinoQueer / BBQ / CrowS-Pairs).

Generalizes winoqueer_segmented_head_analysis.py: instead of a hard-coded queer taxonomy, it reads
the grouping from COLUMNS in the frozen cohort (joined to the per-pair raws on the stable `row_id`):
  identity  (Group_x / Gender_ID_x)   axis (race/gender/…)   block (region for nationality, else identity)
  source    (bbq / crows-pairs / …)   predicate_label_provisional

Per (layer, head) it computes, from the patching + ablation raws:
  WRITE  = normalized_restoration   (bias_effect / denom; injecting the head creates the bias)
  READ   = attn_readout_to_identity / n_identity_tokens   (head attends to the identity)
  NEC    = frac_bias_removed        (ablating the head removes the bias)

Because identities live in different axes (Black≠Muslim), identity-level selectivity / overlap /
READ-vs-WRITE are computed WITHIN each axis; the cross-AXIS comparison (do axes share heads?) and the
per-SOURCE agreement (does BBQ agree with CrowS?) are the dataset-level outputs.

Auto-detects WinoQueer cohorts (no axis column -> derive from the queer IDENTITY_AXIS map; umbrella =
{LGBTQ, Queer}). Everything is overridable by flag. Designed to reproduce the WinoQueer per-(layer,
head) identity stats exactly (regression-tested).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winoqueer_identity_taxonomy import (  # noqa: E402
    MIN_CELL, IDENTITY_AXIS, selectivity, jaccard, rbo, bootstrap_ci, bootstrap_diff_ci,
    random_jaccard_null, pub_style,
)

CELL = "predicate_label_provisional"
PALETTE = plt.cm.tab20.colors


@dataclass
class Spec:
    identity_col: str
    axis_col: str | None
    block_col: str | None
    source_col: str | None
    umbrella: set = field(default_factory=set)


def detect_spec(cohort: pd.DataFrame, args) -> Spec:
    cols = set(cohort.columns)
    identity_col = args.identity_col or ("identity" if "identity" in cols else
                                         "Group_x" if "Group_x" in cols else "Gender_ID_x")
    axis_col = args.axis_col or ("axis" if "axis" in cols else None)
    block_col = args.block_col or ("block" if "block" in cols else None)
    source_col = args.source_col or ("source" if "source" in cols else None)
    if args.umbrella is not None:
        umb = {u.strip() for u in args.umbrella.split(",") if u.strip()}
    else:
        present = set(cohort[identity_col].astype(str))
        umb = {u for u in ("LGBTQ", "Queer") if u in present}  # winoqueer default
    return Spec(identity_col, axis_col, block_col, source_col, umb)


def attach_groups(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Add canonical `identity`/`axis`/`block`/`source` columns from the cohort's columns."""
    out = df.copy()
    out["identity"] = out[spec.identity_col].astype(str)
    if spec.axis_col and spec.axis_col in out.columns:
        out["axis"] = out[spec.axis_col].astype(str)
    else:  # winoqueer: derive from the queer identity->axis map
        out["axis"] = out["identity"].map(IDENTITY_AXIS).astype(str)
    out["block"] = out[spec.block_col].astype(str) if (spec.block_col and spec.block_col in out.columns) else out["identity"]
    out["source"] = out[spec.source_col].astype(str) if (spec.source_col and spec.source_col in out.columns) else "single"
    out["is_umbrella"] = out["identity"].isin(spec.umbrella)
    # identity_mapped: True when the identity label is a clean canonical group (BBQ always; CrowS flags
    # noisy spans False). Restricts per-IDENTITY analysis to trustworthy labels; axis-level uses all.
    out["identity_mapped"] = out["identity_mapped"].astype(bool) if "identity_mapped" in out.columns else True
    return out


# ----------------------------------------------------------------------------- data
def load_long(patching_raw: Path, ablation_raw: Path | None, cohort: Path, spec: Spec) -> pd.DataFrame:
    pat = pd.read_csv(patching_raw)
    nidt = pat["n_identity_tokens"].clip(lower=1) if "n_identity_tokens" in pat.columns else 1
    pat = pat.assign(write=pat["normalized_restoration"], read=pat["attn_readout_to_identity"] / nidt)[
        ["row_id", "layer", "head", "write", "read"]]
    if ablation_raw is not None and Path(ablation_raw).exists():
        abl = pd.read_csv(ablation_raw)
        abl = abl.assign(nec=abl["frac_bias_removed"])[["row_id", "layer", "head", "nec"]]
        pat = pat.merge(abl, on=["row_id", "layer", "head"], how="inner")
    else:
        pat["nec"] = np.nan
    coh = attach_groups(pd.read_csv(cohort), spec)
    keep = ["row_id", "identity", "axis", "block", "source", "is_umbrella", "identity_mapped", CELL]
    return pat.merge(coh[keep].drop_duplicates("row_id"), on="row_id", how="inner")


def group_stats(long: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per (group, layer, head): mean WRITE/READ/NEC, sign-consistency, n, qualifies(n>=MIN_CELL)."""
    g = long.groupby([group_col, "layer", "head"], sort=False)
    s = g.agg(write=("write", "mean"), read=("read", "mean"), nec=("nec", "mean"),
              write_sign=("write", lambda x: float((x > 0).mean())),
              n=("write", "size")).reset_index().rename(columns={group_col: "group"})
    s["qualifies"] = s["n"] >= MIN_CELL
    return s


def pooled_ranking(long: pd.DataFrame) -> pd.DataFrame:
    return long.groupby(["layer", "head"], sort=False).agg(
        write=("write", "mean"), read=("read", "mean"), nec=("nec", "mean"),
        n=("write", "size")).reset_index().sort_values("write", ascending=False)


def top_set(df: pd.DataFrame, col: str, k: int) -> list[tuple[int, int]]:
    d = df.reindex(df[col].abs().sort_values(ascending=False).index).head(k) if col == "_gate" else \
        df.sort_values(col, ascending=False).head(k)
    return [(int(r.layer), int(r.head)) for r in d.itertuples()]


def gate_heads(pooled: pd.DataFrame, frac: float, min_k: int) -> set:
    k = max(min_k, int(round(len(pooled) * frac)))
    top = pooled.reindex(pooled["write"].abs().sort_values(ascending=False).index).head(k)
    return set((int(r.layer), int(r.head)) for r in top.itertuples())


# ----------------------------------------------------------------------------- within-axis identity analysis
def within_axis_identity(long: pd.DataFrame, spec: Spec, top_k: int, out_dir: Path):
    """For each axis: per-identity head stats, EB-shrunk WRITE/READ selectivity, cross-identity Jaccard
    (write & read) vs a random null, and READ-vs-WRITE (Spearman + selectivity gap)."""
    rows_sel, rows_rw, rows_ov = [], [], []
    for axis, la in long.groupby("axis"):
        ids = sorted(set(la.loc[~la["is_umbrella"] & la["identity_mapped"], "identity"]))
        if len(ids) < 2:
            continue
        la = la[la["identity"].isin(ids)]  # restrict to the clean, specific identities of this axis
        st = group_stats(la, "identity")
        piv_w = st.pivot_table(index=["layer", "head"], columns="group", values="write").reindex(columns=ids)
        piv_r = st.pivot_table(index=["layer", "head"], columns="group", values="read").reindex(columns=ids)
        n_by = st.groupby("group")["n"].max().reindex(ids).fillna(0).to_numpy()
        # selectivity per head over this axis's identities
        for (L, H), wrow in piv_w.iterrows():
            w = wrow.fillna(0).to_numpy()
            r = piv_r.loc[(L, H)].fillna(0).to_numpy()
            rows_sel.append({"axis": axis, "layer": int(L), "head": int(H),
                             "selectivity_write": selectivity(w, n_by), "selectivity_read": selectivity(r, n_by),
                             "argmax_write": ids[int(np.argmax(w))], "max_write": float(np.max(w))})
        # cross-identity overlap (write & read), with random null
        pool = st.groupby(["layer", "head"]).ngroups
        nm, nsd = random_jaccard_null(pool, top_k)
        for metric in ("write", "read"):
            sets = {i: set((int(r.layer), int(r.head)) for r in
                           st[st.group == i].sort_values(metric, ascending=False).head(top_k).itertuples()) for i in ids}
            for a in ids:
                for b in ids:
                    if a < b:
                        rows_ov.append({"axis": axis, "metric": metric, "id_a": a, "id_b": b,
                                        "jaccard": jaccard(sets[a], sets[b]), "rbo": rbo(list(sets[a]), list(sets[b])),
                                        "null_mean": nm, "null_sd": nsd})
        # READ vs WRITE per identity
        for ident in ids:
            d = st[st.group == ident]
            tr = set((int(r.layer), int(r.head)) for r in d.sort_values("read", ascending=False).head(top_k).itertuples())
            tw = set((int(r.layer), int(r.head)) for r in d.sort_values("write", ascending=False).head(top_k).itertuples())
            rows_rw.append({"axis": axis, "identity": ident,
                            "spearman_read_write": float(d["read"].corr(d["write"], method="spearman")),
                            "jaccard_read_write": jaccard(tr, tw)})
    sel = pd.DataFrame(rows_sel); rw = pd.DataFrame(rows_rw); ov = pd.DataFrame(rows_ov)
    sel.to_csv(out_dir / "identity_selectivity_within_axis.csv", index=False)
    rw.to_csv(out_dir / "read_vs_write_within_axis.csv", index=False)
    ov.to_csv(out_dir / "identity_overlap_within_axis.csv", index=False)
    return sel, rw, ov


# ----------------------------------------------------------------------------- cross-axis + source
def cross_axis_overlap(long: pd.DataFrame, top_k: int, out_dir: Path):
    """Do different AXES share their top WRITE heads? (axis × axis Jaccard, + random null)."""
    st = group_stats(long, "axis")
    axes = sorted(st["group"].unique())
    sets = {a: set((int(r.layer), int(r.head)) for r in
                   st[st.group == a].sort_values("write", ascending=False).head(top_k).itertuples()) for a in axes}
    J = pd.DataFrame(index=axes, columns=axes, dtype=float)
    for a in axes:
        for b in axes:
            J.loc[a, b] = jaccard(sets[a], sets[b])
    J.to_csv(out_dir / "cross_axis_head_jaccard.csv")
    nm, nsd = random_jaccard_null(st.groupby(["layer", "head"]).ngroups, top_k)
    if len(axes) < 2:  # nothing to compare (single-axis run)
        return J, nm
    pub_style()
    fig, ax = plt.subplots(figsize=(1.1 * len(axes) + 3, 1.0 * len(axes) + 2.5))
    M = J.to_numpy(float); off = M.copy(); np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap="magma", vmin=0, vmax=max(float(np.nanmax(off)), 0.05))
    for i in range(len(axes)):
        for j in range(len(axes)):
            ax.text(j, i, "•" if i == j else f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#bbb" if i == j else ("white" if M[i, j] < 0.6 * max(np.nanmax(off), .05) else "#111"))
    ax.set_xticks(range(len(axes))); ax.set_xticklabels(axes, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(axes))); ax.set_yticklabels(axes, fontsize=8)
    ax.set_title(f"Do social AXES share bias-writing heads?\nJaccard of top-{top_k} WRITE heads  ·  null ≈ {nm:.3f}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("Jaccard")
    fig.tight_layout(); fig.savefig(out_dir / "cross_axis_head_jaccard.png"); plt.close(fig)
    return J, nm


def source_agreement(long: pd.DataFrame, top_k: int, out_dir: Path):
    """Per axis: BBQ vs CrowS top-WRITE-head overlap + mean WRITE (cross-dataset corroboration)."""
    if long["source"].nunique() < 2:
        return None
    rows = []
    for axis, la in long.groupby("axis"):
        srcs = sorted(la["source"].unique())
        if len(srcs) < 2:
            continue
        st = {s: group_stats(la[la.source == s], "axis") for s in srcs}
        sets = {s: set((int(r.layer), int(r.head)) for r in
                       st[s].sort_values("write", ascending=False).head(top_k).itertuples()) for s in srcs}
        rec = {"axis": axis}
        for s in srcs:
            d = la[la.source == s]
            rec[f"mean_write_{s}"] = float(d["write"].mean()); rec[f"n_{s}"] = int(d["row_id"].nunique())
        # pairwise jaccard between the first two sources
        rec["source_top_head_jaccard"] = jaccard(sets[srcs[0]], sets[srcs[1]])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "source_agreement_by_axis.csv", index=False)
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-agnostic segmented circuit analysis.")
    ap.add_argument("--patching_raw", type=Path, required=True)
    ap.add_argument("--ablation_raw", type=Path, default=None)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--identity_col", type=str, default=None)
    ap.add_argument("--axis_col", type=str, default=None)
    ap.add_argument("--block_col", type=str, default=None)
    ap.add_argument("--source_col", type=str, default=None)
    ap.add_argument("--umbrella", type=str, default=None, help="comma list of umbrella identities (auto for WinoQueer)")
    ap.add_argument("--gate_frac", type=float, default=0.05)
    ap.add_argument("--gate_min_k", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = detect_spec(pd.read_csv(args.cohort, nrows=200), args)
    long = load_long(args.patching_raw, args.ablation_raw, args.cohort, spec)
    print(f"long rows: {len(long)} | pairs: {long['row_id'].nunique()} | "
          f"heads: {long.groupby(['layer','head']).ngroups}")
    print(f"identity_col={spec.identity_col} | axes: {sorted(long['axis'].unique())} | "
          f"sources: {sorted(long['source'].unique())} | umbrella: {sorted(spec.umbrella)}")

    # per-level stats (the durable tables)
    for level, col in [("axis", "axis"), ("block", "block"), ("identity", "identity")]:
        group_stats(long, col).to_csv(args.out_dir / f"head_stats__{level}.csv", index=False)
    pooled = pooled_ranking(long); pooled.to_csv(args.out_dir / "pooled_head_ranking.csv", index=False)

    sel, rw, ov = within_axis_identity(long, spec, args.top_k, args.out_dir)
    Jax, nax = cross_axis_overlap(long, args.top_k, args.out_dir)
    src = source_agreement(long, args.top_k, args.out_dir)

    if len(Jax) >= 2:
        print("\n=== cross-axis WRITE-head overlap (mean off-diagonal Jaccard vs null) ===")
        off = Jax.where(~np.eye(len(Jax), dtype=bool)).stack()
        print(f"  mean {off.mean():.3f}  (random null {nax:.3f}) — high => axes share a circuit; low => axis-specific")
    if not rw.empty:
        print("\n=== READ vs WRITE per identity (low Spearman/Jaccard => different heads read vs write) ===")
        print(rw.groupby("axis")[["spearman_read_write", "jaccard_read_write"]].mean().round(3).to_string())
    if src is not None and not src.empty:
        print("\n=== source agreement per axis (BBQ vs CrowS) ===")
        print(src.round(3).to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
