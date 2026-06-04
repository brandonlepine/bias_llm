#!/usr/bin/env python3
"""Dataset-agnostic cross-axis MLP-neuron analysis (WinoQueer / BBQ / CrowS).

Companion to segmented_circuit_analysis.py (heads), for the neuron level. Consumes the per-group
neuron CSVs written by run_winoqueer_mlp_neuron_attribution.py (each row: group='<level>::<value>',
level, layer, neuron, suff_frac, nec_frac, ..., n_pairs) plus the combined layer-profile-by-group.

Outputs:
  - per-AXIS neuron layer profiles (where each axis's neuron bias mass sits in depth)
  - cross-AXIS top-neuron Jaccard (do axes share bias NEURONS? vs random null)
  - within-AXIS identity neuron selectivity (how identity-specific, per axis) — needs the cohort to
    map identity -> axis. Restricted to neurons present for >= ceil(G/2) of an axis's identities
    (the per-group CSVs are top-trimmed, so 'absent' is ambiguous).

Globs both 'winoqueer_mlp_neuron_attribution__*.csv' and the neutral 'mlp_neuron_attribution__*.csv'.
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
from winoqueer_identity_taxonomy import IDENTITY_AXIS, selectivity, jaccard, random_jaccard_null, pub_style  # noqa: E402


def parse_group(g: str) -> tuple[str, str]:
    level, _, val = str(g).partition("::")
    return level, val


def load_groups(in_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(in_dir.glob("*mlp_neuron_attribution__*.csv")) if "group" in pd.read_csv(p, nrows=0).columns]
    if not frames:
        raise SystemExit(f"No per-group MLP CSVs in {in_dir} (need the per-group run output)")
    allg = pd.concat(frames, ignore_index=True)
    lv = allg["group"].map(parse_group)
    allg["level"] = [x[0] for x in lv]; allg["value"] = [x[1] for x in lv]
    return allg


def id_to_axis(cohort_csv: Path) -> dict:
    c = pd.read_csv(cohort_csv)
    idc = "Group_x" if "Group_x" in c.columns else ("Gender_ID_x" if "Gender_ID_x" in c.columns else "identity")
    if "axis" in c.columns:
        return dict(zip(c[idc].astype(str), c["axis"].astype(str)))
    return {i: IDENTITY_AXIS.get(i, "?") for i in c[idc].astype(str).unique()}


def _pal(keys):
    return {k: plt.cm.tab10(i % 10) for i, k in enumerate(sorted(keys))}


def fig_axis_layer_profile(in_dir: Path, out_dir: Path):
    prof_files = list(in_dir.glob("*layer_profile_by_group.csv"))
    if not prof_files:
        return
    prof = pd.read_csv(prof_files[0])
    ax_prof = prof[prof["level"] == "axis"].copy() if "level" in prof.columns else \
        prof.assign(level=prof["group"].map(lambda g: parse_group(g)[0])).query("level=='axis'")
    if ax_prof.empty:
        return
    ax_prof["axis"] = ax_prof["group"].map(lambda g: parse_group(g)[1])
    pal = _pal(ax_prof["axis"].unique())
    pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for ax, col, ttl in [(axes[0], "suff_pos_mass", "SUFFICIENCY"), (axes[1], "nec_pos_mass", "NECESSITY")]:
        for axis, g in ax_prof.groupby("axis"):
            g = g.sort_values("layer")
            ax.plot(g["layer"], g[col], "-o", ms=3, lw=2, color=pal[axis], label=axis)
        ax.set_title(f"{ttl} — Σ positive neuron attribution"); ax.set_xlabel("layer")
        ax.set_ylabel("summed positive neuron attribution (frac of bias gap)"); ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Where each axis's bias is written / needed at the NEURON level", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "mlp_axis_layer_profile.png"); plt.close(fig)


def fig_cross_axis_neuron_jaccard(allg: pd.DataFrame, metric: str, top_k: int, out_dir: Path):
    ax = allg[allg["level"] == "axis"].copy()
    axes = sorted(ax["value"].unique())
    if len(axes) < 2:
        return None
    ax["ln"] = list(zip(ax["layer"].astype(int), ax["neuron"].astype(int)))
    sets = {a: list(ax[ax.value == a].sort_values(metric, ascending=False).head(top_k)["ln"]) for a in axes}
    J = pd.DataFrame(index=axes, columns=axes, dtype=float)
    for a in axes:
        for b in axes:
            J.loc[a, b] = jaccard(sets[a], sets[b])
    J.to_csv(out_dir / "mlp_cross_axis_neuron_jaccard.csv")
    pool = int(ax["ln"].nunique())
    nm, _ = random_jaccard_null(pool, top_k)
    M = J.to_numpy(float); off = M.copy(); np.fill_diagonal(off, np.nan)
    pub_style()
    fig, a = plt.subplots(figsize=(1.1 * len(axes) + 3, 1.0 * len(axes) + 2.5))
    im = a.imshow(off, cmap="magma", vmin=0, vmax=max(float(np.nanmax(off)), 0.05))
    for i in range(len(axes)):
        for j in range(len(axes)):
            a.text(j, i, "•" if i == j else f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                   color="#bbb" if i == j else ("white" if M[i, j] < 0.6 * max(np.nanmax(off), .05) else "#111"))
    a.set_xticks(range(len(axes))); a.set_xticklabels(axes, rotation=50, ha="right", fontsize=8)
    a.set_yticks(range(len(axes))); a.set_yticklabels(axes, fontsize=8)
    a.set_title(f"Do social axes share bias NEURONS?\nJaccard of top-{top_k} {metric} neurons  ·  null ≈ {nm:.3f}", fontsize=11)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02).set_label("Jaccard")
    fig.tight_layout(); fig.savefig(out_dir / "mlp_cross_axis_neuron_jaccard.png"); plt.close(fig)
    return J, nm


def within_axis_identity_selectivity(allg: pd.DataFrame, id2ax: dict, metric: str, out_dir: Path):
    idg = allg[allg["level"] == "identity"].copy()
    if idg.empty:
        return pd.DataFrame()
    idg["axis"] = idg["value"].map(id2ax)
    rows = []
    for axis, ga in idg.groupby("axis"):
        ids = sorted(ga["value"].unique())
        if len(ids) < 2:
            continue
        ga = ga.assign(ln=list(zip(ga["layer"].astype(int), ga["neuron"].astype(int))))
        wide = ga.pivot_table(index="ln", columns="value", values=metric).reindex(columns=ids)
        n_by = ga.groupby("value")["n_pairs"].max().reindex(ids).fillna(0).to_numpy()
        min_present = max(2, (len(ids) + 1) // 2)
        wsel = wide[wide.notna().sum(axis=1) >= min_present]
        vals = wsel.fillna(0).to_numpy()
        for i in range(len(wsel)):
            rows.append({"axis": axis, "layer": wsel.index[i][0], "neuron": wsel.index[i][1],
                         "selectivity": selectivity(vals[i], n_by), "max_val": float(vals[i].max()),
                         "argmax_identity": ids[int(np.argmax(vals[i]))]})
    sel = pd.DataFrame(rows)
    sel.to_csv(out_dir / "mlp_identity_selectivity_within_axis.csv", index=False)
    if not sel.empty:
        pub_style()
        order = sel.groupby("axis")["selectivity"].median().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(10, 5.4))
        bp = ax.boxplot([sel[sel.axis == a]["selectivity"].dropna().values for a in order],
                        labels=list(order), patch_artist=True, showfliers=False)
        for patch, a in zip(bp["boxes"], order):
            patch.set_facecolor(_pal(order)[a]); patch.set_alpha(0.7)
        ax.set_ylabel("neuron selectivity (0=shared, 1=one-identity)")
        ax.set_xticklabels(list(order), rotation=35, ha="right")
        ax.set_title("Within each axis, how identity-specific are the bias NEURONS?")
        fig.tight_layout(); fig.savefig(out_dir / "mlp_selectivity_by_axis.png"); plt.close(fig)
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-agnostic cross-axis MLP-neuron analysis.")
    ap.add_argument("--in_dir", type=Path, required=True, help="dir with the per-group MLP CSVs")
    ap.add_argument("--cohort", type=Path, required=True, help="cohort (for identity->axis mapping)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["suff_frac", "nec_frac"], default="suff_frac")
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    allg = load_groups(args.in_dir)
    id2ax = id_to_axis(args.cohort)
    print(f"per-group rows: {len(allg)} | levels: {sorted(allg['level'].unique())} | "
          f"axes: {sorted(allg.loc[allg.level=='axis','value'].unique())}")

    fig_axis_layer_profile(args.in_dir, args.out_dir)
    res = fig_cross_axis_neuron_jaccard(allg, args.metric, args.top_k, args.out_dir)
    sel = within_axis_identity_selectivity(allg, id2ax, args.metric, args.out_dir)

    if res is not None:
        J, nm = res
        off = J.where(~np.eye(len(J), dtype=bool)).stack()
        print(f"\ncross-axis neuron Jaccard: mean {off.mean():.3f}  (null {nm:.3f})")
    if not sel.empty:
        print("\nmedian neuron selectivity per axis:")
        print(sel.groupby("axis")["selectivity"].median().round(3).sort_values(ascending=False).to_string())
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
