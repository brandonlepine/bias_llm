#!/usr/bin/env python3
"""Identity-segmented analysis of the WinoQueer MLP-neuron attribution (CPU; post-processing).

Consumes the per-group neuron CSVs from run_winoqueer_mlp_neuron_attribution.py
(`winoqueer_mlp_neuron_attribution__<level>__<group>.csv`, each carrying group/level/layer/neuron/
suff_frac/nec_frac/.../n_pairs) and the combined layer profile
(`winoqueer_mlp_layer_profile_by_group.csv`).

Figures (publication-styled, axis-ordered: orientation | gender | umbrella):
  - per-AXIS layer profiles (where in depth each axis's bias mass lives)
  - per-IDENTITY layer profiles (small multiples, colored by axis)
  - neuron specificity vs magnitude (EB-shrunk selectivity over the 7 specific identities)
  - cross-identity neuron Jaccard (do identities share the same neurons? vs a random null)
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
    SPECIFIC_IDENTITIES, IDENTITY_AXIS, IDENTITY_ORDER, AXIS_COLORS, AXIS_SHORT,
    selectivity, jaccard, random_jaccard_null, pub_style, ordered_present, axis_separators, color_ticklabels,
)


def group_value(group: str) -> str | None:
    parts = str(group).split("::")
    return parts[1] if len(parts) >= 2 else None


def load_group_csvs(in_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(in_dir.glob("mlp_neuron_attribution__*.csv")):
        df = pd.read_csv(p)
        if "group" in df.columns:
            frames.append(df)
    if not frames:
        raise SystemExit(f"No per-group MLP CSVs found in {in_dir}")
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- layer profiles
def plot_layer_profiles_by_axis(prof: pd.DataFrame, out_dir: Path):
    ax_prof = prof[prof["level"] == "axis"].copy()
    if ax_prof.empty:
        return
    ax_prof["axis"] = ax_prof["group"].map(group_value)
    pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)
    for ax, col, title in [(axes[0], "suff_pos_mass", "SUFFICIENCY"), (axes[1], "nec_pos_mass", "NECESSITY")]:
        for axis in ["sexual_orientation", "gender_identity", "umbrella"]:
            d = ax_prof[ax_prof["axis"] == axis].sort_values("layer")
            if d.empty:
                continue
            ax.plot(d["layer"], d[col], "-o", ms=3.5, lw=2.2, color=AXIS_COLORS[axis], label=AXIS_SHORT[axis])
        ax.set_title(f"{title} — Σ positive neuron attribution", fontsize=11.5)
        ax.set_xlabel("layer"); ax.set_ylabel("summed positive neuron attribution (frac of bias gap)")
        ax.legend(frameon=False, title="axis")
    fig.suptitle("Where in the network each axis's bias is written / needed\n"
                 "do orientation and gender identities peak at different depths?", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_mlp_layer_profiles_by_axis.png"); plt.close(fig)


def plot_layer_profiles_per_identity(prof: pd.DataFrame, out_dir: Path):
    idp = prof[prof["level"] == "identity"].copy()
    if idp.empty:
        return
    idp["identity"] = idp["group"].map(group_value)
    ids = ordered_present(idp["identity"])
    pub_style()
    ncol = 3
    nrow = (len(ids) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.0, nrow * 2.7), squeeze=False, sharex=True)
    ymax = float(idp["suff_pos_mass"].max()) * 1.05
    for ax, ident in zip(axes.ravel(), ids):
        d = idp[idp["identity"] == ident].sort_values("layer")
        c = AXIS_COLORS[IDENTITY_AXIS[ident]]
        ax.plot(d["layer"], d["suff_pos_mass"], lw=2, color=c, label="suff")
        ax.plot(d["layer"], d["nec_pos_mass"], lw=1.4, color=c, alpha=0.5, ls="--", label="nec")
        ax.set_title(ident, fontsize=10, color=c)
        ax.set_ylim(0, ymax); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(ids):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Per-identity MLP layer profile  (solid = sufficiency, dashed = necessity)\n"
                 "$\\bf{orientation}$ (blue) · $\\bf{gender}$ (red) · $\\bf{umbrella}$ (purple)", fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_mlp_layer_profiles.png"); plt.close(fig)


# ----------------------------------------------------------------------------- selectivity + jaccard
def identity_wide(allg: pd.DataFrame, metric: str):
    idg = allg[allg["level"] == "identity"].copy()
    idg["identity"] = idg["group"].map(group_value)
    idg = idg[idg["identity"].isin(SPECIFIC_IDENTITIES)]
    if idg.empty:
        return None, None
    idg["ln"] = list(zip(idg["layer"].astype(int), idg["neuron"].astype(int)))
    wide = idg.pivot_table(index="ln", columns="identity", values=metric, aggfunc="mean").reindex(columns=SPECIFIC_IDENTITIES)
    n_by_id = idg.groupby("identity")["n_pairs"].max().reindex(SPECIFIC_IDENTITIES).fillna(0).to_numpy()
    return wide, n_by_id


def plot_selectivity_scatter(seltab: pd.DataFrame, metric: str, out_dir: Path, min_present: int, n_ids: int):
    if seltab.empty:
        return
    pub_style()
    fig, ax = plt.subplots(figsize=(9.2, 7))
    palette = {ident: plt.cm.tab10(k % 10) for k, ident in enumerate(IDENTITY_ORDER)}
    for ident in ordered_present(seltab["argmax_identity"]):
        d = seltab[seltab["argmax_identity"] == ident]
        ax.scatter(d["max_val"], d["selectivity"], s=42, alpha=0.8, color=palette[ident],
                   edgecolor="white", linewidth=0.4, label=ident, zorder=3)
    for _, r in seltab.head(12).iterrows():
        ax.annotate(f"L{int(r['layer'])}N{int(r['neuron'])}", (r["max_val"], r["selectivity"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=7, fontweight="bold")
    ax.set_xlabel(f"MAGNITUDE  →  strongest single-identity {metric}")
    ax.set_ylabel("SPECIFICITY  →  neuron selectivity across 7 identities\n(0 = shared by all, 1 = exclusive to one)")
    ax.set_title(f"Are strong bias neurons shared or identity-specific?\n"
                 f"(neurons top-ranked for ≥{min_present}/{n_ids} identities)", fontsize=12)
    ax.legend(title="dominant identity", ncol=2, frameon=False, loc="upper right")
    ax.margins(0.06)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_mlp_selectivity_vs_magnitude.png"); plt.close(fig)


def plot_neuron_jaccard(wide: pd.DataFrame, top_k: int, metric: str, out_dir: Path):
    topsets = {}
    for ident in SPECIFIC_IDENTITIES:
        if ident in wide.columns:
            col = wide[ident].dropna().sort_values(ascending=False).head(top_k)
            topsets[ident] = list(col.index)
    ids = [i for i in ordered_present(topsets) if topsets.get(i)]
    if len(ids) < 2:
        return None
    J = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for a in ids:
        for b in ids:
            J.loc[a, b] = jaccard(topsets[a], topsets[b])
    J.to_csv(out_dir / "segmented_mlp_neuron_jaccard.csv")
    pool = int(pd.Index([n for s in topsets.values() for n in s]).nunique())
    nm, nsd = random_jaccard_null(pool, top_k)

    M = J.to_numpy(dtype=float)
    off = M.copy(); np.fill_diagonal(off, np.nan)
    vmax = max(float(np.nanmax(off)), 0.05)
    seps = axis_separators(ids)
    pub_style()
    fig, ax = plt.subplots(figsize=(1.15 * len(ids) + 3.4, 1.05 * len(ids) + 3.0))
    im = ax.imshow(off, cmap="magma", vmin=0, vmax=vmax)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                ax.text(j, i, "•", ha="center", va="center", color="#bbb", fontsize=12)
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="white" if M[i, j] < 0.62 * vmax else "#111")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=50, ha="right")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids)
    color_ticklabels(ax, ids, "x"); color_ticklabels(ax, ids, "y")
    for s in seps:
        ax.axvline(s - 0.5, color="white", lw=2.5); ax.axhline(s - 0.5, color="white", lw=2.5)
    ax.set_title(f"Do identities share the same bias NEURONS?\n"
                 f"Jaccard overlap of each identity's top-{top_k} {metric} neurons   ·   random baseline ≈ {nm:.3f}", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Jaccard overlap  (fraction of shared top neurons)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "segmented_mlp_neuron_jaccard.png"); plt.close(fig)
    return {"null_mean": nm, "null_sd": nsd, "J": J}


def main() -> None:
    ap = argparse.ArgumentParser(description="Identity-segmented WinoQueer MLP-neuron analysis.")
    ap.add_argument("--in_dir", type=Path, required=True, help="Dir with the per-group MLP CSVs.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=50, help="Top-K neuron-set size for cross-identity Jaccard.")
    ap.add_argument("--metric", choices=["suff_frac", "nec_frac"], default="suff_frac")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allg = load_group_csvs(args.in_dir)

    prof_path = args.in_dir / "mlp_layer_profile_by_group.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        plot_layer_profiles_by_axis(prof, args.out_dir)
        plot_layer_profiles_per_identity(prof, args.out_dir)

    wide, n_by_id = identity_wide(allg, args.metric)
    if wide is None:
        print("No specific-identity MLP groups present; wrote layer profiles only.")
        return
    # The per-group CSVs are top-trimmed, so a neuron absent from an identity's list is ambiguous
    # (genuinely ~0, or just below the trim). Compute selectivity only on neurons that are top-ranked
    # for >=min_present identities, where "absent" really does mean a small value -> honest specificity.
    min_present = max(2, (wide.shape[1] + 1) // 2)
    wsel = wide[wide.notna().sum(axis=1) >= min_present]
    print(f"Selectivity computed on {len(wsel)} neurons present for >={min_present}/{wide.shape[1]} identities "
          f"(of {len(wide)} total top-ranked).")
    vals = wsel.fillna(0.0).to_numpy()
    seltab = pd.DataFrame({
        "layer": [ln[0] for ln in wsel.index], "neuron": [ln[1] for ln in wsel.index],
        "selectivity": [selectivity(vals[i], n_by_id) for i in range(vals.shape[0])],
        "max_val": vals.max(axis=1),
        "argmax_identity": [SPECIFIC_IDENTITIES[int(np.argmax(vals[i]))] for i in range(vals.shape[0])],
    }).sort_values("selectivity", ascending=False)
    seltab.to_csv(args.out_dir / "segmented_mlp_neuron_selectivity.csv", index=False)

    plot_selectivity_scatter(seltab, args.metric, args.out_dir, min_present, wide.shape[1])
    nullj = plot_neuron_jaccard(wide, args.top_k, args.metric, args.out_dir)

    print(f"Wrote segmented MLP analysis to {args.out_dir}")
    print("\nTop specific MLP neurons by selectivity:")
    print(seltab.head(12).to_string(index=False))
    if nullj is not None:
        offdiag = nullj["J"].where(~np.eye(len(nullj["J"]), dtype=bool)).stack()
        print(f"\nMean off-diagonal neuron Jaccard: {offdiag.mean():.3f}  (random null {nullj['null_mean']:.3f} ± {nullj['null_sd']:.3f})")


if __name__ == "__main__":
    main()
