#!/usr/bin/env python3
"""Identity-segmented analysis of the WinoQueer MLP-neuron attribution (CPU; post-processing).

Consumes the per-group neuron CSVs written by run_winoqueer_mlp_neuron_attribution.py
(`winoqueer_mlp_neuron_attribution__<level>__<group>.csv`, each carrying `group`, `level`,
`layer`, `neuron`, `suff_frac`, `nec_frac`, ... `n_pairs`) and the combined layer profile
(`winoqueer_mlp_layer_profile_by_group.csv`).

Produces: per-group layer profiles (small multiples), per-neuron EB-shrunk selectivity over the 7
specific identities + a specificity-vs-magnitude scatter, and a cross-identity neuron-Jaccard matrix
on the union of top neurons. Neurons absent from a group's (top-trimmed) CSV are treated as ~0.
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
from winoqueer_identity_taxonomy import SPECIFIC_IDENTITIES, selectivity, jaccard  # noqa: E402


def load_group_csvs(in_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(in_dir.glob("winoqueer_mlp_neuron_attribution__*.csv")):
        df = pd.read_csv(p)
        if "group" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise SystemExit(f"No per-group MLP CSVs found in {in_dir}")
    return pd.concat(frames, ignore_index=True)


def identity_value(group: str) -> str | None:
    parts = str(group).split("::")
    return parts[1] if len(parts) >= 2 else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Identity-segmented WinoQueer MLP-neuron analysis.")
    ap.add_argument("--in_dir", type=Path, required=True, help="Dir with the per-group MLP CSVs.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=50, help="Top-K neuron-set size for cross-identity Jaccard.")
    ap.add_argument("--metric", choices=["suff_frac", "nec_frac"], default="suff_frac")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allg = load_group_csvs(args.in_dir)

    # ---- per-group layer profiles (small multiples) ----
    prof_path = args.in_dir / "winoqueer_mlp_layer_profile_by_group.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        groups = sorted(prof["group"].unique())
        ncol = 3
        nrow = (len(groups) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 2.8), squeeze=False)
        for ax, grp in zip(axes.ravel(), groups):
            g = prof[prof["group"] == grp].sort_values("layer")
            ax.plot(g["layer"], g["suff_pos_mass"], color="#2c7fb8", lw=1.5, label="suff")
            ax.plot(g["layer"], g["nec_pos_mass"], color="#c0392b", lw=1.5, label="nec")
            ax.set_title(grp, fontsize=8)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Per-group MLP layer profiles (positive neuron attribution mass)", fontsize=12)
        fig.tight_layout(); fig.savefig(args.out_dir / "segmented_mlp_layer_profiles.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- identity-level wide table for selectivity ----
    idg = allg[allg["level"] == "identity"].copy()
    idg["identity"] = idg["group"].map(identity_value)
    idg = idg[idg["identity"].isin(SPECIFIC_IDENTITIES)]
    if idg.empty:
        print("No specific-identity MLP groups present; wrote layer profiles only.")
        return

    idg["ln"] = list(zip(idg["layer"].astype(int), idg["neuron"].astype(int)))
    wide = idg.pivot_table(index="ln", columns="identity", values=args.metric, aggfunc="mean")
    wide = wide.reindex(columns=SPECIFIC_IDENTITIES)
    n_by_id = idg.groupby("identity")["n_pairs"].max().reindex(SPECIFIC_IDENTITIES).fillna(0).to_numpy()
    vals = wide.fillna(0.0).to_numpy()
    sel = [selectivity(vals[i], n_by_id) for i in range(vals.shape[0])]
    argmax_id = [SPECIFIC_IDENTITIES[int(np.argmax(vals[i]))] for i in range(vals.shape[0])]
    seltab = pd.DataFrame({
        "layer": [ln[0] for ln in wide.index], "neuron": [ln[1] for ln in wide.index],
        "selectivity": sel, "max_val": vals.max(axis=1), "argmax_identity": argmax_id,
    }).sort_values("selectivity", ascending=False)
    seltab.to_csv(args.out_dir / "segmented_mlp_neuron_selectivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(seltab["max_val"], seltab["selectivity"], s=20, alpha=0.7, c="#2c7fb8", edgecolor="white", linewidth=0.2)
    for _, r in seltab.head(12).iterrows():
        ax.annotate(f"L{int(r.layer)}N{int(r.neuron)}\n{r.argmax_identity}", (r["max_val"], r["selectivity"]),
                    xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel(f"max-identity {args.metric}"); ax.set_ylabel("neuron selectivity (EB-shrunk, 7 identities)")
    ax.set_title("MLP neuron specificity vs magnitude")
    fig.tight_layout(); fig.savefig(args.out_dir / "segmented_mlp_selectivity_vs_magnitude.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- cross-identity neuron Jaccard on top-K ----
    topsets = {}
    for ident in SPECIFIC_IDENTITIES:
        if ident in wide.columns:
            col = wide[ident].dropna().sort_values(ascending=False).head(args.top_k)
            topsets[ident] = set(col.index)
    present = [i for i in SPECIFIC_IDENTITIES if topsets.get(i)]
    if len(present) >= 2:
        J = pd.DataFrame(index=present, columns=present, dtype=float)
        for a in present:
            for b in present:
                J.loc[a, b] = jaccard(topsets[a], topsets[b])
        J.to_csv(args.out_dir / "segmented_mlp_neuron_jaccard.csv")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(J.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=8)
        ax.set_title(f"Cross-identity MLP-neuron Jaccard (top-{args.top_k}, {args.metric})")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        fig.tight_layout(); fig.savefig(args.out_dir / "segmented_mlp_neuron_jaccard.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    print(f"Wrote segmented MLP analysis to {args.out_dir}")
    print("\nTop specific MLP neurons by selectivity:")
    print(seltab.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
