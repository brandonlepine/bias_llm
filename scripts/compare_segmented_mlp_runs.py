#!/usr/bin/env python3
"""Cross-dataset / cross-run comparison of MLP-NEURON attribution (the neuron analog of
compare_segmented_runs.py, which does the same for heads).

Takes >=2 per-group MLP dirs (e.g. WinoQueer's segmented/mlp and the combined BBQ+CrowS
combined_mlp) and asks whether the datasets' bias circuits overlap AT THE NEURON LEVEL:

  - cross-DATASET pooled top-neuron Jaccard: do the datasets' OVERALL bias circuits share
    neurons? (the headline OOD-validation figure — the neuron mirror of
    compare_cross_dataset_head_jaccard.png)
  - per-(identity AXIS) cross-dataset neuron Jaccard: for axes present in >=2 datasets
    (e.g. gender, sexual_orientation), do the datasets agree on that axis's bias neurons?
  - per-axis neuron layer profiles overlaid (where each axis writes bias in depth, dataset
    by dataset; colour = axis, line style = dataset)

Each --run DIR is the per-group MLP dir (same input as segmented_mlp_circuit_analysis.py).
The pooled (all-pairs) ranking is auto-located near DIR; override with --pooled LABEL FILE.

Usage:
  python scripts/compare_segmented_mlp_runs.py \
      --run winoqueer pod_results/winoqueer/segmented/mlp \
      --run bbq_crows pod_results/bbq_crows_combined/combined_mlp \
      --out_dir pod_results/cross_dataset/mlp  [--top_k 50] [--metric suff_frac]
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
from winoqueer_identity_taxonomy import pub_style, jaccard, random_jaccard_null  # noqa: E402


# Canonicalise dataset-specific axis labels onto a shared name so the SAME social axis aligns
# across datasets (WinoQueer calls it gender_identity; BBQ/CrowS call it gender). Unmapped axes
# pass through unchanged (e.g. WinoQueer 'umbrella' has no BBQ analog and stays its own axis).
CANON = {
    "gender_identity": "gender", "gender": "gender",
    "sexual_orientation": "sexual_orientation", "orientation": "sexual_orientation",
    "race_ethnicity": "race", "race": "race",
    "ses": "socioeconomic", "socioeconomic": "socioeconomic",
    "disability_status": "disability", "disability": "disability",
}
# Shared categorical order so a given axis keeps the same colour/slot in every dataset.
AXIS_ORDER = ["sexual_orientation", "gender", "race", "religion", "age", "nationality",
              "socioeconomic", "physical_appearance", "disability", "umbrella"]


def canon(a) -> str:
    return CANON.get(str(a), str(a))


def axis_rank(a):
    a = str(a)
    return (AXIS_ORDER.index(a) if a in AXIS_ORDER else len(AXIS_ORDER), a)


def order_axes(values):
    return sorted(set(map(str, values)), key=axis_rank)


def parse_group(g: str) -> tuple[str, str]:
    level, _, val = str(g).partition("::")
    return level, val


def find_pooled(in_dir: Path, override: Path | None) -> Path | None:
    """Locate the all-pairs pooled neuron ranking (mlp_neuron_attribution.csv, NO group suffix)
    for a run. combined_mlp keeps it in-dir; WinoQueer keeps it under pooled/mlp/."""
    if override is not None:
        return override
    cands: list[Path] = []
    for d in (in_dir, in_dir.parent / "mlp", in_dir.parent / "pooled" / "mlp",
              in_dir.parent.parent / "pooled" / "mlp"):
        if d.is_dir():
            cands += [p for p in sorted(d.glob("*mlp_neuron_attribution.csv")) if "__" not in p.name]
    return cands[0] if cands else None


def load_run(label: str, in_dir: Path, pooled_override: Path | None) -> dict:
    frames = [pd.read_csv(p) for p in sorted(in_dir.glob("*mlp_neuron_attribution__*.csv"))
              if "group" in pd.read_csv(p, nrows=0).columns]
    allg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not allg.empty:
        lv = allg["group"].map(parse_group)
        allg["level"] = [x[0] for x in lv]
        allg["value"] = [x[1] for x in lv]

    pooled_path = find_pooled(in_dir, pooled_override)
    pooled = pd.read_csv(pooled_path) if pooled_path is not None else pd.DataFrame()
    if pooled_path is None:
        print(f"[{label}] WARNING: no pooled mlp_neuron_attribution.csv found near {in_dir} "
              f"— skipping this run in the pooled-overlap figure.")
    else:
        print(f"[{label}] pooled ranking: {pooled_path}")

    prof_files = list(in_dir.glob("*layer_profile_by_group.csv"))
    prof = pd.read_csv(prof_files[0]) if prof_files else pd.DataFrame()
    return {"label": label, "allg": allg, "pooled": pooled, "prof": prof}


def fig_cross_dataset_neuron_overlap(runs: list[dict], metric: str, top_k: int, out_dir: Path):
    """Top-neuron Jaccard BETWEEN datasets' overall (pooled) bias circuits."""
    labels = [r["label"] for r in runs if not r["pooled"].empty]
    if len(labels) < 2:
        print("pooled-overlap figure skipped (need >=2 runs with a pooled ranking).")
        return None
    by = {r["label"]: r for r in runs}
    sets, pool = {}, 0
    for lab in labels:
        p = by[lab]["pooled"].sort_values(metric, ascending=False).head(top_k)
        sets[lab] = set((int(r.layer), int(r.neuron)) for r in p.itertuples())
        pool = max(pool, by[lab]["pooled"].groupby(["layer", "neuron"]).ngroups)
    J = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            J.loc[a, b] = jaccard(sets[a], sets[b])
    J.to_csv(out_dir / "compare_cross_dataset_neuron_jaccard.csv")
    nm, _ = random_jaccard_null(pool, top_k)
    pub_style()
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 1.4 * len(labels) + 2.5))
    M = J.to_numpy(float)
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontweight="bold",
                    color="white" if M[i, j] < 0.6 else "#111")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(f"Do datasets share bias NEURONS?\nJaccard of top-{top_k} {metric} neurons "
                 f"·  random null ≈ {nm:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Jaccard")
    fig.tight_layout(); fig.savefig(out_dir / "compare_cross_dataset_neuron_jaccard.png"); plt.close(fig)
    return J, nm


def fig_axis_cross_dataset_jaccard(runs: list[dict], metric: str, top_k: int, out_dir: Path):
    """For each (canonical) axis present in >=2 datasets, cross-dataset Jaccard of that axis's
    top-k neurons. With exactly 2 datasets this is one Jaccard per axis; with more, the mean of
    all cross-dataset pairs."""
    # axis-level neuron sets per run, keyed by canonical axis
    per_run_axis: dict[str, dict[str, set]] = {}
    for r in runs:
        a = r["allg"]
        if a.empty:
            continue
        ax = a[a["level"] == "axis"].copy()
        ax["caxis"] = ax["value"].map(canon)
        d = {}
        for cax, g in ax.groupby("caxis"):
            top = g.sort_values(metric, ascending=False).head(top_k)
            d[cax] = set((int(t.layer), int(t.neuron)) for t in top.itertuples())
        per_run_axis[r["label"]] = d
    labels = list(per_run_axis)
    if len(labels) < 2:
        return None
    shared = order_axes({ax for ax in set().union(*[set(d) for d in per_run_axis.values()])
                         if sum(ax in per_run_axis[l] for l in labels) >= 2})
    if not shared:
        print("per-axis cross-dataset figure skipped (no axis shared by >=2 datasets).")
        return None
    # null pool: union of axis-level neurons across all runs (top-trimmed -> CONSERVATIVE null,
    # same caveat as segmented_mlp_circuit_analysis.py).
    allneur = set()
    for r in runs:
        if not r["allg"].empty:
            ax = r["allg"][r["allg"]["level"] == "axis"]
            allneur |= set(zip(ax["layer"].astype(int), ax["neuron"].astype(int)))
    nm, _ = random_jaccard_null(max(len(allneur), top_k + 1), top_k)

    rows = []
    for cax in shared:
        present = [l for l in labels if cax in per_run_axis[l]]
        pairs = [jaccard(per_run_axis[a][cax], per_run_axis[b][cax])
                 for i, a in enumerate(present) for b in present[i + 1:]]
        rows.append({"axis": cax, "jaccard": float(np.mean(pairs)), "n_datasets": len(present)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "compare_axis_cross_dataset_neuron_jaccard.csv", index=False)

    pub_style()
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(shared) + 2), 5.2))
    colors = {a: plt.cm.tab10(i % 10) for i, a in enumerate(shared)}
    ax.bar(range(len(df)), df["jaccard"], color=[colors[a] for a in df["axis"]], width=0.72)
    ax.axhline(nm, ls="--", lw=1.4, color="#888", label=f"random null ≈ {nm:.4f}")
    for i, v in enumerate(df["jaccard"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["axis"], rotation=30, ha="right")
    ax.set_ylabel(f"cross-dataset Jaccard of top-{top_k} {metric} neurons")
    ax.set_title("Per identity axis: do datasets agree on the bias NEURONS?\n"
                 "(" + " vs ".join(labels) + ")")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "compare_axis_cross_dataset_neuron_jaccard.png"); plt.close(fig)
    return df, nm


def fig_axis_layer_profiles(runs: list[dict], metric: str, out_dir: Path):
    col = "suff_pos_mass" if metric == "suff_frac" else "nec_pos_mass"
    styles = {r["label"]: ls for r, ls in zip(runs, ["-", "--", ":", "-."])}
    # collect canonical axes present anywhere, colour shared across datasets
    profs = {}
    for r in runs:
        p = r["prof"]
        if p.empty:
            continue
        ax = p[p["level"] == "axis"].copy() if "level" in p.columns else \
            p.assign(level=p["group"].map(lambda g: parse_group(g)[0])).query("level=='axis'")
        ax["caxis"] = ax["group"].map(lambda g: canon(parse_group(g)[1]))
        profs[r["label"]] = ax
    if not profs:
        return
    all_axes = order_axes({a for ax in profs.values() for a in ax["caxis"]})
    acolor = {a: plt.cm.tab20(k % 20) for k, a in enumerate(all_axes)}
    pub_style()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, p in profs.items():
        # a canonical axis may merge >1 raw axis within a dataset (rare) -> mean over layer
        g = p.groupby(["caxis", "layer"])[col].mean().reset_index()
        for cax in order_axes(g["caxis"]):
            gg = g[g["caxis"] == cax].sort_values("layer")
            ax.plot(gg["layer"], gg[col], styles[label], lw=1.7, color=acolor[cax],
                    label=f"{label}:{cax}")
    ax.set_xlabel("layer"); ax.set_ylabel(f"Σ positive neuron attribution ({col})")
    ax.set_title("Where each axis writes bias at the NEURON level, across datasets\n"
                 "(colour = axis, line style = dataset)")
    ax.legend(fontsize=6, ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "compare_mlp_axis_layer_profiles.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare MLP-neuron attribution across runs/datasets.")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True,
                    help="per-group MLP dir (repeatable, >=2)")
    ap.add_argument("--pooled", nargs=2, action="append", metavar=("LABEL", "FILE"), default=[],
                    help="override pooled ranking path for a run LABEL")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["suff_frac", "nec_frac"], default="suff_frac")
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_over = {lab: Path(f) for lab, f in args.pooled}

    runs = [load_run(lab, Path(d), pooled_over.get(lab)) for lab, d in args.run]
    if len(runs) < 2:
        raise SystemExit("Need >=2 runs to compare.")

    res = fig_cross_dataset_neuron_overlap(runs, args.metric, args.top_k, args.out_dir)
    axres = fig_axis_cross_dataset_jaccard(runs, args.metric, args.top_k, args.out_dir)
    fig_axis_layer_profiles(runs, args.metric, args.out_dir)

    print(f"\nWrote comparison figures to {args.out_dir}")
    if res is not None:
        J, nm = res
        print(f"\n=== cross-dataset top-{args.top_k} {args.metric} NEURON Jaccard (null {nm:.4f}) ===")
        print(J.round(3).to_string())
    if axres is not None:
        df, nm = axres
        print(f"\n=== per-axis cross-dataset NEURON Jaccard (null {nm:.4f}) ===")
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
