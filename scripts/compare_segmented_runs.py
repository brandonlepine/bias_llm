#!/usr/bin/env python3
"""Cross-dataset / cross-run comparison of segmented_circuit_analysis outputs.

Takes >=2 analysis output dirs (e.g. WinoQueer and the combined BBQ+CrowS run) and overlays their
per-axis metrics on shared figures, so you can compare circuit structure ACROSS datasets:

  - READ-vs-WRITE for ALL axes from every run (is the two-stage circuit universal everywhere?)
  - WRITE selectivity for ALL axes (how identity-specific, dataset by dataset)
  - per-(axis,layer) WRITE profile overlaid (where each axis writes bias, across datasets)
  - cross-DATASET top-WRITE-head Jaccard: do the datasets' overall bias circuits overlap?
    (the headline "is queerness bias written by the same heads as race/religion/... bias?")

Usage:  --run winoqueer DIR1 --run bbq_crows DIR2  [--top_k 30]
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


# Shared categorical axis order so a given axis occupies the same slot / colour in EVERY dataset's
# bar group and line. (Value-sorting per dataset made the same axis unalignable across blocks.)
# Unknown axes sort last, alphabetically.
AXIS_ORDER = ["sexual_orientation", "orientation", "gender_identity", "gender", "gender_binary",
              "race", "race_ethnicity", "religion", "age", "nationality", "socioeconomic", "ses",
              "physical_appearance", "disability", "disability_status"]


def axis_rank(a):
    a = str(a)
    return (AXIS_ORDER.index(a) if a in AXIS_ORDER else len(AXIS_ORDER), a)


def order_axes(values):
    return sorted(set(map(str, values)), key=axis_rank)


def load(dirs: dict[str, Path]) -> dict[str, dict]:
    out = {}
    for label, d in dirs.items():
        rd = lambda f: pd.read_csv(d / f) if (d / f).exists() else pd.DataFrame()
        out[label] = {"rw": rd("read_vs_write_within_axis.csv"), "sel": rd("identity_selectivity_within_axis.csv"),
                      "pooled": rd("pooled_head_ranking.csv"), "axis": rd("head_stats__axis.csv")}
    return out


def fig_read_vs_write_all(data, out_dir):
    pub_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    rows = []
    for label, d in data.items():
        if d["rw"].empty:
            continue
        g = d["rw"].groupby("axis")["spearman_read_write"].mean()
        for axis, v in g.items():
            rows.append({"label": label, "axis": f"{axis}", "val": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["axis"] = pd.Categorical(df["axis"], categories=order_axes(df["axis"]), ordered=True)
    df = df.sort_values(["label", "axis"])
    labels = list(df["label"].unique())
    cmap = {lab: plt.cm.Set1(i) for i, lab in enumerate(labels)}
    xs = []
    pos = 0
    for lab in labels:
        sub = df[df.label == lab]
        ax.bar(np.arange(len(sub)) + pos, sub["val"], color=cmap[lab], label=lab, width=0.9)
        xs += [(pos + i, a) for i, a in enumerate(sub["axis"])]
        pos += len(sub) + 1
    ax.set_xticks([p for p, _ in xs]); ax.set_xticklabels([a for _, a in xs], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Spearman(READ, WRITE) over heads")
    ax.set_title("Two-stage circuit across DATASETS — low everywhere ⇒ reading ≠ writing is universal")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "compare_read_vs_write_all_axes.png"); plt.close(fig)


def fig_selectivity_all(data, out_dir):
    pub_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    rows = []
    for label, d in data.items():
        if d["sel"].empty:
            continue
        s = d["sel"].dropna(subset=["selectivity_write"])
        s = s[s["max_write"] >= s["max_write"].quantile(0.8)]
        for axis, g in s.groupby("axis"):
            rows.append({"label": label, "axis": axis, "median_sel": g["selectivity_write"].median()})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["axis"] = pd.Categorical(df["axis"], categories=order_axes(df["axis"]), ordered=True)
    df = df.sort_values(["label", "axis"])
    labels = list(df["label"].unique())
    cmap = {lab: plt.cm.Set1(i) for i, lab in enumerate(labels)}
    xs = []; pos = 0
    for lab in labels:
        sub = df[df.label == lab]
        ax.bar(np.arange(len(sub)) + pos, sub["median_sel"], color=cmap[lab], label=lab, width=0.9)
        xs += [(pos + i, a) for i, a in enumerate(sub["axis"])]
        pos += len(sub) + 1
    ax.set_xticks([p for p, _ in xs]); ax.set_xticklabels([a for _, a in xs], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("median WRITE selectivity (strong heads)")
    ax.set_title("How identity-specific are the bias-writing heads, per axis, across datasets?")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "compare_selectivity_all_axes.png"); plt.close(fig)


def fig_cross_dataset_head_overlap(data, top_k, out_dir):
    """Top-WRITE-head Jaccard BETWEEN datasets' overall circuits — do they share heads?"""
    labels = [l for l in data if not data[l]["pooled"].empty]
    if len(labels) < 2:
        return None
    sets, pool = {}, 0
    for lab in labels:
        p = data[lab]["pooled"].sort_values("write", ascending=False).head(top_k)
        sets[lab] = set((int(r.layer), int(r.head)) for r in p.itertuples())
        pool = max(pool, data[lab]["pooled"].groupby(["layer", "head"]).ngroups)
    J = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            J.loc[a, b] = jaccard(sets[a], sets[b])
    J.to_csv(out_dir / "compare_cross_dataset_head_jaccard.csv")
    nm, _ = random_jaccard_null(pool, top_k)
    pub_style()
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 1.4 * len(labels) + 2.5))
    M = J.to_numpy(float)
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontweight="bold",
                    color="white" if M[i, j] < 0.6 else "#111")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(f"Do datasets share a bias-writing circuit?\nJaccard of top-{top_k} WRITE heads  ·  random null ≈ {nm:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Jaccard")
    fig.tight_layout(); fig.savefig(out_dir / "compare_cross_dataset_head_jaccard.png"); plt.close(fig)
    return J, nm


def fig_axis_layer_profiles(data, out_dir):
    pub_style()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    styles = {lab: ls for lab, ls in zip(data, ["-", "--", ":"])}
    # colour keyed by axis (shared across datasets) so the SAME axis is the same colour in both
    # datasets; line style encodes the dataset.
    all_axes = order_axes({a for d in data.values() if not d["axis"].empty for a in d["axis"]["group"]})
    acolor = {a: plt.cm.tab20(k % 20) for k, a in enumerate(all_axes)}
    for label, d in data.items():
        if d["axis"].empty:
            continue
        prof = d["axis"].assign(w=d["axis"]["write"].clip(lower=0)).groupby(["group", "layer"])["w"].mean().reset_index()
        for axis in order_axes(prof["group"]):
            g = prof[prof["group"].astype(str) == axis].sort_values("layer")
            ax.plot(g["layer"], g["w"], styles[label], lw=1.6, color=acolor[axis], label=f"{label}:{axis}")
    ax.set_xlabel("layer"); ax.set_ylabel("mean positive WRITE over heads")
    ax.set_title("Where each axis writes bias, across datasets (colour = axis, line style = dataset)")
    ax.legend(fontsize=6, ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "compare_axis_layer_profiles.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare segmented analysis across runs/datasets.")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"), required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = {lab: Path(d) for lab, d in args.run}
    data = load(dirs)

    fig_read_vs_write_all(data, args.out_dir)
    fig_selectivity_all(data, args.out_dir)
    res = fig_cross_dataset_head_overlap(data, args.top_k, args.out_dir)
    fig_axis_layer_profiles(data, args.out_dir)

    print(f"Wrote comparison figures to {args.out_dir}")
    if res is not None:
        J, nm = res
        print(f"\n=== cross-dataset top-{args.top_k} WRITE-head Jaccard (random null {nm:.3f}) ===")
        print(J.round(3).to_string())


if __name__ == "__main__":
    main()
