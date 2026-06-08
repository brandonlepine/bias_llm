#!/usr/bin/env python3
"""Readable visualizations for the OOD steering-transfer sweeps (run_bbq_steering_transfer.py).

The built-in curve plots one line per layer (32 crisscrossing lines + a legend that eats the figure)
— impossible to read. This regenerates, from the existing bbq_steering_transfer_raw.csv files (no GPU):

  PER CONDITION (written into each transfer dir):
    transfer_heatmap.png      layer x alpha grid, color = Δbias vs the unsteered baseline. One glance
                              shows which layers steer and that the effect is monotonic in alpha.
    transfer_best_layers.png  only the top-K most effective layers as lines vs alpha, over the
                              norm-matched RANDOM control band (median + IQR). Shows linearity + that
                              real beats random.

  ACROSS CONDITIONS (written to --out_dir):
    transfer_compare_heatmaps.png   all conditions' heatmaps on a shared color scale (matched vs
                                    cross-construct at a glance).
    transfer_compare_summary.png    best INDUCE (Δ at α>0) and best DE-BIAS (Δ at α<0) per condition,
                                    with the random control's max |Δ| as a reference.

Usage:
  python scripts/plot_steering_transfer.py --dirs pod_results/.../steering_transfer/transfer_* \
      --out_dir pod_results/.../steering_transfer/_viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ABBR = {"sexual_orientation": "SO", "gender_identity": "GI", "Sexual_orientation": "SO_QA",
        "Gender_identity": "GI_QA"}


def condition_label(d: Path) -> str:
    """transfer_vec-<vecaxis>__<bbqfile>-<subcat>  ->  'SO → GI_QA·trans'."""
    name = d.name.replace("transfer_vec-", "")
    try:
        vec, rest = name.split("__", 1)
        bbq, sub = rest.rsplit("-", 1)
        return f"{ABBR.get(vec, vec)} → {ABBR.get(bbq, bbq)}·{sub}"
    except ValueError:
        return d.name


def load(d: Path, metric: str):
    """Return (real_grid[layer x alpha] of Δ vs baseline, baseline, random_by_alpha, layers, alphas)."""
    raw = pd.read_csv(d / "bbq_steering_transfer_raw.csv")
    alphas = sorted(raw.alpha.unique())
    layers = sorted(raw.layer.unique())
    real = raw[raw.kind == "real"]
    baseline = float(real[real.alpha == 0.0][metric].mean())
    grid = real.pivot_table(index="layer", columns="alpha", values=metric).reindex(index=layers, columns=alphas)
    delta = grid - baseline
    rnd = raw[raw.kind == "random"]                       # all (layer, seed) per alpha
    return delta, baseline, rnd, layers, alphas, metric


def heatmap(ax, delta, layers, alphas, vmax, title):
    im = ax.imshow(delta.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper", interpolation="nearest",
                   extent=[-0.5, len(alphas) - 0.5, len(layers) - 0.5, -0.5])
    ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=7)
    ax.set_yticks(range(0, len(layers), 4)); ax.set_yticklabels(layers[::4], fontsize=7)
    ax.set_xlabel("alpha"); ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    return im


def per_condition_figures(d: Path, metric: str, top_k: int):
    delta, baseline, rnd, layers, alphas, metric = load(d, metric)
    vmax = float(np.nanmax(np.abs(delta.values))) or 1e-6
    label = condition_label(d)

    fig, ax = plt.subplots(figsize=(5.2, 6))
    im = heatmap(ax, delta, layers, alphas, vmax,
                 f"OOD steering transfer  {label}\nΔ {metric} vs unsteered baseline ({baseline:.3f})")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    cb.set_label("Δ (red=more biased, blue=less)", fontsize=8)
    fig.tight_layout(); fig.savefig(d / "transfer_heatmap.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # top-K layers by peak |Δ| across alpha
    peak = delta.abs().max(axis=1).sort_values(ascending=False)
    top = list(peak.head(top_k).index)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    # random control band (median + IQR over all layers & seeds at each alpha), Δ vs baseline
    rg = rnd.groupby("alpha")[metric].agg(["median",
                                           lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    rg.columns = ["med", "q25", "q75"]
    ax.fill_between(rg.index, rg.q25 - baseline, rg.q75 - baseline, color="#bbb", alpha=0.5,
                    label="random IQR", zorder=1)
    ax.plot(rg.index, rg["med"] - baseline, "--", color="#777", lw=1.5, label="random median", zorder=2)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(top)))
    for c, L in zip(cmap, top):
        ax.plot(alphas, delta.loc[L].values, "-o", ms=4, lw=1.8, color=c, label=f"L{L}", zorder=3)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("alpha (coefficient on WinoQueer v_L)")
    ax.set_ylabel(f"Δ {metric} vs baseline ({baseline:.3f})")
    ax.set_title(f"OOD steering transfer — top {top_k} steering layers  {label}\n"
                 f"(real layers vs norm-matched random control)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(d / "transfer_best_layers.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    return label, delta, baseline, rnd, layers, alphas


def main() -> None:
    ap = argparse.ArgumentParser(description="Readable steering-transfer plots from existing raw CSVs.")
    ap.add_argument("--dirs", type=Path, nargs="+", required=True, help="transfer_* dirs (raw CSV inside)")
    ap.add_argument("--out_dir", type=Path, required=True, help="where cross-condition figures go")
    ap.add_argument("--metric", type=str, default="mean_p_biased")
    ap.add_argument("--top_k", type=int, default=4)
    args = ap.parse_args()

    dirs = [d for d in args.dirs if (d / "bbq_steering_transfer_raw.csv").exists()]
    if not dirs:
        raise SystemExit("no transfer dirs with bbq_steering_transfer_raw.csv found")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for d in dirs:
        label, delta, baseline, rnd, layers, alphas = per_condition_figures(d, args.metric, args.top_k)
        results.append((label, delta, baseline, layers, alphas))
        print(f"  {label}: wrote transfer_heatmap.png + transfer_best_layers.png in {d.name}")

    # shared color scale across conditions
    vmax = max(float(np.nanmax(np.abs(delta.values))) for _, delta, *_ in results) or 1e-6
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 6), squeeze=False)
    im = None
    for ax, (label, delta, baseline, layers, alphas) in zip(axes[0], results):
        im = heatmap(ax, delta, layers, alphas, vmax, label)
    fig.suptitle(f"OOD steering transfer — Δ{args.metric} vs baseline (shared scale)", fontsize=11)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="Δ (red=more biased)")
    fig.savefig(args.out_dir / "transfer_compare_heatmaps.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # summary bars: best induce / best de-bias per condition (+ random control max|Δ|)
    labels, induce, debias = [], [], []
    for d, (label, delta, baseline, layers, alphas) in zip(dirs, results):
        a = np.array(alphas)
        pos = delta.values[:, a > 0]; neg = delta.values[:, a < 0]
        labels.append(label)
        induce.append(float(np.nanmax(pos)))            # most +Δ at α>0
        debias.append(float(np.nanmin(neg)))            # most -Δ at α<0
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.7 * len(labels) + 3, 5))
    ax.bar(x - w/2, induce, w, color="#c0392b", label="best INDUCE (Δ at α>0)")
    ax.bar(x + w/2, debias, w, color="#2c7fb8", label="best DE-BIAS (Δ at α<0)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"Δ {args.metric} vs baseline")
    ax.set_title("Steering transfer effect by condition (matched vs cross-construct)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "transfer_compare_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote cross-condition figures -> {args.out_dir}/transfer_compare_heatmaps.png + transfer_compare_summary.png")


if __name__ == "__main__":
    main()
