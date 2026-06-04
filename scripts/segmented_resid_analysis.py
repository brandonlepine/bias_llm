#!/usr/bin/env python3
"""Dataset-agnostic 'WHERE in the model is the bias injected' analysis (resid-stream patching).

Companion to segmented_circuit_analysis.py (which answers WHICH heads/neurons). This consumes the
per-(pair, layer, token) resid_pre patching raw and the cohort, joined on `row_id`, and asks at which
LAYER and via which token SPAN each social axis's bias gets written into the residual stream:

  identity_all span (token_position = -1): inject the whole queer/target identity state at layer L
  -> Δ logP(stereotype).  The per-layer curve of this, one line per axis, is the headline:
  "at what depth does race vs age vs religion bias appear?"

Plus per-axis layer×span heatmaps and a per-SOURCE (BBQ vs CrowS) split. Uses normalized_restoration
(frac of the bias gap) so axes with different baseline gaps are comparable.
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
from segmented_circuit_analysis import detect_spec, attach_groups  # noqa: E402
from winoqueer_identity_taxonomy import pub_style  # noqa: E402

LBL = ""  # dataset label stamped on figures (--label)
def _save(fig, path, **kw):
    if LBL:
        fig.text(0.005, 0.995, LBL, ha="left", va="top", fontsize=10, fontweight="bold", color="#444", bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#ccc", alpha=0.9))
    fig.savefig(path, **kw); plt.close(fig)

SPAN_ORDER = ["shared_pre", "identity", "shared_post", "continuation"]
VAL = "normalized_restoration"


def load(resid_raw: Path, cohort: Path, spec) -> pd.DataFrame:
    r = pd.read_csv(resid_raw)
    if VAL not in r.columns:
        r[VAL] = r["bias_effect"]
    coh = attach_groups(pd.read_csv(cohort), spec)
    keep = ["row_id", "identity", "axis", "block", "source", "is_umbrella", "identity_mapped"]
    return r.merge(coh[keep].drop_duplicates("row_id"), on="row_id", how="inner")


def plot_identity_layer_by_axis(long: pd.DataFrame, out_dir: Path, source: str | None = None):
    d = long[long["span"] == "identity_all"] if "identity_all" in set(long["span"]) else long[long["span"] == "identity"]
    if source:
        d = d[d["source"] == source]
    if d.empty:
        return
    prof = d.groupby(["axis", "layer"])[VAL].mean().reset_index()
    tag = f"_{source}" if source else ""
    prof.to_csv(out_dir / f"resid_identity_layer_by_axis{tag}.csv", index=False)
    pub_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10.colors
    for i, (axis, g) in enumerate(prof.groupby("axis")):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g[VAL], "-o", ms=3, lw=2, color=cmap[i % 10], label=axis)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("Δ logP(stereotype) from injecting the identity state\n(normalized; frac of the bias gap)")
    ttl = "Where each axis's bias is injected into the residual stream"
    ax.set_title(ttl + (f"  —  {source}" if source else ""))
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(); _save(fig, out_dir / f"resid_identity_layer_by_axis{tag}.png")


def plot_span_heatmaps(long: pd.DataFrame, out_dir: Path):
    d = long[long["span"].isin(SPAN_ORDER)]
    axes = sorted(d["axis"].unique())
    if not axes:
        return
    pub_style()
    ncol = 3
    nrow = (len(axes) + ncol - 1) // ncol
    fig, axarr = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.0), squeeze=False)
    vmax = max(float(d[VAL].abs().quantile(0.97)), 1e-9)
    for ax, axis in zip(axarr.ravel(), axes):
        piv = d[d.axis == axis].groupby(["layer", "span"])[VAL].mean().unstack("span").reindex(columns=SPAN_ORDER).sort_index()
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(axis, fontsize=9)
        ax.set_xticks(range(len(SPAN_ORDER))); ax.set_xticklabels([s[:6] for s in SPAN_ORDER], rotation=45, ha="right", fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axarr.ravel()[len(axes):]:
        ax.axis("off")
    fig.colorbar(im, ax=axarr.ravel().tolist(), fraction=0.02, pad=0.02).set_label("mean normalized bias_effect", fontsize=8)
    fig.suptitle("Per-axis resid patching: layer × prompt span (where bias enters)", fontsize=12)
    _save(fig, out_dir / "resid_span_heatmaps_by_axis.png", bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-agnostic resid-patching 'where' analysis.")
    ap.add_argument("--resid_raw", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--label", type=str, default="", help="dataset label stamped on figures")
    ap.add_argument("--identity_col", type=str, default=None)
    ap.add_argument("--axis_col", type=str, default=None)
    ap.add_argument("--block_col", type=str, default=None)
    ap.add_argument("--source_col", type=str, default=None)
    ap.add_argument("--umbrella", type=str, default=None)
    args = ap.parse_args()
    global LBL
    LBL = args.label
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = detect_spec(pd.read_csv(args.cohort, nrows=200), args)
    long = load(args.resid_raw, args.cohort, spec)
    print(f"resid rows: {len(long)} | pairs: {long['row_id'].nunique()} | axes: {sorted(long['axis'].unique())} | "
          f"sources: {sorted(long['source'].unique())}")

    plot_identity_layer_by_axis(long, args.out_dir)
    plot_span_heatmaps(long, args.out_dir)
    for src in sorted(long["source"].unique()):
        if long["source"].nunique() > 1:
            plot_identity_layer_by_axis(long, args.out_dir, source=src)

    # peak-layer summary (where each axis's identity-injection bias is strongest)
    d = long[long["span"].isin(["identity_all", "identity"])]
    prof = d.groupby(["axis", "layer"])[VAL].mean().reset_index()
    peak = prof.loc[prof.groupby("axis")[VAL].idxmax()][["axis", "layer", VAL]].rename(columns={"layer": "peak_layer"})
    peak.to_csv(args.out_dir / "resid_peak_layer_by_axis.csv", index=False)
    print("\n=== peak injection layer per axis (where each axis's bias is strongest) ===")
    print(peak.sort_values("peak_layer").round(3).to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
