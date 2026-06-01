#!/usr/bin/env python3
"""Sufficiency x necessity for WinoQueer attention heads.

Joins the head-PATCHING ranking (sufficiency: injecting a head's queer state into the control
run creates bias) with the head-ABLATION ranking (necessity: removing a head's queer behavior
in the queer run reduces bias). Heads that are high on BOTH are the core bias circuit.

  x = bias_effect  (sufficiency, from patching)
  y = ablation_effect (necessity, from ablation)
  size = ablation sign-consistency (reliability)  ->  small + unreliable heads shrink away
  core_score = z(bias_effect) + z(ablation_effect)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Sufficiency (patching) x necessity (ablation) head overlap.")
    ap.add_argument("--patching_csv", type=Path, required=True, help="winoqueer_head_circuit_ranking.csv")
    ap.add_argument("--ablation_csv", type=Path, required=True, help="winoqueer_head_ablation_ranking.csv")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=15)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pat = pd.read_csv(args.patching_csv)[["layer", "head", "bias_effect", "attn_readout_to_identity", "circuit_score"]]
    abl = pd.read_csv(args.ablation_csv)[["layer", "head", "ablation_effect", "consistency"]]
    m = pat.merge(abl, on=["layer", "head"], how="inner")

    bz = (m["bias_effect"] - m["bias_effect"].mean()) / (m["bias_effect"].std() + 1e-9)
    az = (m["ablation_effect"] - m["ablation_effect"].mean()) / (m["ablation_effect"].std() + 1e-9)
    m["core_score"] = bz + az
    m = m.sort_values("core_score", ascending=False).reset_index(drop=True)
    merged_csv = args.out_dir / "winoqueer_head_sufficiency_necessity.csv"
    m.to_csv(merged_csv, index=False)

    # Scatter: sufficiency vs necessity, size by reliability, color by layer.
    fig, ax = plt.subplots(figsize=(10, 8.5))
    cons = m["consistency"].fillna(0.5)
    sizes = 12 + 260 * ((cons - 0.5).clip(lower=0))  # >0.5 consistency grows; <=0.5 tiny
    sc = ax.scatter(m["bias_effect"], m["ablation_effect"], c=m["layer"], cmap="viridis",
                    s=sizes, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.axvline(0.0, color="#888", lw=0.8)
    for _, r in m.head(args.top_k).iterrows():
        ax.annotate(f"L{int(r['layer'])}H{int(r['head'])}", (r["bias_effect"], r["ablation_effect"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=9, fontweight="bold", color="#111")
    ax.set_xlabel("sufficiency:  bias_effect from PATCHING  (injecting head → creates bias)", fontsize=11)
    ax.set_ylabel("necessity:  ablation_effect  (removing head → reduces bias)", fontsize=11)
    ax.set_title("WinoQueer head circuit — sufficiency × necessity\n(marker size = ablation sign-consistency / reliability)", fontsize=12)
    ax.text(0.985, 0.97, "core circuit\n(sufficient + necessary)\n→ top-right", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color="#555",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("layer", fontsize=10)
    png = args.out_dir / "winoqueer_head_sufficiency_necessity.png"
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)

    print(f"Wrote {merged_csv}\nWrote {png}")
    print("\nTop core-circuit heads (high sufficiency AND necessity):")
    show = m.head(args.top_k)[["layer", "head", "bias_effect", "ablation_effect", "consistency", "attn_readout_to_identity", "core_score"]]
    print(show.to_string(index=False))
    # how aligned are sufficiency and necessity overall?
    print(f"\nSpearman(bias_effect, ablation_effect) over all 1024 heads: {m['bias_effect'].corr(m['ablation_effect'], method='spearman'):.3f}")


if __name__ == "__main__":
    main()
