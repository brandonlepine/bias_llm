#!/usr/bin/env python3
"""Compare per-group greedy head-knockout runs for WinoQueer (CPU; post-processing).

Each per-group run of run_winoqueer_greedy_knockout.py writes a curve CSV (step, layer, head,
mean_frac_remaining) and, with --seeds>1, a selection-frequency CSV. This script overlays the
group curves, builds a Jaccard matrix of the selected-head sets at a matched k, runs the
umbrella = union-of-specifics test on the selected sets, and (when given) plots selection-frequency
robustness bars.

Usage:
  --curve LABEL path/to/winoqueer_greedy_knockout_curve.csv   (repeatable)
  --freq  LABEL path/to/winoqueer_greedy_selection_frequency.csv   (optional, repeatable)
Labels matching the umbrella terms (Queer / LGBTQ) drive the union test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winoqueer_identity_taxonomy import UMBRELLA, jaccard  # noqa: E402


def selected_at_k(curve: pd.DataFrame, k: int) -> set[tuple[int, int]]:
    d = curve[curve["step"] >= 1].sort_values("step").head(k)
    return set((int(r.layer), int(r.head)) for r in d.itertuples())


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare per-group WinoQueer greedy knockout runs.")
    ap.add_argument("--curve", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True)
    ap.add_argument("--freq", nargs=2, action="append", metavar=("LABEL", "PATH"), default=[])
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10, help="Matched k for the selected-set Jaccard/union tests.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = {label: pd.read_csv(path) for label, path in args.curve}

    # ---- overlay curves ----
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for label, c in curves.items():
        ax.plot(c["step"], c["mean_frac_remaining"], "-o", ms=3, lw=1.8, label=label)
    ax.axhline(1.0, color="#999", ls=":", lw=1); ax.axhline(0.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("# heads ablated"); ax.set_ylabel("fraction of bias remaining")
    ax.set_title("Greedy knockout per group — circuit concentration by identity/axis")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "segmented_greedy_curves.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- Jaccard of selected-head sets at matched k ----
    labels = list(curves)
    sets = {lab: selected_at_k(c, args.k) for lab, c in curves.items()}
    J = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            J.loc[a, b] = jaccard(sets[a], sets[b])
    J.to_csv(args.out_dir / "segmented_greedy_selected_jaccard.csv")
    if len(labels) >= 2:
        fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.0 * len(labels) + 2.5))
        im = ax.imshow(J.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"Jaccard of greedy-selected heads at k={args.k}")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        fig.tight_layout(); fig.savefig(args.out_dir / "segmented_greedy_selected_jaccard.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- umbrella = union-of-specifics? ----
    umb_labels = [l for l in labels if l in UMBRELLA]
    spec_labels = [l for l in labels if l not in UMBRELLA]
    if umb_labels and spec_labels:
        union = set().union(*(sets[l] for l in spec_labels))
        rows = [{"umbrella": l, "jaccard_with_union_of_specifics": jaccard(sets[l], union),
                 "n_union_heads": len(union)} for l in umb_labels]
        pd.DataFrame(rows).to_csv(args.out_dir / "segmented_greedy_umbrella_union.csv", index=False)

    # ---- selection-frequency robustness bars ----
    for label, path in args.freq:
        f = pd.read_csv(path).sort_values("selection_frequency", ascending=False).head(30).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, max(4, len(f) * 0.3)))
        ax.barh(range(len(f)), f["selection_frequency"], color="#16a085")
        ax.set_yticks(range(len(f)))
        ax.set_yticklabels([f"L{int(l)}H{int(h)}" for l, h in zip(f["layer"], f["head"])], fontsize=8)
        ax.set_xlabel("selection frequency"); ax.set_title(f"Greedy selection frequency — {label}")
        fig.tight_layout(); fig.savefig(args.out_dir / f"segmented_greedy_freq__{label}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    print(f"Wrote greedy comparison outputs to {args.out_dir}")
    print("\nSelected-head Jaccard at k=%d:" % args.k)
    print(J.to_string())


if __name__ == "__main__":
    main()
