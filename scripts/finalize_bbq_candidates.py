#!/usr/bin/env python3
"""Turn the Llama-scored BBQ pairs into a clean candidate set for patching.

Pruning (from the frame-agreement + no-bias analysis on Llama-3.1-8B):
  - keep ALL frames by default for statistical power (name_who/name_being are weaker but still
    ~67% biased; drop them only via --drop_frames if you want the cleanest-signal subset);
  - drop stereotypes the model RESISTS (mean bias_score < 0 across frames) — no signal HERE, but
    these "anti-biased" instances are worth a separate analysis (see GitHub issue); --keep_resisted
    retains them;
  - keep only the BIASED pairs (bias_score > 0) — where the model actually expresses the stereotype.

Outputs:
  data/bbq/stereotypes/bbq_candidates_final.csv     all kept candidate pairs (combined)
  data/bbq/stereotypes/bbq_scored_summary.csv        per-stereotype stats (mean bias, %biased, kept)
  data/bbq/results/<category>/patching_candidates/bbq_<category>_candidates.csv   per-category (tracked)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ["category", "Group_x", "Group_y", "predicate_label_provisional"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize BBQ patching candidates from scored pairs.")
    ap.add_argument("--scored", type=Path, default=Path("pod_results/05312026/bbq_scoring/bbq_pairs_scored.csv"))
    ap.add_argument("--out_final", type=Path, default=Path("data/bbq/stereotypes/bbq_candidates_final.csv"))
    ap.add_argument("--out_summary", type=Path, default=Path("data/bbq/stereotypes/bbq_scored_summary.csv"))
    ap.add_argument("--candidates_dir", type=Path, default=Path("data/bbq/results"))
    ap.add_argument("--drop_frames", type=str, default="", help="comma list of frames to drop (default: keep all)")
    ap.add_argument("--keep_resisted", action="store_true", help="keep stereotypes the model resists (mean bias<0)")
    args = ap.parse_args()
    DROP_FRAMES = {f.strip() for f in args.drop_frames.split(",") if f.strip()}
    args.out_final.parent.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.scored)

    # per-stereotype summary (across ALL frames, before pruning) — for transparency / drops
    summ = d.groupby(KEY).agg(
        n_pairs=("bias_score", "size"), mean_bias=("bias_score", "mean"),
        pct_biased=("bias_score", lambda s: round((s > 0).mean() * 100, 1)),
        n_frames_biased=("frame", lambda s: 0),  # placeholder, filled below
    ).reset_index()
    fb = (d.groupby(KEY + ["frame"])["bias_score"].mean() > 0).groupby(KEY).sum()
    summ["n_frames_biased"] = summ.set_index(KEY).index.map(fb).astype(int)
    summ["resisted"] = summ["mean_bias"] < 0
    summ = summ.sort_values("mean_bias", ascending=False)
    summ.to_csv(args.out_summary, index=False)

    keep_mask = summ["resisted"] if args.keep_resisted else ~summ["resisted"]
    keep_stereo = set(map(tuple, summ.loc[(summ["resisted"] == False) | args.keep_resisted, KEY].itertuples(index=False)))
    if not args.keep_resisted:
        keep_stereo = set(map(tuple, summ.loc[~summ["resisted"], KEY].itertuples(index=False)))
    d["_key"] = list(zip(*[d[k] for k in KEY]))
    fmask = ~d["frame"].isin(DROP_FRAMES) if DROP_FRAMES else pd.Series(True, index=d.index)
    cand = d[fmask & (d["bias_score"] > 0) & (d["_key"].isin(keep_stereo))].drop(columns="_key")
    cand = cand.sort_values("bias_score", ascending=False).reset_index(drop=True)
    cand.to_csv(args.out_final, index=False)

    for cat, g in cand.groupby("category"):
        dd = args.candidates_dir / cat / "patching_candidates"
        dd.mkdir(parents=True, exist_ok=True)
        g.to_csv(dd / f"bbq_{cat}_candidates.csv", index=False)

    print(f"Scored pairs: {len(d)} | stereotypes: {len(summ)} | resisted (dropped): {int(summ['resisted'].sum())}")
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_final}  ({len(cand)} candidate pairs after pruning weak frames + resisted + bias<=0)")
    print(f"Wrote per-category candidates under {args.candidates_dir}/<category>/patching_candidates/")
    print("\ncandidate pairs per axis:")
    print(cand.groupby("axis").size().sort_values(ascending=False).to_string())
    print("\ncandidate pairs per frame:")
    print(cand.groupby("frame").size().to_string())


if __name__ == "__main__":
    main()
