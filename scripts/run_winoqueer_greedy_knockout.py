#!/usr/bin/env python3
"""Greedy attention-head knockout for WinoQueer — the rigorous test of circuit redundancy.

The cumulative knockout in run_winoqueer_head_ablation.py ablates heads ranked by their
*marginal* single-head necessity, which ignores interactions/redundancy. Greedy knockout
instead, at each step, adds the head that — GIVEN the already-ablated set — most reduces the
remaining bias. If even greedy selection leaves lots of bias after K heads, the circuit is
genuinely distributed; if greedy drops fast, the marginal ordering was just suboptimal.

Ablation = resample (control→queer): replace a head's queer-run output with the control-run
output at the aligned positions (same alignment/metric as the patching + ablation scripts).

Efficiency: at each step every remaining candidate is evaluated in ONE batched forward per
pair (batch row i ablates selected ∪ {candidate_i}), so cost ≈ K × n_pairs forwards.

Outputs: greedy selection order + knockout curve (CSV + PNG), overlaid on the marginal-ranked
curve when --marginal_curve_csv is given.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_winoqueer_head_ablation import prep_pair, ablate_positions, load_model, continuation_logp  # noqa: E402


@torch.no_grad()
def precompute(model, tokenizer, device, pairs, candidate_heads, ablate_mode):
    """For each pair, cache the control head outputs (aligned to queer positions) for the
    candidate heads, plus the queer/control baselines and metadata."""
    from transformer_lens import utils as tl_utils
    cand_layers = sorted({L for L, _ in candidate_heads})
    store = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Precompute"):
        p = prep_pair(tokenizer, row, device)
        if p is None:
            continue
        apos = ablate_positions(p, ablate_mode, device)
        _, ccache = model.run_with_cache(p["c_ids"], names_filter=lambda n: n.endswith("hook_z"))
        queer_avg = float(continuation_logp(model(p["q_ids"]), p["q_ids"][0], p["q_cont_start"])[0].item()) / p["cont_count"]
        control_avg = float(continuation_logp(model(p["c_ids"]), p["c_ids"][0], p["c_cont_start"])[0].item()) / p["cont_count"]
        denom = queer_avg - control_avg
        if abs(denom) < 1e-6:
            continue
        # control head output, queer-aligned, per candidate head: [len_q, d_head]
        cz = {}
        for L in cand_layers:
            aligned = ccache[tl_utils.get_act_name("z", L)][0, p["qsrc"]]  # [len_q, H, d]
            for (l, h) in candidate_heads:
                if l == L:
                    cz[(L, h)] = aligned[:, h, :].contiguous()
        store.append(dict(q_ids=p["q_ids"], apos=apos, q_cont_start=p["q_cont_start"],
                          cont_count=p["cont_count"], control_avg=control_avg, denom=denom, cz=cz))
        del ccache
    return store


@torch.no_grad()
def eval_candidates(model, store, selected, remaining):
    """Mean fraction-of-bias-remaining for ablating selected∪{cand} for each cand in remaining,
    averaged over all pairs. One batched forward per pair."""
    from transformer_lens import utils as tl_utils
    B = len(remaining)
    totals = torch.zeros(B)
    counts = 0
    for s in store:
        q = s["q_ids"]; apos = s["apos"]; cz = s["cz"]
        involved = {L for L, _ in selected} | {L for L, _ in remaining}
        fwd = []
        for L in involved:
            sel_h = [h for (l, h) in selected if l == L]
            cand_rows = [(i, h) for i, (l, h) in enumerate(remaining) if l == L]

            def hook(z, hook, L=L, sel_h=sel_h, cand_rows=cand_rows, cz=cz, apos=apos):
                z = z.clone()
                for h in sel_h:
                    z[:, apos, h, :] = cz[(L, h)][apos].to(z.dtype)
                for i, h in cand_rows:
                    z[i, apos, h, :] = cz[(L, h)][apos].to(z.dtype)
                return z

            fwd.append((tl_utils.get_act_name("z", L), hook))
        logits = model.run_with_hooks(q.repeat(B, 1), fwd_hooks=fwd)
        sums = continuation_logp(logits, q[0], s["q_cont_start"])  # [B]
        avg = sums / s["cont_count"]
        frac = (avg.cpu() - s["control_avg"]) / s["denom"]
        totals += frac
        counts += 1
    return (totals / max(counts, 1)).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="WinoQueer greedy head knockout.")
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ablation_ranking_csv", type=Path, required=True,
                    help="winoqueer_head_ablation_ranking.csv — candidate pool = its top heads.")
    ap.add_argument("--marginal_curve_csv", type=Path, default=None,
                    help="winoqueer_head_knockout_curve.csv to overlay for comparison.")
    ap.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--max_per_predicate", type=int, default=None)
    ap.add_argument("--candidate_pool", type=int, default=60, help="# top necessary heads considered.")
    ap.add_argument("--greedy_steps", type=int, default=30)
    ap.add_argument("--ablate_positions", choices=["all", "identity", "readout"], default="all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = args.out_dir / "winoqueer_greedy_knockout_curve.csv"

    # candidate pool = top-N heads by single-head necessity
    arank = pd.read_csv(args.ablation_ranking_csv).sort_values("ablation_effect", ascending=False)
    candidate_heads = [(int(r.layer), int(r.head)) for r in arank.head(args.candidate_pool).itertuples()]

    pairs = pd.read_csv(args.pairs_csv).sort_values("bias_score", ascending=False)
    if args.max_per_predicate is not None and "predicate" in pairs.columns:
        pairs = pairs.groupby("predicate", sort=False, group_keys=False).head(args.max_per_predicate).sort_values("bias_score", ascending=False)
    pairs = pairs.head(args.max_pairs).reset_index(drop=True)
    print(f"Pairs: {len(pairs)} | candidate pool: {len(candidate_heads)} | greedy steps: {args.greedy_steps}")

    model, tokenizer, device = load_model(args)
    store = precompute(model, tokenizer, device, pairs, candidate_heads, args.ablate_positions)
    print(f"Usable pairs: {len(store)}")

    # resume
    selected: list[tuple[int, int]] = []
    rows = []
    if curve_csv.exists() and not args.overwrite:
        prev = pd.read_csv(curve_csv)
        for r in prev.itertuples():
            if int(r.step) >= 1:
                selected.append((int(r.layer), int(r.head)))
                rows.append({"step": int(r.step), "layer": int(r.layer), "head": int(r.head),
                             "mean_frac_remaining": float(r.mean_frac_remaining)})
        print(f"Resuming greedy from step {len(selected)+1} ({len(selected)} heads already chosen).")
    if not rows:
        rows.append({"step": 0, "layer": -1, "head": -1, "mean_frac_remaining": 1.0})

    K = min(args.greedy_steps, len(candidate_heads))
    for step in range(len(selected) + 1, K + 1):
        remaining = [h for h in candidate_heads if h not in selected]
        means = eval_candidates(model, store, selected, remaining)
        best_i = int(min(range(len(remaining)), key=lambda i: means[i]))
        best_head = remaining[best_i]
        selected.append(best_head)
        rows.append({"step": step, "layer": best_head[0], "head": best_head[1], "mean_frac_remaining": means[best_i]})
        pd.DataFrame(rows).to_csv(curve_csv, index=False)  # checkpoint each step
        print(f"  step {step:>2}: +L{best_head[0]}H{best_head[1]} -> {means[best_i]*100:.1f}% bias remaining")
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- plot ----
    g = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(g["step"], g["mean_frac_remaining"], "-o", color="#16a085", lw=2.2, ms=4, label="GREEDY (optimal next head)")
    if args.marginal_curve_csv and args.marginal_curve_csv.exists():
        m = pd.read_csv(args.marginal_curve_csv)
        ax.plot(m["k"], m["mean"], "-s", color="#c0392b", lw=2, ms=3, alpha=0.8, label="marginal-ranked (single-head order)")
    ax.axhline(1.0, color="#999", ls=":", lw=1); ax.axhline(0.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("# of heads ablated together"); ax.set_ylabel("fraction of bias remaining")
    ax.set_title("Greedy vs marginal head knockout — how concentrated is the bias circuit?")
    ax.legend()
    png = args.out_dir / "winoqueer_greedy_knockout_curve.png"
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)

    print(f"\nWrote {curve_csv}\nWrote {png}")
    print("\nGreedy-selected heads in order:")
    print(g[g["step"] >= 1][["step", "layer", "head", "mean_frac_remaining"]].to_string(index=False))
    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
