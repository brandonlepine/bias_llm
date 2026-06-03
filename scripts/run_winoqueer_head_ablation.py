#!/usr/bin/env python3
"""Attention-head ABLATION for WinoQueer — the NECESSITY complement to head patching.

Head patching showed *sufficiency* (injecting a head's queer state into the control run
creates the bias). Ablation shows *necessity*: if we remove a head's queer-specific behavior
in the QUEER run, how much of the stereotype probability disappears?

Resample ablation (clean, mirror of the patching): in the queer prompt, replace a head's
output (hook_z) with the CONTROL run's output for that head at the aligned positions. This
removes exactly the head's identity-driven contribution, using the same single-identity-span
token alignment as run_winoqueer_resid_patching.py.

    queer_avg   = continuation avg logp of the queer prompt (biased, high)
    control_avg = continuation avg logp of the control prompt (neutral, low)
    ablated_avg = queer prompt with head h's output replaced by control's
    ablation_effect[h]   = queer_avg - ablated_avg            (>0 => removing h reduces the bias)
    frac_bias_removed[h] = ablation_effect / (queer_avg - control_avg)   (1 => fully neutralized)

Outputs:
  1. SINGLE-HEAD necessity sweep  -> layer x head heatmap + reliability (volcano) + ranking.
  2. CUMULATIVE knockout          -> ablate the top-k most-necessary heads together; curve of
     bias remaining vs k (how concentrated the circuit is).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_winoqueer_resid_patching import align_pair, continuation_logp, resolve_device  # noqa: E402


RAW_HEADER = [
    "pair_id", "row_id", "Gender_ID_x", "Gender_ID_y", "predicate", "predicate_label_provisional",
    "layer", "head", "ablation_effect", "frac_bias_removed",
    "queer_cont_avg_logp", "control_cont_avg_logp",
]
CUM_HEADER = ["pair_id", "k", "frac_bias_remaining", "ablated_cont_avg_logp",
              "queer_cont_avg_logp", "control_cont_avg_logp"]


def queer_to_control_source(P: int, len_q: int, len_c: int) -> list[int]:
    """Inverse of the control->queer alignment: for each queer position, the control source."""
    delta = len_q - len_c
    return [q if q < P else min(max(q - delta, P), len_c - 1) for q in range(len_q)]


def load_model(args):
    from transformer_lens import HookedTransformer
    device = resolve_device(args.device)
    if device == "mps" and args.dtype == "bfloat16":
        args.dtype = "float16"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"\nDevice: {device} | dtype: {args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    print("Wrapping with TransformerLens...")
    model = HookedTransformer.from_pretrained(
        args.tl_model_name, hf_model=hf_model, tokenizer=tokenizer, device=device, dtype=dtype,
        fold_ln=False, center_writing_weights=False, center_unembed=False, default_prepend_bos=True,
    )
    model.eval()
    return model, tokenizer, device


def prep_pair(tokenizer, row, device):
    """Return alignment + tensors + control hook_z cache handle inputs, or None if unalignable."""
    aln = align_pair(tokenizer, str(row["sent_x"]), str(row["sent_y"]), str(row["prefix_y"]))
    if aln is None or not aln["ok"]:
        return None
    q_ids = torch.tensor([aln["q_ids"]], device=device)
    c_ids = torch.tensor([aln["c_ids"]], device=device)
    len_q, len_c = q_ids.shape[1], c_ids.shape[1]
    cont_count = aln["cont_count"]
    q_cont_start = len_q - cont_count
    qsrc = torch.tensor(queer_to_control_source(aln["P"], len_q, len_c), device=device)
    return dict(aln=aln, q_ids=q_ids, c_ids=c_ids, len_q=len_q, len_c=len_c,
                cont_count=cont_count, q_cont_start=q_cont_start, c_cont_start=aln["cont_start_c"], qsrc=qsrc)


def ablate_positions(p, mode: str, device) -> torch.Tensor:
    aln = p["aln"]
    if mode == "identity":
        # queer identity span = [P, P+Lx)
        return torch.arange(aln["P"], aln["P"] + aln["Lx"], device=device)
    if mode == "readout":
        return torch.tensor([p["q_cont_start"] - 1], device=device)
    return torch.arange(p["len_q"], device=device)


@torch.no_grad()
def run_single_head_ablation(args, model, tokenizer, device, pairs, raw_path):
    from transformer_lens import utils as tl_utils
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    hbs = max(1, int(args.head_batch_size))

    done: set[int] = set()
    resume = raw_path.exists() and not args.overwrite
    if resume:
        try:
            ex = pd.read_csv(raw_path)
        except Exception:
            ex = pd.DataFrame()
        present = sorted(int(p) for p in ex.get("pair_id", pd.Series([], dtype=int)).dropna().unique())
        if present:
            done = set(present[:-1])
            ex[ex["pair_id"].isin(done)].reindex(columns=RAW_HEADER).to_csv(raw_path, index=False)
            print(f"Resuming single-head: {len(done)} pairs done; redoing {present[-1]}.")
            f = raw_path.open("a", newline="", encoding="utf-8"); writer = csv.DictWriter(f, fieldnames=RAW_HEADER)
        else:
            resume = False
    if not resume:
        f = raw_path.open("w", newline="", encoding="utf-8"); writer = csv.DictWriter(f, fieldnames=RAW_HEADER); writer.writeheader()

    nf = lambda n: n.endswith("hook_z")
    skipped = 0
    try:
        for pair_id, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Single-head ablation"):
            pair_id = int(pair_id)
            if pair_id in done:
                continue
            p = prep_pair(tokenizer, row, device)
            if p is None:
                skipped += 1
                continue
            apos = ablate_positions(p, args.ablate_positions, device)
            _, ccache = model.run_with_cache(p["c_ids"], names_filter=nf)
            queer_avg = float(continuation_logp(model(p["q_ids"]), p["q_ids"][0], p["q_cont_start"])[0].item()) / p["cont_count"]
            control_avg = float(continuation_logp(model(p["c_ids"]), p["c_ids"][0], p["c_cont_start"])[0].item()) / p["cont_count"]
            denom = queer_avg - control_avg

            rows: list[dict[str, Any]] = []
            for layer in range(n_layers):
                zname = tl_utils.get_act_name("z", layer)
                cz_aligned = ccache[zname][0, p["qsrc"]]  # [len_q, H, d] control head outputs, queer-aligned
                abl = [float("nan")] * n_heads
                for hs in range(0, n_heads, hbs):
                    heads = list(range(hs, min(hs + hbs, n_heads)))
                    batched = p["q_ids"].repeat(len(heads), 1)

                    def hook(z, hook, heads=heads, cz_aligned=cz_aligned, apos=apos):
                        z = z.clone()
                        for i, h in enumerate(heads):
                            z[i, apos, h, :] = cz_aligned[apos, h, :].to(z.dtype)
                        return z

                    logits = model.run_with_hooks(batched, fwd_hooks=[(zname, hook)])
                    sums = continuation_logp(logits, p["q_ids"][0], p["q_cont_start"])
                    for i, h in enumerate(heads):
                        abl[h] = float(sums[i].item()) / p["cont_count"]
                for h in range(n_heads):
                    eff = queer_avg - abl[h]
                    rows.append({
                        "pair_id": pair_id, "row_id": row.get("row_id"),
                        "Gender_ID_x": row.get("Gender_ID_x"), "Gender_ID_y": row.get("Gender_ID_y"),
                        "predicate": row.get("predicate"), "predicate_label_provisional": row.get("predicate_label_provisional"),
                        "layer": layer, "head": h, "ablation_effect": eff,
                        "frac_bias_removed": (eff / denom) if abs(denom) > 1e-8 else float("nan"),
                        "queer_cont_avg_logp": queer_avg, "control_cont_avg_logp": control_avg,
                    })
            writer.writerows(rows); f.flush()
            del ccache
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        f.close()
    print(f"Single-head: skipped {skipped}/{len(pairs)}")
    return pd.read_csv(raw_path)


@torch.no_grad()
def run_cumulative_knockout(args, model, tokenizer, device, pairs, ranking, cum_path):
    """Ablate the top-k most-necessary heads TOGETHER (k=0..K) and record bias remaining."""
    from transformer_lens import utils as tl_utils
    ks = list(range(0, min(args.cumulative_max_heads, len(ranking)) + 1))
    with cum_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CUM_HEADER); writer.writeheader()
        for pair_id, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Cumulative knockout"):
            p = prep_pair(tokenizer, row, device)
            if p is None:
                continue
            apos = ablate_positions(p, args.ablate_positions, device)
            _, ccache = model.run_with_cache(p["c_ids"], names_filter=lambda n: n.endswith("hook_z"))
            queer_avg = float(continuation_logp(model(p["q_ids"]), p["q_ids"][0], p["q_cont_start"])[0].item()) / p["cont_count"]
            control_avg = float(continuation_logp(model(p["c_ids"]), p["c_ids"][0], p["c_cont_start"])[0].item()) / p["cont_count"]
            denom = queer_avg - control_avg
            cz_by_layer = {L: ccache[tl_utils.get_act_name("z", L)][0, p["qsrc"]] for L in {l for l, _ in ranking[:max(ks)]}}
            for k in ks:
                if k == 0:
                    ablated = queer_avg
                else:
                    by_layer = defaultdict(list)
                    for (L, H) in ranking[:k]:
                        by_layer[L].append(H)
                    fwd = []
                    for L, heads in by_layer.items():
                        cz = cz_by_layer[L]

                        def hook(z, hook, heads=heads, cz=cz, apos=apos):
                            z = z.clone()
                            for h in heads:
                                z[0, apos, h, :] = cz[apos, h, :].to(z.dtype)
                            return z

                        fwd.append((tl_utils.get_act_name("z", L), hook))
                    logits = model.run_with_hooks(p["q_ids"], fwd_hooks=fwd)
                    ablated = float(continuation_logp(logits, p["q_ids"][0], p["q_cont_start"])[0].item()) / p["cont_count"]
                writer.writerow({
                    "pair_id": int(pair_id), "k": k,
                    "frac_bias_remaining": ((ablated - control_avg) / denom) if abs(denom) > 1e-8 else float("nan"),
                    "ablated_cont_avg_logp": ablated, "queer_cont_avg_logp": queer_avg, "control_cont_avg_logp": control_avg,
                })
            f.flush(); del ccache
            if device == "cuda":
                torch.cuda.empty_cache()
    return pd.read_csv(cum_path)


def make_outputs(raw_df: pd.DataFrame, cum_df: pd.DataFrame | None, out_dir: Path):
    paths = []
    agg = raw_df.groupby(["layer", "head"]).agg(
        ablation_effect=("ablation_effect", "mean"),
        frac=("frac_bias_removed", "mean"),
        consistency=("ablation_effect", lambda s: float((s > 0).mean())),
        n=("ablation_effect", "size"),
    ).reset_index()
    agg = agg.sort_values("ablation_effect", ascending=False).reset_index(drop=True)
    rank_csv = out_dir / "head_ablation_ranking.csv"
    agg.to_csv(rank_csv, index=False); paths.append(rank_csv)

    # necessity heatmap (layer x head)
    pivot = agg.pivot(index="layer", columns="head", values="ablation_effect").sort_index().sort_index(axis=1)
    nL, nH = pivot.shape
    vmax = max(float(pd.Series(pivot.values.ravel()).abs().quantile(0.99)), 1e-9)
    fig, ax = plt.subplots(figsize=(max(9, nH * 0.34), max(7.5, nL * 0.30)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("Head ablation (control→queer): mean Δ logP drop  (>0 = head NECESSARY for bias)", fontsize=12)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(nH)); ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=7)
    ax.set_yticks(range(nL)); ax.set_yticklabels([str(int(i)) for i in pivot.index], fontsize=7)
    ax.set_xticks(np.arange(-0.5, nH, 1), minor=True); ax.set_yticks(np.arange(-0.5, nL, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.5); ax.tick_params(which="minor", length=0)
    layers, heads = list(pivot.index), list(pivot.columns)
    for _, r in agg.head(10).iterrows():
        if r["layer"] in layers and r["head"] in heads:
            xi, yi = heads.index(r["head"]), layers.index(r["layer"])
            ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="#00d000", linewidth=1.8))
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("mean ablation_effect")
    hp = out_dir / "head_ablation_heatmap.png"
    fig.tight_layout(); fig.savefig(hp, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(hp)

    # reliability volcano: mean effect vs sign-consistency
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(agg["ablation_effect"], agg["consistency"], c=agg["layer"], cmap="viridis", s=26, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, color="#888", ls="--", lw=0.8); ax.axvline(0.0, color="#888", lw=0.8)
    for _, r in agg.head(12).iterrows():
        ax.annotate(f"L{int(r['layer'])}H{int(r['head'])}", (r["ablation_effect"], r["consistency"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("mean ablation_effect (necessity)"); ax.set_ylabel("sign-consistency (fraction of pairs with effect>0)")
    ax.set_title("Head ablation reliability — robust necessary heads are top-right")
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("layer")
    vp = out_dir / "head_ablation_volcano.png"
    fig.tight_layout(); fig.savefig(vp, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(vp)

    # cumulative knockout curve
    if cum_df is not None and not cum_df.empty:
        curve = cum_df.groupby("k")["frac_bias_remaining"].agg(["mean", "std", "count"]).reset_index()
        curve.to_csv(out_dir / "head_knockout_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(curve["k"], curve["mean"], "-o", color="#c0392b", lw=2, ms=4)
        se = curve["std"] / curve["count"].clip(lower=1) ** 0.5
        ax.fill_between(curve["k"], curve["mean"] - se, curve["mean"] + se, color="#c0392b", alpha=0.2)
        ax.axhline(1.0, color="#999", ls=":", lw=1); ax.axhline(0.0, color="#999", ls=":", lw=1)
        ax.set_xlabel("# of top necessary heads ablated together"); ax.set_ylabel("fraction of bias remaining")
        ax.set_title("Cumulative knockout: how concentrated is the bias circuit?")
        kp = out_dir / "head_knockout_curve.png"
        fig.tight_layout(); fig.savefig(kp, dpi=160, bbox_inches="tight"); plt.close(fig)
        paths += [out_dir / "head_knockout_curve.csv", kp]
    return paths, agg


def main() -> None:
    parser = argparse.ArgumentParser(description="WinoQueer head ABLATION (necessity) + cumulative knockout.")
    parser.add_argument("--pairs_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_per_predicate", type=int, default=None)
    parser.add_argument("--head_batch_size", type=int, default=32)
    parser.add_argument("--ablate_positions", choices=["all", "identity", "readout"], default="all")
    parser.add_argument("--cumulative_max_pairs", type=int, default=300)
    parser.add_argument("--cumulative_max_heads", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_resort", action="store_true",
                        help="Consume --pairs_csv in file order (no bias_score re-sort / per-predicate cap). "
                             "Use with a pre-frozen cohort so pair_id == cohort row order.")
    args = parser.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "head_ablation_raw.csv"
    cum_path = args.out_dir / "head_knockout_raw.csv"

    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"--plot_only needs {raw_path}")
        raw_df = pd.read_csv(raw_path)
        cum_df = pd.read_csv(cum_path) if cum_path.exists() else None
    else:
        if args.pairs_csv is None:
            parser.error("--pairs_csv required unless --plot_only")
        pairs = pd.read_csv(args.pairs_csv)
        if args.no_resort:
            pairs = pairs.reset_index(drop=True)
            print(f"Pairs: {len(pairs)} (no_resort: consuming cohort in file order; ablate_positions={args.ablate_positions})")
        else:
            pairs = pairs.sort_values("bias_score", ascending=False)
            if args.max_per_predicate is not None and "predicate" in pairs.columns:
                pairs = pairs.groupby("predicate", sort=False, group_keys=False).head(args.max_per_predicate).sort_values("bias_score", ascending=False)
            if args.max_pairs is not None:
                pairs = pairs.head(args.max_pairs)
            pairs = pairs.reset_index(drop=True)
            print(f"Pairs: {len(pairs)} (ablate_positions={args.ablate_positions})")
        model, tokenizer, device = load_model(args)

        raw_df = run_single_head_ablation(args, model, tokenizer, device, pairs, raw_path)
        ranking = [(int(r.layer), int(r.head)) for r in
                   raw_df.groupby(["layer", "head"])["ablation_effect"].mean().reset_index()
                   .sort_values("ablation_effect", ascending=False).itertuples()]
        cum_pairs = pairs.head(min(args.cumulative_max_pairs, len(pairs)))
        print(f"\nCumulative knockout over {len(cum_pairs)} pairs, top-{args.cumulative_max_heads} heads...")
        cum_df = run_cumulative_knockout(args, model, tokenizer, device, cum_pairs, ranking, cum_path)

    out_paths, agg = make_outputs(raw_df, cum_df, args.out_dir)

    print("\nWrote:")
    for p in [raw_path, cum_path] + out_paths:
        print(f"  {p}")
    print("\nTop 12 NECESSARY heads (ablating them most reduces the bias):")
    print(agg.head(12)[["layer", "head", "ablation_effect", "frac", "consistency"]].to_string(index=False))
    if cum_df is not None and not cum_df.empty:
        curve = cum_df.groupby("k")["frac_bias_remaining"].mean()
        for k in [1, 3, 5, 10, 20]:
            if k in curve.index:
                print(f"  ablating top-{k:>2} heads -> {curve[k]*100:.1f}% of bias remains")
    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
