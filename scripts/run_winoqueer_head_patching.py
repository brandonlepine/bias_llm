#!/usr/bin/env python3
"""Attention-HEAD analysis for WinoQueer: localize the stereotype circuit to specific heads.

Two complementary analyses over the same token-aligned queer/control pairs used by
run_winoqueer_resid_patching.py:

1. HEAD-LEVEL ACTIVATION PATCHING (which heads *write* the bias)
   Patch each head's output (hook_z) from the queer run into the control run (at the aligned
   positions) and re-score the continuation:
       bias_effect[layer, head] = patched_continuation_avg_logp - control_continuation_avg_logp
   >0 => replacing this head's output with its queer-run value raises the stereotype prob.

2. ATTENTION-PATTERN ANALYSIS (which heads *move* the identity signal)
   In the queer prompt, measure how much each head attends FROM the read-out position (the
   token that predicts the predicate) TO the queer identity token(s):
       attn_readout_to_identity[layer, head] = sum_k pattern[head, readout, identity_k]
   High => this head reads the identity when producing the stereotype.

Heads high on BOTH are the core bias circuit (read identity -> write stereotype).

Alignment + metric are imported from run_winoqueer_resid_patching.py so the stringent
single-identity-span alignment is identical.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
    "layer", "head", "bias_effect", "attn_readout_to_identity", "n_identity_tokens",
    "control_cont_avg_logp", "queer_cont_avg_logp", "normalized_restoration",
]


def safe_norm(num: float, denom: float) -> float:
    return num / denom if abs(denom) > 1e-8 else float("nan")


@torch.no_grad()
def run_head_patching(args, pairs: pd.DataFrame, raw_out_path: Path) -> pd.DataFrame:
    from transformer_lens import HookedTransformer, utils as tl_utils

    device = resolve_device(args.device)
    if device == "mps" and args.dtype == "bfloat16":
        args.dtype = "float16"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"\nDevice: {device} | dtype: {args.dtype} | head_batch_size: {args.head_batch_size} | positions: {args.patch_positions}")

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
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    hbs = max(1, int(args.head_batch_size))
    print(f"n_layers={n_layers} n_heads={n_heads}")

    # ---- resume (per-pair atomic) ----
    done_pair_ids: set[int] = set()
    resume = raw_out_path.exists() and not args.overwrite
    if resume:
        try:
            existing = pd.read_csv(raw_out_path)
        except Exception:
            existing = pd.DataFrame()
        present = sorted(int(p) for p in existing.get("pair_id", pd.Series([], dtype=int)).dropna().unique())
        if present:
            done_pair_ids = set(present[:-1])
            existing[existing["pair_id"].isin(done_pair_ids)].reindex(columns=RAW_HEADER).to_csv(raw_out_path, index=False)
            print(f"Resuming: {len(done_pair_ids)} pairs done; redoing pair {present[-1]}.")
            f = raw_out_path.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=RAW_HEADER)
        else:
            resume = False
    if not resume:
        f = raw_out_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=RAW_HEADER)
        writer.writeheader()

    def names_filter(n: str) -> bool:
        return n.endswith("hook_z") or n.endswith("hook_pattern")

    skipped = 0
    try:
        for pair_id, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Head patching"):
            pair_id = int(pair_id)
            if pair_id in done_pair_ids:
                continue
            aln = align_pair(tokenizer, str(row["sent_x"]), str(row["sent_y"]), str(row["prefix_y"]))
            if aln is None or not aln["ok"]:
                skipped += 1
                continue

            q_ids = torch.tensor([aln["q_ids"]], device=device)
            c_ids = torch.tensor([aln["c_ids"]], device=device)
            c_ids_flat = c_ids[0]
            len_q, len_c = q_ids.shape[1], c_ids.shape[1]
            cont_start_c = aln["cont_start_c"]
            cont_count = aln["cont_count"]
            P, Lx = aln["P"], aln["Lx"]
            source_pos = torch.tensor(aln["source_pos"], device=device)  # control->queer position map

            # Control positions to patch (per the chosen mode).
            if args.patch_positions == "identity":
                patch_pos = torch.tensor(aln["identity_control_positions"], device=device)
            elif args.patch_positions == "readout":
                patch_pos = torch.tensor([cont_start_c - 1], device=device)
            else:  # all aligned positions = the head's full contribution
                patch_pos = torch.arange(len_c, device=device)

            # Queer cache: per-head outputs (hook_z) and attention patterns.
            _, qcache = model.run_with_cache(q_ids, names_filter=names_filter)
            control_logits = model(c_ids)
            control_avg = float(continuation_logp(control_logits, c_ids_flat, cont_start_c)[0].item()) / cont_count
            queer_logits = model(q_ids)
            queer_avg = float(continuation_logp(queer_logits, q_ids[0], len_q - cont_count)[0].item()) / cont_count
            denom = queer_avg - control_avg

            # Read-out + identity positions in the QUEER prompt (for attention analysis).
            q_readout = len_q - cont_count - 1
            q_identity = list(range(P, P + Lx))

            pair_rows: list[dict[str, Any]] = []
            for layer in range(n_layers):
                z_name = tl_utils.get_act_name("z", layer)
                qz = qcache[z_name]  # [1, T_q, H, d_head]
                qz_aligned = qz[0, source_pos]  # [T_c, H, d_head] (queer head outputs, control-aligned)

                # Attention pattern: readout(query) -> identity(keys) in the queer prompt.
                patt = qcache[tl_utils.get_act_name("pattern", layer)]  # [1, H, q, k]
                attn_to_id = patt[0, :, q_readout, q_identity].sum(dim=-1)  # [H]

                # Head patching, batched over heads.
                bias_by_head = [float("nan")] * n_heads
                for hstart in range(0, n_heads, hbs):
                    heads = list(range(hstart, min(hstart + hbs, n_heads)))
                    batched = c_ids.repeat(len(heads), 1)

                    def hook(z, hook, heads=heads, qz_aligned=qz_aligned, patch_pos=patch_pos):
                        z = z.clone()
                        for i, h in enumerate(heads):
                            z[i, patch_pos, h, :] = qz_aligned[patch_pos, h, :].to(z.dtype)
                        return z

                    patched_logits = model.run_with_hooks(batched, fwd_hooks=[(z_name, hook)])
                    sums = continuation_logp(patched_logits, c_ids_flat, cont_start_c)  # [len(heads)]
                    for i, h in enumerate(heads):
                        bias_by_head[h] = float(sums[i].item()) / cont_count

                for h in range(n_heads):
                    be = bias_by_head[h] - control_avg
                    pair_rows.append({
                        "pair_id": pair_id, "row_id": row.get("row_id"),
                        "Gender_ID_x": row.get("Gender_ID_x"), "Gender_ID_y": row.get("Gender_ID_y"),
                        "predicate": row.get("predicate"),
                        "predicate_label_provisional": row.get("predicate_label_provisional"),
                        "layer": layer, "head": h,
                        "bias_effect": be,
                        "attn_readout_to_identity": float(attn_to_id[h].item()),
                        "n_identity_tokens": int(Lx),
                        "control_cont_avg_logp": control_avg, "queer_cont_avg_logp": queer_avg,
                        "normalized_restoration": safe_norm(be, denom),
                    })

            writer.writerows(pair_rows)
            f.flush()
            del qcache
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        f.close()

    print(f"\nSkipped (not cleanly alignable): {skipped} / {len(pairs)}")
    return pd.read_csv(raw_out_path)


def _hl(label: str) -> str:
    return label  # kept for future styling hooks


def plot_layer_head_heatmap(pivot: pd.DataFrame, png: Path, title: str, cbar_label: str,
                            diverging: bool, box_cells: list[tuple[int, int]] | None = None) -> None:
    if pivot.empty:
        return
    vals = pd.Series(pivot.values.ravel()).dropna()
    if diverging:
        vmax = max(float(vals.abs().quantile(0.99)), 1e-9)
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        kw = dict(cmap="magma", vmin=0.0, vmax=max(float(vals.quantile(0.99)), 1e-9))
    nL, nH = pivot.shape
    fig, ax = plt.subplots(figsize=(max(9, nH * 0.34), max(7.5, nL * 0.30)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", interpolation="nearest", **kw)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("head", fontsize=11); ax.set_ylabel("layer", fontsize=11)
    ax.set_xticks(range(nH)); ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=7)
    ax.set_yticks(range(nL)); ax.set_yticklabels([str(int(i)) for i in pivot.index], fontsize=7)
    # faint cell grid
    ax.set_xticks(np.arange(-0.5, nH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nL, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", length=0)
    # box+label the standout heads
    layers = list(pivot.index); heads = list(pivot.columns)
    for (L, H) in (box_cells or []):
        if L in layers and H in heads:
            xi, yi = heads.index(H), layers.index(L)
            ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="#00d000", linewidth=1.8))
            ax.text(xi, yi, f"L{L}H{H}", ha="center", va="center", fontsize=5.5, color="#003300", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_circuit_scatter(merged: pd.DataFrame, png: Path, top_k: int = 14) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sc = ax.scatter(merged["attn_readout_to_identity"], merged["bias_effect"],
                    c=merged["layer"], cmap="viridis", s=30, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.axvline(float(merged["attn_readout_to_identity"].median()), color="#888", lw=0.8, ls="--")
    for _, r in merged.head(top_k).iterrows():
        ax.annotate(f"L{int(r['layer'])}H{int(r['head'])}",
                    (r["attn_readout_to_identity"], r["bias_effect"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8.5, fontweight="bold", color="#111")
    ax.set_xlabel("read-out → identity attention   (head READS the identity)", fontsize=11)
    ax.set_ylabel("bias_effect = Δ logP(predicate)   (head WRITES the stereotype)", fontsize=11)
    ax.set_title("WinoQueer bias circuit — each point is one head", fontsize=13)
    ax.text(0.985, 0.97, "core circuit\n(reads + writes)\n→ top-right", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color="#555",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.8))
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("layer", fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_top_heads_bar(merged: pd.DataFrame, png: Path, top_k: int = 20) -> None:
    top = merged.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(5, top_k * 0.34)))
    denom = float(merged["layer"].max()) + 1e-9
    colors = plt.cm.viridis(top["layer"] / denom)
    ax.barh(range(len(top)), top["circuit_score"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"L{int(l)}H{int(h)}" for l, h in zip(top["layer"], top["head"])], fontsize=8.5)
    ax.set_xlabel("circuit_score  =  z(bias_effect) + z(read-out→identity attention)", fontsize=10)
    ax.set_title(f"Top {top_k} bias-circuit heads (read identity AND write stereotype)", fontsize=12)
    xmax = float(top["circuit_score"].max())
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["circuit_score"] + 0.01 * xmax, i,
                f"Δ={r['bias_effect']:+.2f}  attn={r['attn_readout_to_identity']:.2f}",
                va="center", fontsize=7.5, color="#333")
    ax.margins(x=0.18)
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_layer_summary(bias: pd.DataFrame, attn: pd.DataFrame, png: Path) -> None:
    bl = bias.groupby("layer")["bias_effect"].mean()
    al = attn.groupby("layer")["attn_readout_to_identity"].mean()
    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    l1, = ax1.plot(bl.index, bl.values, "-o", color="#c0392b", lw=2, ms=4, label="head-patching bias_effect")
    ax1.set_xlabel("layer", fontsize=11)
    ax1.set_ylabel("mean bias_effect (writes stereotype)", color="#c0392b", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.axhline(0, color="#ccc", lw=0.8)
    ax2 = ax1.twinx()
    l2, = ax2.plot(al.index, al.values, "-s", color="#2c7fb8", lw=2, ms=4, label="read-out→identity attention")
    ax2.set_ylabel("mean attention (reads identity)", color="#2c7fb8", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#2c7fb8")
    ax1.set_title("Per-layer summary: where heads write the bias vs read the identity", fontsize=12)
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)


def make_outputs(raw_df: pd.DataFrame, out_dir: Path):
    bias = raw_df.groupby(["layer", "head"], as_index=False)["bias_effect"].mean()
    attn = raw_df.groupby(["layer", "head"], as_index=False)["attn_readout_to_identity"].mean()
    bias_csv = out_dir / "winoqueer_head_patching_layer_head.csv"
    attn_csv = out_dir / "winoqueer_head_attention_layer_head.csv"
    bias.to_csv(bias_csv, index=False)
    attn.to_csv(attn_csv, index=False)

    # Combined ranking: heads that both WRITE the bias and READ the identity.
    merged = bias.merge(attn, on=["layer", "head"])
    bz = (merged["bias_effect"] - merged["bias_effect"].mean()) / (merged["bias_effect"].std() + 1e-9)
    az = (merged["attn_readout_to_identity"] - merged["attn_readout_to_identity"].mean()) / (merged["attn_readout_to_identity"].std() + 1e-9)
    merged["circuit_score"] = bz + az
    merged = merged.sort_values("circuit_score", ascending=False).reset_index(drop=True)
    rank_csv = out_dir / "winoqueer_head_circuit_ranking.csv"
    merged.to_csv(rank_csv, index=False)

    # heads to highlight on the heatmaps
    top_bias = [(int(r["layer"]), int(r["head"])) for _, r in bias.reindex(bias["bias_effect"].abs().sort_values(ascending=False).index).head(10).iterrows()]
    top_attn = [(int(r["layer"]), int(r["head"])) for _, r in attn.sort_values("attn_readout_to_identity", ascending=False).head(10).iterrows()]

    bias_p = bias.pivot(index="layer", columns="head", values="bias_effect").sort_index().sort_index(axis=1)
    attn_p = attn.pivot(index="layer", columns="head", values="attn_readout_to_identity").sort_index().sort_index(axis=1)
    bias_png = out_dir / "winoqueer_head_patching_heatmap.png"
    attn_png = out_dir / "winoqueer_head_attention_heatmap.png"
    scatter_png = out_dir / "winoqueer_head_circuit_scatter.png"
    bar_png = out_dir / "winoqueer_head_top_circuit_bar.png"
    layer_png = out_dir / "winoqueer_head_layer_summary.png"

    plot_layer_head_heatmap(bias_p, bias_png, "Head patching (queer→control): mean Δ logP(predicate)",
                            "mean bias_effect  (>0 = head writes stereotype)", diverging=True, box_cells=top_bias)
    plot_layer_head_heatmap(attn_p, attn_png, "Attention: read-out → queer-identity (mean over pairs)",
                            "mean attention weight  (head reads identity)", diverging=False, box_cells=top_attn)
    plot_circuit_scatter(merged, scatter_png)
    plot_top_heads_bar(merged, bar_png)
    plot_layer_summary(bias, attn, layer_png)

    return [bias_csv, attn_csv, rank_csv, bias_png, attn_png, scatter_png, bar_png, layer_png], bias, attn, merged


def main() -> None:
    parser = argparse.ArgumentParser(description="WinoQueer attention-head patching + attention-pattern analysis.")
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
    parser.add_argument("--patch_positions", choices=["all", "identity", "readout"], default="all",
                        help="Which control positions of a head's output to replace with the queer value.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_resort", action="store_true",
                        help="Consume --pairs_csv in file order (no bias_score re-sort / per-predicate cap). "
                             "Use with a pre-frozen cohort so pair_id == cohort row order.")
    args = parser.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "winoqueer_head_patching_raw.csv"

    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"--plot_only needs {raw_path}")
        raw_df = pd.read_csv(raw_path)
        print(f"plot_only: {raw_df['pair_id'].nunique()} pairs from {raw_path}")
    else:
        if args.pairs_csv is None:
            parser.error("--pairs_csv is required unless --plot_only")
        pairs = pd.read_csv(args.pairs_csv)
        if args.no_resort:
            pairs = pairs.reset_index(drop=True)
            print(f"Pairs: {len(pairs)} (no_resort: consuming cohort in file order)")
        else:
            pairs = pairs.sort_values("bias_score", ascending=False)
            if args.max_per_predicate is not None and "predicate" in pairs.columns:
                pairs = pairs.groupby("predicate", sort=False, group_keys=False).head(args.max_per_predicate)
                pairs = pairs.sort_values("bias_score", ascending=False)
            if args.max_pairs is not None:
                pairs = pairs.head(args.max_pairs)
            pairs = pairs.reset_index(drop=True)
            print(f"Pairs: {len(pairs)} (max_per_predicate={args.max_per_predicate}, max_pairs={args.max_pairs})")
        raw_df = run_head_patching(args, pairs, raw_path)

    out_paths, bias, attn, ranking = make_outputs(raw_df, args.out_dir)

    print("\nWrote:")
    for p in [raw_path] + out_paths:
        print(f"  {p}")
    print("\nTop 12 heads by patching bias_effect (which heads WRITE the bias):")
    print(bias.sort_values("bias_effect", ascending=False).head(12).to_string(index=False))
    print("\nTop 12 heads by read-out→identity attention (which heads READ the identity):")
    print(attn.sort_values("attn_readout_to_identity", ascending=False).head(12).to_string(index=False))
    print("\nTop 12 core circuit heads (high on BOTH):")
    print(ranking.head(12)[["layer", "head", "bias_effect", "attn_readout_to_identity", "circuit_score"]].to_string(index=False))
    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
