#!/usr/bin/env python3
"""OOD steering transfer: apply a WinoQueer-learned bias DIRECTION to the BBQ QA task.

The steering sweep (run_winoqueer_steering_sweep.py --save_vectors) learns a per-layer
difference-of-means vector v_L in resid_pre[L] space that linearly controls LGBTQ stereotype in
WinoQueer's CONTINUATION task. This script asks the OOD-transfer question: does that SAME direction
move bias on the structurally different BBQ multiple-choice QA task? If injecting +alpha*v_L raises
the model's preference for the stereotyped answer (and -alpha lowers it), the bias direction is
shared across task formats — a causal, not merely correlational, cross-task result.

Design (mirrors the WinoQueer sweep so the two are comparable):
  - reuse the validated BBQ MCQ scorer (identify_biased_bbq_items.py) for prompt building and the
    bias metric (p_biased, biased_prediction_rate, biased_vs_unknown_logp_diff) — UNMODIFIED;
  - inject alpha * v_L at resid_pre[L] during the forward pass, at the decision token (default) or
    all real positions, sweeping (layer, alpha, sign);
  - alpha=0 is the unsteered baseline; a norm-matched RANDOM direction is the control (must NOT move
    bias -> proves the effect is the direction, not the injected norm).

CONSTRUCT MATCHING: pair the vector with the matching BBQ file for the clean causal claim
(gender_identity vector -> BBQ trans items; sexual_orientation vector -> BBQ SO items). Pairing a
vector with a DIFFERENT axis's BBQ file is the cross-construct generalization test.

Output: long CSV (kind, layer, alpha, polarity x metrics) + bias-vs-alpha plot (real vs random) per
layer, and a summary of the best transfer operating points.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils as tl_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the validated BBQ scorer pieces (prompt construction + bias metric) without touching them.
from identify_biased_bbq_items import (  # noqa: E402
    LETTERS, build_prompt, load_jsonl, resolve_device, score_example_from_logps,
)
from run_winoqueer_steering_sweep import random_matched  # noqa: E402


def filter_subcategory(rows, sub):
    """BBQ Gender_identity splits into trans vs cis-binary via `stereotyped_groups`, NOT the
    `subcategory` field (which is adult/child). trans = any group token mentions trans/nonbinary
    (864 items); binary = the F/M-only items (4808). 'all' = no filter. On non-gender files there
    are no trans groups, so 'trans' yields 0 rows (guarded by the caller) and 'binary' is a no-op.
    """
    if sub == "all":
        return rows

    def is_trans(r):
        groups = r.get("additional_metadata", {}).get("stereotyped_groups", [])
        return any("trans" in str(g).lower() or "nonbinary" in str(g).lower() for g in groups)

    return [r for r in rows if (is_trans(r) if sub == "trans" else not is_trans(r))]


def load_model(model_path: str, tl_model_name: str, device: str, dtype: torch.dtype):
    """Load local HF weights wrapped in TransformerLens (same config as the BBQ scorer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained(
        tl_model_name, hf_model=hf_model, tokenizer=tokenizer, device=device, dtype=dtype,
        fold_ln=False, center_writing_weights=False, center_unembed=False, default_prepend_bos=True,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_batch_steered(model, tokenizer, prompts, device, layer, vec, alpha, positions):
    """Next-token logprobs for ' A'/' B'/' C', with alpha*vec injected at resid_pre[layer].

    positions: 'last' = inject only at each row's decision token (the position whose next token is
    the answer letter); 'all' = inject at every real (non-pad) token. alpha=0 / vec=None = baseline.
    """
    choice_ids = {}
    for letter in LETTERS:
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Choice {letter} not single-token: {ids}")
        choice_ids[letter] = ids[0]

    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    tokens = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    last_pos = attn.sum(dim=1) - 1                       # decision token per row (right-padded)
    batch_idx = torch.arange(tokens.shape[0], device=device)

    hooks = []
    if vec is not None and alpha != 0.0:
        v = (alpha * vec.to(device)).to(model.cfg.dtype)

        def hook(act, hook, v=v, last_pos=last_pos, attn=attn, batch_idx=batch_idx):
            act = act.clone()
            if positions == "all":
                act = act + attn.unsqueeze(-1).to(act.dtype) * v   # only real tokens
            else:
                act[batch_idx, last_pos, :] = act[batch_idx, last_pos, :] + v
            return act

        hooks = [(tl_utils.get_act_name("resid_pre", layer), hook)]

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    next_logits = logits[batch_idx, last_pos, :]
    logps = torch.log_softmax(next_logits.float(), dim=-1)
    return [{letter: float(logps[i, tid].item()) for letter, tid in choice_ids.items()}
            for i in range(len(prompts))]


def score_all(model, tokenizer, rows, device, layer, vec, alpha, positions, batch_size):
    """Score every BBQ row under one steering setting; return the per-example metric frame."""
    scored = []
    for s in range(0, len(rows), batch_size):
        batch = rows[s:s + batch_size]
        prompts = [build_prompt(ex) for ex in batch]
        logps = score_batch_steered(model, tokenizer, prompts, device, layer, vec, alpha, positions)
        for ex, lp in zip(batch, logps):
            scored.append(score_example_from_logps(ex, lp))
    return pd.DataFrame(scored)


def summarize(df: pd.DataFrame) -> dict:
    """Collapse a scored frame to the bias metrics we track Delta on (overall + per polarity)."""
    out = {
        "n": len(df),
        "accuracy": float(df["is_correct"].mean()),
        "biased_prediction_rate": float(df["is_biased_prediction"].mean()),
        "unknown_prediction_rate": float(df["is_unknown_prediction"].mean()),
        "mean_p_biased": float(df["p_biased"].mean()),
        "mean_biased_vs_unknown_logp_diff": float(df["biased_vs_unknown_logp_diff"].mean(skipna=True)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a WinoQueer steering vector OOD to the BBQ QA task.")
    ap.add_argument("--bbq_path", type=Path, required=True, help="BBQ jsonl (match the vector's axis)")
    ap.add_argument("--vectors", type=Path, required=True, help=".pt from steering sweep --save_vectors")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--context_condition", choices=["ambig", "disambig", "all"], default="ambig")
    ap.add_argument("--subcategory", choices=["all", "trans", "binary"], default="all",
                    help="Gender_identity subset via stereotyped_groups: trans (matched to the WQ "
                         "gender_identity vector) vs binary (cis F/M roles). 'all' = no filter.")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--layers", type=str, default=None, help="comma list; default = all saved layers")
    ap.add_argument("--alphas", type=str, default="-3,-2,-1,-0.5,0,0.5,1,2,3")
    ap.add_argument("--positions", choices=["last", "all"], default="last",
                    help="inject at the decision token only (last) or every real token (all)")
    ap.add_argument("--no_random", action="store_true", help="skip the norm-matched random control")
    ap.add_argument("--n_random_seeds", type=int, default=5)
    args = ap.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    blob = torch.load(args.vectors, map_location="cpu")
    vectors, norms = blob["vectors"], blob["norms"]
    n_layers = blob.get("n_layers", vectors.shape[0])
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))
    alphas = [float(x) for x in args.alphas.split(",")]
    rand_list = ([] if args.no_random
                 else [random_matched(vectors, norms, seed_offset=s) for s in range(args.n_random_seeds)])
    print(f"Vectors: {args.vectors} (pos={blob.get('vector_position')}, "
          f"{blob.get('n_train_pairs')} train pairs) | layers={layers} | alphas={alphas} | "
          f"positions={args.positions}")

    rows = load_jsonl(args.bbq_path)
    if args.context_condition != "all":
        rows = [r for r in rows if r.get("context_condition") == args.context_condition]
    rows = filter_subcategory(rows, args.subcategory)
    if not rows:
        raise SystemExit(f"No BBQ rows after filtering (context={args.context_condition}, "
                         f"subcategory={args.subcategory}) — wrong --subcategory for this file?")
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    print(f"BBQ examples: {len(rows)} from {args.bbq_path} "
          f"({args.context_condition}, subcategory={args.subcategory})")

    model, tokenizer = load_model(args.model_path, args.tl_model_name, device, dtype)

    records = []
    # real direction across the (layer, alpha) grid; alpha=0 scored once per layer is the baseline.
    for L in tqdm(layers, desc="layers"):
        for a in alphas:
            s = summarize(score_all(model, tokenizer, rows, device, L, vectors[L], a, args.positions, args.batch_size))
            records.append({"kind": "real", "rand_seed": -1, "layer": L, "alpha": a, **s})
            for si, rv in enumerate(rand_list):
                if a == 0.0:           # alpha=0 is identical to baseline regardless of direction
                    continue
                sr = summarize(score_all(model, tokenizer, rows, device, L, rv[L], a, args.positions, args.batch_size))
                records.append({"kind": "random", "rand_seed": si, "layer": L, "alpha": a, **sr})
        if device == "cuda":
            torch.cuda.empty_cache()

    raw = pd.DataFrame(records)
    raw_path = args.out_dir / "bbq_steering_transfer_raw.csv"
    raw.to_csv(raw_path, index=False)

    # bias-vs-alpha curves (real vs random, median over seeds), one line per layer
    metric = "mean_p_biased"
    base = float(raw[(raw.kind == "real") & (raw.alpha == 0.0)][metric].mean())
    fig, ax = plt.subplots(figsize=(9, 6))
    for L in layers:
        r = raw[(raw.kind == "real") & (raw.layer == L)].groupby("alpha")[metric].mean()
        ax.plot(r.index, r.values, "-o", ms=3, lw=1.4, label=f"L{L}")
    if rand_list:
        rnd = raw[raw.kind == "random"].groupby("alpha")[metric].median()
        ax.plot(rnd.index, rnd.values, "--", color="#888", lw=2, label="random (median)")
    ax.axhline(base, color="#2c7fb8", ls=":", lw=1, label="unsteered baseline")
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("alpha (coefficient on WinoQueer v_L)"); ax.set_ylabel(metric)
    ax.set_title(f"OOD steering transfer to BBQ QA — {args.bbq_path.stem} [{args.subcategory}]\n"
                 f"(does a WinoQueer bias direction move BBQ answer bias?)")
    ax.legend(fontsize=7, ncol=2)
    curve_path = args.out_dir / "bbq_steering_transfer_curve.png"
    fig.tight_layout(); fig.savefig(curve_path, dpi=160, bbox_inches="tight"); plt.close(fig)

    # best transfer: largest bias increase at alpha>0 and decrease at alpha<0, real direction
    real = raw[raw.kind == "real"]
    up = real[real.alpha > 0].sort_values(metric, ascending=False).head(1)
    dn = real[real.alpha < 0].sort_values(metric, ascending=True).head(1)
    print(f"\nUnsteered baseline {metric} = {base:.4f}")
    for tag, row in [("max INDUCE (alpha>0)", up), ("max DE-BIAS (alpha<0)", dn)]:
        if not row.empty:
            r = row.iloc[0]
            print(f"  {tag}: L{int(r.layer)} alpha={r.alpha:g} -> {metric}={r[metric]:.4f} "
                  f"(delta {r[metric]-base:+.4f}), biased_pred_rate={r.biased_prediction_rate:.3f}")
    print(f"\nWrote:\n  {raw_path}\n  {curve_path}")
    print(f"runtime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
