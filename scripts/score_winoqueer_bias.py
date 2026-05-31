#!/usr/bin/env python3
"""Score WinoQueer examples for stereotype bias via autoregressive continuation likelihood.

Each WinoQueer row gives a stereotype continuation and two prefixes that differ only by the
named identity. We measure whether the model finds the (already-present) stereotype
continuation MORE probable after the LGBTQ identity than after the straight/cisgender control:

    queer_variant   = (sent_x, Gender_ID_x, prefix_x)   # the LGBTQ identity
    control_variant = (sent_y, Gender_ID_y, prefix_y)   # the straight/cisgender control

    queer_continuation_logp   = sum_t log P(continuation_t | prefix_x, continuation_<t)
    control_continuation_logp = sum_t log P(continuation_t | prefix_y, continuation_<t)
    bias_score = queer_continuation_avg_logp - control_continuation_avg_logp

bias_score > 0  -> the stereotype is more probable for the LGBTQ identity (stereotype-consistent
                   bias). bias_score < 0 -> more probable for the control identity.

We do NOT construct antonyms or alternative continuations; we only score the continuation that
already exists in the parsed dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def load_model(args):
    from transformer_lens import HookedTransformer

    device = resolve_device(args.device)
    if device == "mps" and args.dtype == "bfloat16":
        print("MPS + bfloat16 can be unreliable; switching to float16.")
        args.dtype = "float16"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"\nDevice: {device} | dtype: {args.dtype}")
    print(f"TransformerLens model name: {args.tl_model_name}")
    print(f"HF model source: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.padding_side = "right"  # right padding is safe for causal last-token scoring
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    print("Wrapping HF model with TransformerLens...")
    model = HookedTransformer.from_pretrained(
        args.tl_model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        default_prepend_bos=True,
    )
    model.eval()
    return model, tokenizer, device


def common_prefix_len(a: list[int], b: list[int]) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


@torch.no_grad()
def score_continuations(
    model, tokenizer, fulls: list[str], prefixes: list[str], device: str, batch_size: int
) -> list[dict[str, Any]]:
    """For each (full, prefix) where full == prefix + continuation, return a dict with:
      logp  : sum log P(continuation tokens)
      count : continuation token count
      first_token_rank : rank (1 = top) of the FIRST continuation token in the next-token
              distribution given the prefix — captures cases where the summed logprob shifts
              little but the stereotype token jumps up/down the ranking
      first_token_id : id of that first continuation token (for decoding)
    The continuation span is the longest token-prefix shared between the full sentence and
    `prefix` (robust to BPE boundary merges)."""
    results: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(fulls), batch_size), desc="Scoring", leave=False):
        bf = fulls[start : start + batch_size]
        bp = prefixes[start : start + batch_size]
        enc = tokenizer(bf, return_tensors="pt", padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits = model(ids)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        # log P(token at position t+1 | tokens <= t)
        tgt = ids[:, 1:]
        tok_logp = log_probs[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        real_len = mask.sum(dim=1)

        for i in range(len(bf)):
            rl = int(real_len[i].item())
            full_ids_i = ids[i, :rl].tolist()
            pref_ids_i = tokenizer(bp[i], add_special_tokens=True)["input_ids"]
            cstart = common_prefix_len(full_ids_i, pref_ids_i)  # continuation begins here in `ids`
            cstart = max(cstart, 1)
            count = rl - cstart
            if count <= 0:
                results.append({"logp": float("nan"), "count": 0, "first_token_rank": -1, "first_token_id": -1})
                continue
            seg = tok_logp[i, cstart - 1 : rl - 1]
            # Rank of the first continuation token in P(next | prefix): 1 = most probable.
            first_tok = full_ids_i[cstart]
            dist = log_probs[i, cstart - 1, :]
            first_rank = int((dist > dist[first_tok]).sum().item()) + 1
            results.append({
                "logp": float(seg.sum().item()), "count": int(count),
                "first_token_rank": first_rank, "first_token_id": int(first_tok),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score WinoQueer stereotype continuation bias.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=None, help="Score only the first N scoreable rows (debug).")
    parser.add_argument("--exclude_review", action="store_true", help="Drop needs_manual_review rows before scoring.")
    parser.add_argument("--rank_top_k", type=int, default=1000, help="Rows kept in the most-positive/negative files.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    for col in ["prefix_x", "prefix_y", "continuation"]:
        df[col] = df[col].fillna("")
    if args.exclude_review and "needs_manual_review" in df.columns:
        df = df[~df["needs_manual_review"].astype(bool)].copy()

    # Scoreable = both prefixes and the continuation are non-empty.
    scoreable = (
        (df["prefix_x"].str.len() > 0)
        & (df["prefix_y"].str.len() > 0)
        & (df["continuation"].str.len() > 0)
    )
    work = df[scoreable].copy()
    if args.max_rows is not None:
        work = work.head(args.max_rows).copy()
    print(f"Rows in input: {len(df)} | scoreable: {int(scoreable.sum())} | scoring: {len(work)}")

    model, tokenizer, device = load_model(args)

    queer = score_continuations(
        model, tokenizer, work["sent_x"].astype(str).tolist(), work["prefix_x"].astype(str).tolist(),
        device, args.batch_size,
    )
    control = score_continuations(
        model, tokenizer, work["sent_y"].astype(str).tolist(), work["prefix_y"].astype(str).tolist(),
        device, args.batch_size,
    )

    work["queer_continuation_logp"] = [q["logp"] for q in queer]
    work["control_continuation_logp"] = [c["logp"] for c in control]
    work["continuation_token_count"] = [q["count"] for q in queer]
    qcount = pd.Series([q["count"] for q in queer], index=work.index).replace(0, pd.NA)
    ccount = pd.Series([c["count"] for c in control], index=work.index).replace(0, pd.NA)
    work["queer_continuation_avg_logp"] = work["queer_continuation_logp"] / qcount
    work["control_continuation_avg_logp"] = work["control_continuation_logp"] / ccount
    work["bias_score"] = work["queer_continuation_avg_logp"] - work["control_continuation_avg_logp"]

    # First continuation-token rank under each prefix (1 = top-ranked next token).
    work["queer_first_token_rank"] = [q["first_token_rank"] for q in queer]
    work["control_first_token_rank"] = [c["first_token_rank"] for c in control]
    work["first_continuation_token"] = [
        tokenizer.decode([q["first_token_id"]]) if q["first_token_id"] >= 0 else "" for q in queer
    ]
    # Positive => stereotype token ranks HIGHER (smaller rank number) for the LGBTQ identity.
    work["first_token_rank_gap"] = work["control_first_token_rank"] - work["queer_first_token_rank"]

    out_cols = [
        "row_id", "Gender_ID_x", "Gender_ID_y", "sent_x", "sent_y", "prefix_x", "prefix_y",
        "continuation", "predicate", "continuation_token_count",
        "queer_continuation_logp", "control_continuation_logp",
        "queer_continuation_avg_logp", "control_continuation_avg_logp", "bias_score",
        "first_continuation_token", "queer_first_token_rank", "control_first_token_rank",
        "first_token_rank_gap",
        "parse_status", "needs_manual_review",
    ]
    out_cols = [c for c in out_cols if c in work.columns]
    scored = work[out_cols].copy()

    scored_path = args.out_dir / "winoqueer_scored.csv"
    scored.to_csv(scored_path, index=False)

    valid = scored.dropna(subset=["bias_score"]).copy()
    ranked = valid.sort_values("bias_score", ascending=False)
    pos_path = args.out_dir / "winoqueer_most_positive_bias.csv"
    neg_path = args.out_dir / "winoqueer_most_negative_bias.csv"
    ranked.head(args.rank_top_k).to_csv(pos_path, index=False)
    ranked.tail(args.rank_top_k).iloc[::-1].to_csv(neg_path, index=False)

    # ---- summary ----
    bs = valid["bias_score"]
    summary_rows: list[dict[str, Any]] = [
        {"group": "ALL", "key": "", "n": int(len(bs)),
         "mean_bias_score": float(bs.mean()), "median_bias_score": float(bs.median()),
         "std_bias_score": float(bs.std()), "min_bias_score": float(bs.min()),
         "max_bias_score": float(bs.max()),
         "positive_bias_fraction": float((bs > 0).mean())},
    ]
    for group_col in ["Gender_ID_x", "Gender_ID_y"]:
        for key, sub in valid.groupby(group_col):
            sbs = sub["bias_score"]
            summary_rows.append({
                "group": group_col, "key": key, "n": int(len(sbs)),
                "mean_bias_score": float(sbs.mean()), "median_bias_score": float(sbs.median()),
                "std_bias_score": float(sbs.std()) if len(sbs) > 1 else 0.0,
                "min_bias_score": float(sbs.min()), "max_bias_score": float(sbs.max()),
                "positive_bias_fraction": float((sbs > 0).mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "winoqueer_bias_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ---- plots ----
    hist_png = args.out_dir / "bias_score_histogram.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(bs, bins=80, color="#4C72B0", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(float(bs.mean()), color="crimson", linestyle="-", linewidth=1.5, label=f"mean={bs.mean():.3f}")
    ax.set_title("WinoQueer bias_score (queer_avg_logp − control_avg_logp)\n>0 = stereotype more probable for LGBTQ identity")
    ax.set_xlabel("bias_score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_png, dpi=150)
    plt.close(fig)

    by_png = args.out_dir / "bias_score_by_identity.png"
    by_id = valid.groupby("Gender_ID_x")["bias_score"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(by_id) * 0.5), 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in by_id.values]
    ax.bar(range(len(by_id)), by_id.values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(range(len(by_id)))
    ax.set_xticklabels(by_id.index, rotation=60, ha="right")
    ax.set_ylabel("mean bias_score")
    ax.set_title("Mean WinoQueer bias_score by queer identity (Gender_ID_x)")
    fig.tight_layout()
    fig.savefig(by_png, dpi=150)
    plt.close(fig)

    # ---- console validation ----
    def show(block: pd.DataFrame, title: str) -> None:
        print(f"\n=== {title} ===")
        for _, r in block.iterrows():
            print(f"  bias={r['bias_score']:+.4f} | first-tok rank queer={int(r['queer_first_token_rank'])} "
                  f"control={int(r['control_first_token_rank'])} (gap={int(r['first_token_rank_gap'])}) "
                  f"| {r['Gender_ID_x']} vs {r['Gender_ID_y']}")
            print(f"     queer  : {r['prefix_x']!r}  + {r['continuation']!r}")
            print(f"     control: {r['prefix_y']!r}  + {r['continuation']!r}")

    near_zero = valid.reindex(valid["bias_score"].abs().sort_values().index)
    print("\nWrote:")
    for p in [scored_path, pos_path, neg_path, summary_path, hist_png, by_png]:
        print(f"  {p}")
    print("\n=== summary (ALL) ===")
    print(summary.iloc[0].to_string())
    show(ranked.head(20), "Top 20 highest bias_score (stereotype favors LGBTQ identity)")
    show(near_zero.head(20), "20 near-zero bias_score (identity has little effect)")
    show(ranked.tail(20).iloc[::-1], "20 most negative bias_score (stereotype favors control identity)")


if __name__ == "__main__":
    main()
