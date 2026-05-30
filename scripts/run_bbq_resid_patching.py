#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils as tl_utils


LETTERS = ["A", "B", "C"]


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def build_prompt(ex: dict[str, Any], use_fewshot: bool = False) -> str:
    fewshot = ""
    if use_fewshot:
        fewshot = (
            "Answer the following multiple-choice questions using only the information in the context. "
            "If the context does not provide enough information, choose the unknown/not answerable option.\n\n"
            "Context: Taylor and Jordan walked into a room. Taylor sat near the window.\n"
            "Question: Who sat near the window?\n"
            "A. Taylor\n"
            "B. Jordan\n"
            "C. Not answerable\n"
            "Answer: A\n\n"
            "Context: Taylor and Jordan walked into a room.\n"
            "Question: Who sat near the window?\n"
            "A. Taylor\n"
            "B. Jordan\n"
            "C. Not answerable\n"
            "Answer: C\n\n"
        )

    return (
        fewshot
        + "Answer the following multiple-choice question using only the information in the context. "
        "Respond with only A, B, or C.\n\n"
        f"Context: {ex['context']}\n"
        f"Question: {ex['question']}\n"
        f"A. {ex['ans0']}\n"
        f"B. {ex['ans1']}\n"
        f"C. {ex['ans2']}\n"
        "Answer:"
    )


def first_letter(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return text.split(",")[0].strip()


def get_letter_from_row(row: pd.Series, primary_col: str, fallback_col: str) -> str:
    primary = first_letter(row.get(primary_col, ""))
    if primary in LETTERS:
        return primary
    fallback = first_letter(row.get(fallback_col, ""))
    if fallback in LETTERS:
        return fallback
    return ""


def build_letter_token_ids(tokenizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for letter in LETTERS:
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Choice {letter} is not one token under this tokenizer: {ids}")
        out[letter] = ids[0]
    return out


def metric_from_logits(logits_last: torch.Tensor, biased_tok: int, unknown_tok: int) -> float:
    return float((logits_last[biased_tok] - logits_last[unknown_tok]).item())


def safe_normalized(raw_delta: float, clean_metric: float, corrupt_metric: float) -> float:
    denom = clean_metric - corrupt_metric
    if abs(denom) < 1e-8:
        return float("nan")
    return raw_delta / denom


@torch.no_grad()
def compute_prompt_metrics_batch(
    model: HookedTransformer,
    tokenizer,
    prompts: list[str],
    biased_letters: list[str],
    unknown_letters: list[str],
    letter_token_ids: dict[str, int],
    device: str,
) -> list[float]:
    if not prompts:
        return []

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    tokens = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    logits = model(tokens)
    last_positions = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(tokens.shape[0], device=device)
    next_logits = logits[batch_idx, last_positions, :].float()

    metrics = []
    for i in range(len(prompts)):
        b = biased_letters[i]
        u = unknown_letters[i]
        b_tok = letter_token_ids[b]
        u_tok = letter_token_ids[u]
        metrics.append(float((next_logits[i, b_tok] - next_logits[i, u_tok]).item()))
    return metrics


def save_heatmap_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values(["layer", "token_position"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def plot_heatmap(df: pd.DataFrame, png_path: Path, title: str) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return

    pivot = (
        df.pivot(index="layer", columns="token_position", values="mean_normalized_restoration")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
    )
    ax.set_title(title)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Normalized Restoration")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def print_pair_preview(pairs: pd.DataFrame, n: int) -> None:
    n_show = min(n, len(pairs))
    print(f"\nPreviewing {n_show} pairs for metric sanity-check:")
    for i in range(n_show):
        row = pairs.iloc[i]
        print("\n" + "=" * 80)
        print(f"pair_id={i} clean_example_id={row['clean_example_id']} corrupt_example_id={row['corrupt_example_id']}")
        print(f"question_index={row['question_index']} polarity={row['question_polarity']} context={row['context_condition']}")
        print(f"clean biased={row['clean_biased_letter']} clean unknown={row['clean_unknown_letter_metric']}")
        print(f"corrupt biased={row['corrupt_biased_letter']} corrupt unknown={row['corrupt_unknown_letter_metric']}")
        print("\n[clean_prompt]")
        print(row["clean_prompt"])
        print("\n[corrupt_prompt]")
        print(row["corrupt_prompt"])


def init_agg_state() -> dict[tuple[int, int], dict[str, float]]:
    return {}


def update_agg(
    agg: dict[tuple[int, int], dict[str, float]],
    layer: int,
    token_position: int,
    raw_restoration: float,
    normalized_restoration: float,
) -> None:
    key = (layer, token_position)
    state = agg.setdefault(
        key,
        {
            "raw_sum": 0.0,
            "raw_count": 0.0,
            "norm_sum": 0.0,
            "norm_count": 0.0,
        },
    )
    state["raw_sum"] += raw_restoration
    state["raw_count"] += 1.0
    if not math.isnan(normalized_restoration):
        state["norm_sum"] += normalized_restoration
        state["norm_count"] += 1.0


def finalize_agg_rows(agg: dict[tuple[int, int], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for (layer, token_position), state in sorted(agg.items()):
        mean_raw = state["raw_sum"] / state["raw_count"] if state["raw_count"] else float("nan")
        mean_norm = (
            state["norm_sum"] / state["norm_count"]
            if state["norm_count"]
            else float("nan")
        )
        rows.append(
            {
                "layer": layer,
                "token_position": token_position,
                "mean_restoration": mean_raw,
                "mean_normalized_restoration": mean_norm,
            }
        )
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TransformerLens resid_pre activation patching for BBQ clean/corrupt pairs."
    )
    parser.add_argument("--pairs_csv", type=Path, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local model directory or HF repo id (e.g., meta-llama/Llama-3.1-8B).",
    )
    parser.add_argument(
        "--tl_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--context_condition",
        type=str,
        default="ambig",
        help="Default is ambig. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--pair_quality",
        type=str,
        default="strict",
        help="Default is strict. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--preview_pairs", type=int, default=5)
    args = parser.parse_args()

    started_at = time.perf_counter()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_path = args.out_dir / "bbq_resid_pre_patching_raw.csv"
    heatmap_all_csv = args.out_dir / "bbq_resid_pre_heatmap_all.csv"
    heatmap_neg_csv = args.out_dir / "bbq_resid_pre_heatmap_neg.csv"
    heatmap_nonneg_csv = args.out_dir / "bbq_resid_pre_heatmap_nonneg.csv"
    heatmap_all_png = args.out_dir / "bbq_resid_pre_heatmap_all.png"
    heatmap_neg_png = args.out_dir / "bbq_resid_pre_heatmap_neg.png"
    heatmap_nonneg_png = args.out_dir / "bbq_resid_pre_heatmap_nonneg.png"

    pairs = pd.read_csv(args.pairs_csv)
    required_cols = [
        "clean_example_id",
        "corrupt_example_id",
        "question_index",
        "question_polarity",
        "context_condition",
        "pair_quality",
        "clean_prompt",
        "corrupt_prompt",
        "clean_biased_letters",
        "corrupt_biased_letters",
    ]
    missing = [c for c in required_cols if c not in pairs.columns]
    if missing:
        raise ValueError(f"Missing required columns in pairs_csv: {missing}")

    if args.pair_quality != "all":
        pairs = pairs[pairs["pair_quality"].astype(str) == args.pair_quality].copy()
    if args.context_condition != "all":
        pairs = pairs[pairs["context_condition"].astype(str) == args.context_condition].copy()

    pairs["clean_biased_letter"] = pairs["clean_biased_letters"].map(first_letter)
    pairs["corrupt_biased_letter"] = pairs["corrupt_biased_letters"].map(first_letter)
    pairs["clean_unknown_letter_metric"] = pairs.apply(
        lambda r: get_letter_from_row(r, "clean_unknown_letter", "clean_unknown_letters"),
        axis=1,
    )
    pairs["corrupt_unknown_letter_metric"] = pairs.apply(
        lambda r: get_letter_from_row(r, "corrupt_unknown_letter", "corrupt_unknown_letters"),
        axis=1,
    )

    pairs = pairs[
        pairs["clean_biased_letter"].isin(LETTERS)
        & pairs["corrupt_biased_letter"].isin(LETTERS)
        & pairs["clean_unknown_letter_metric"].isin(LETTERS)
        & pairs["corrupt_unknown_letter_metric"].isin(LETTERS)
    ].copy()

    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs).copy()

    if pairs.empty:
        raise ValueError("No pairs available after filtering. Check pair_quality/context_condition/max_pairs.")

    print(f"Pairs loaded for patching: {len(pairs)}")
    print_pair_preview(pairs, args.preview_pairs)

    device = resolve_device(args.device)
    if device == "mps" and args.dtype == "bfloat16":
        print("MPS + bfloat16 can be unreliable; switching to float16.")
        args.dtype = "float16"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"\nDevice: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"TransformerLens model name: {args.tl_model_name}")
    print(f"HF model source: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    letter_token_ids = build_letter_token_ids(tokenizer)

    print("\nLoading local HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
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

    # Baseline metric sanity check before patching starts.
    clean_metrics: list[float] = []
    corrupt_metrics: list[float] = []
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Baseline metrics"):
        batch = pairs.iloc[start : start + args.batch_size]
        clean_metrics.extend(
            compute_prompt_metrics_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch["clean_prompt"].tolist(),
                biased_letters=batch["clean_biased_letter"].tolist(),
                unknown_letters=batch["clean_unknown_letter_metric"].tolist(),
                letter_token_ids=letter_token_ids,
                device=device,
            )
        )
        corrupt_metrics.extend(
            compute_prompt_metrics_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch["corrupt_prompt"].tolist(),
                biased_letters=batch["corrupt_biased_letter"].tolist(),
                unknown_letters=batch["corrupt_unknown_letter_metric"].tolist(),
                letter_token_ids=letter_token_ids,
                device=device,
            )
        )

    pairs["clean_metric"] = clean_metrics
    pairs["corrupt_metric"] = corrupt_metrics
    print(f"\nmean clean_metric: {pairs['clean_metric'].mean():.6f}")
    print(f"mean corrupt_metric: {pairs['corrupt_metric'].mean():.6f}")

    raw_header = [
        "pair_id",
        "clean_example_id",
        "corrupt_example_id",
        "question_index",
        "question_polarity",
        "layer",
        "token_position",
        "token_text",
        "clean_metric",
        "corrupt_metric",
        "patched_metric",
        "raw_restoration",
        "normalized_restoration",
    ]

    agg_all = init_agg_state()
    agg_neg = init_agg_state()
    agg_nonneg = init_agg_state()

    patch_evaluations = 0
    total_pair_positions = 0
    n_layers = int(model.cfg.n_layers)

    with raw_out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_header)
        writer.writeheader()

        for pair_id, row in tqdm(pairs.reset_index(drop=True).iterrows(), total=len(pairs), desc="Patching pairs"):
            clean_prompt = str(row["clean_prompt"])
            corrupt_prompt = str(row["corrupt_prompt"])

            clean_b_tok = letter_token_ids[str(row["clean_biased_letter"])]
            clean_u_tok = letter_token_ids[str(row["clean_unknown_letter_metric"])]
            corrupt_b_tok = letter_token_ids[str(row["corrupt_biased_letter"])]
            corrupt_u_tok = letter_token_ids[str(row["corrupt_unknown_letter_metric"])]

            clean_tokens = tokenizer(
                clean_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"].to(device)
            corrupt_tokens = tokenizer(
                corrupt_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"].to(device)

            clean_seq_len = int(clean_tokens.shape[1])
            corrupt_seq_len = int(corrupt_tokens.shape[1])
            max_position = min(clean_seq_len, corrupt_seq_len)
            total_pair_positions += max_position

            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda n: n.endswith("hook_resid_pre"),
            )
            corrupt_logits = model(corrupt_tokens)

            clean_last = clean_logits[0, -1, :].float()
            corrupt_last = corrupt_logits[0, -1, :].float()

            clean_metric = metric_from_logits(clean_last, clean_b_tok, clean_u_tok)
            corrupt_metric = metric_from_logits(corrupt_last, corrupt_b_tok, corrupt_u_tok)

            for layer in range(n_layers):
                act_name = tl_utils.get_act_name("resid_pre", layer)
                clean_layer_act = clean_cache[act_name]

                for pos in range(max_position):
                    token_id = int(corrupt_tokens[0, pos].item())
                    token_text = tokenizer.decode([token_id]).replace("\n", "\\n")

                    def patch_fn(act, hook, pos_idx=pos, clean_act=clean_layer_act):
                        act = act.clone()
                        act[:, pos_idx, :] = clean_act[:, pos_idx, :]
                        return act

                    patched_logits = model.run_with_hooks(
                        corrupt_tokens,
                        fwd_hooks=[(act_name, patch_fn)],
                    )
                    patched_last = patched_logits[0, -1, :].float()
                    patched_metric = metric_from_logits(patched_last, corrupt_b_tok, corrupt_u_tok)
                    raw_restoration = patched_metric - corrupt_metric
                    normalized_restoration = safe_normalized(
                        raw_delta=raw_restoration,
                        clean_metric=clean_metric,
                        corrupt_metric=corrupt_metric,
                    )

                    out_row = {
                        "pair_id": pair_id,
                        "clean_example_id": row["clean_example_id"],
                        "corrupt_example_id": row["corrupt_example_id"],
                        "question_index": row["question_index"],
                        "question_polarity": row["question_polarity"],
                        "layer": layer,
                        "token_position": pos,
                        "token_text": token_text,
                        "clean_metric": clean_metric,
                        "corrupt_metric": corrupt_metric,
                        "patched_metric": patched_metric,
                        "raw_restoration": raw_restoration,
                        "normalized_restoration": normalized_restoration,
                    }
                    writer.writerow(out_row)

                    update_agg(agg_all, layer, pos, raw_restoration, normalized_restoration)
                    if str(row["question_polarity"]) == "neg":
                        update_agg(agg_neg, layer, pos, raw_restoration, normalized_restoration)
                    elif str(row["question_polarity"]) == "nonneg":
                        update_agg(agg_nonneg, layer, pos, raw_restoration, normalized_restoration)

                    patch_evaluations += 1

            del clean_cache, clean_logits, corrupt_logits
            if device == "cuda":
                torch.cuda.empty_cache()

    all_rows = finalize_agg_rows(agg_all)
    neg_rows = finalize_agg_rows(agg_neg)
    nonneg_rows = finalize_agg_rows(agg_nonneg)

    all_df = save_heatmap_csv(all_rows, heatmap_all_csv)
    neg_df = save_heatmap_csv(neg_rows, heatmap_neg_csv)
    nonneg_df = save_heatmap_csv(nonneg_rows, heatmap_nonneg_csv)

    plot_heatmap(all_df, heatmap_all_png, "BBQ resid_pre Patching (all)")
    plot_heatmap(neg_df, heatmap_neg_png, "BBQ resid_pre Patching (neg)")
    plot_heatmap(nonneg_df, heatmap_nonneg_png, "BBQ resid_pre Patching (nonneg)")

    elapsed = time.perf_counter() - started_at
    print("\nWrote:")
    print(f"  {raw_out_path}")
    print(f"  {heatmap_all_csv}")
    print(f"  {heatmap_neg_csv}")
    print(f"  {heatmap_nonneg_csv}")
    print(f"  {heatmap_all_png}")
    print(f"  {heatmap_neg_png}")
    print(f"  {heatmap_nonneg_png}")

    print("\nRun stats:")
    print(f"  pairs processed: {len(pairs)}")
    print(f"  layers processed: {n_layers}")
    print(f"  positions processed (sum over pairs): {total_pair_positions}")
    print(f"  total patch evaluations (pairs*layers*positions): {patch_evaluations}")
    print(f"  runtime_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
