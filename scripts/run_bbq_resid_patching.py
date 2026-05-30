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


LETTERS = ["A", "B", "C"]


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


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


def truncate_label(text: str, max_chars: int) -> str:
    clean = text.replace("\n", "\\n")
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1] + "…"


def normalize_token_label(token: str) -> str:
    text = str(token)
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text


def is_interpretable_token_label(token: str) -> bool:
    text = normalize_token_label(token)
    if not text:
        return False
    # Skip common special/control placeholders that are not semantically useful.
    if text.startswith("<|") and text.endswith("|>"):
        return False
    if text in {"<s>", "</s>", "<pad>", "[PAD]", "[BOS]", "[EOS]"}:
        return False
    return True


def robust_symmetric_vlim(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return 1.0
    vmax = float(finite.abs().quantile(0.95))
    return max(vmax, 1e-6)


def prepare_pairs(pairs_csv: Path, pair_quality: str, context_condition: str, max_pairs: int | None) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_csv)
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

    if pair_quality != "all":
        pairs = pairs[pairs["pair_quality"].astype(str) == pair_quality].copy()
    if context_condition != "all":
        pairs = pairs[pairs["context_condition"].astype(str) == context_condition].copy()

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

    if max_pairs is not None:
        pairs = pairs.head(max_pairs).copy()
    if pairs.empty:
        raise ValueError("No pairs available after filtering. Check pair_quality/context_condition/max_pairs.")

    return pairs.reset_index(drop=True)


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


def print_pair_bias_metric_preview(pairs: pd.DataFrame, n: int) -> None:
    n_show = min(n, len(pairs))
    print(f"\nPair bias-metric preview ({n_show}):")
    for i in range(n_show):
        row = pairs.iloc[i]
        print(
            f"  {int(row['clean_example_id'])}->{int(row['corrupt_example_id'])} | "
            f"polarity={row['question_polarity']} | "
            f"clean {row['clean_biased_letter']}/{row['clean_unknown_letter_metric']} "
            f"metric={row['clean_bias_metric']:.6f} | "
            f"corrupt {row['corrupt_biased_letter']}/{row['corrupt_unknown_letter_metric']} "
            f"metric={row['corrupt_bias_metric']:.6f}"
        )


def ensure_metric_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "clean_bias_metric" not in df.columns and "clean_metric" in df.columns:
        df["clean_bias_metric"] = df["clean_metric"]
    if "corrupt_bias_metric" not in df.columns and "corrupt_metric" in df.columns:
        df["corrupt_bias_metric"] = df["corrupt_metric"]
    if "patched_bias_metric" not in df.columns and "patched_metric" in df.columns:
        df["patched_bias_metric"] = df["patched_metric"]
    if (
        "bias_effect" not in df.columns
        and "patched_bias_metric" in df.columns
        and "corrupt_bias_metric" in df.columns
    ):
        df["bias_effect"] = (
            pd.to_numeric(df["patched_bias_metric"], errors="coerce")
            - pd.to_numeric(df["corrupt_bias_metric"], errors="coerce")
        )
    return df


@torch.no_grad()
def compute_prompt_metrics_batch(
    model,
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
        b_tok = letter_token_ids[biased_letters[i]]
        u_tok = letter_token_ids[unknown_letters[i]]
        metrics.append(float((next_logits[i, b_tok] - next_logits[i, u_tok]).item()))
    return metrics


def aggregate_from_raw(raw_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        raw_df.groupby(["layer", "token_position"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
        .sort_values(["layer", "token_position"])
        .reset_index(drop=True)
    )


def save_aggregate_heatmap_csv(df: pd.DataFrame, out_path: Path, value_col: str) -> pd.DataFrame:
    mean_col = f"mean_{value_col}"
    out = df.rename(columns={"value": mean_col}).copy()
    out = out[["layer", "token_position", mean_col]]
    out.to_csv(out_path, index=False)
    return out


def plot_numeric_heatmap(df: pd.DataFrame, png_path: Path, title: str, colorbar_label: str) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return

    pivot = df.pivot(index="layer", columns="token_position", values="value").sort_index(axis=0).sort_index(axis=1)
    vmax = robust_symmetric_vlim(df["value"])

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def build_token_label_table(pairs: pd.DataFrame, raw_df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_pair_pos = (
        raw_df.groupby("pair_id")["token_position"].max().reset_index().rename(columns={"token_position": "max_pos"})
    )
    pair_map = pairs.reset_index().rename(columns={"index": "pair_id"})
    merged = raw_pair_pos.merge(pair_map, on="pair_id", how="left")

    for _, row in merged.iterrows():
        clean_prompt = str(row["clean_prompt"])
        corrupt_prompt = str(row["corrupt_prompt"])
        clean_ids = tokenizer(clean_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        max_position = min(int(row["max_pos"]) + 1, len(clean_ids), len(corrupt_ids))
        for pos in range(max_position):
            rows.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "token_position": pos,
                    "clean_token_text": tokenizer.decode([clean_ids[pos]]).replace("\n", "\\n"),
                    "corrupt_token_text": tokenizer.decode([corrupt_ids[pos]]).replace("\n", "\\n"),
                }
            )
    return pd.DataFrame(rows)


def build_label_diagnostics(token_labels: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for pos, sub in token_labels.groupby("token_position"):
        counts = sub["clean_token_text"].value_counts()
        total = int(counts.sum())
        most = counts.index[0]
        most_count = int(counts.iloc[0])
        top5 = "; ".join([f"{tok}:{int(cnt)}" for tok, cnt in counts.head(5).items()])
        records.append(
            {
                "token_position": int(pos),
                "most_common_token": most,
                "most_common_token_count": most_count,
                "total_count": total,
                "agreement_rate": most_count / total if total else float("nan"),
                "is_interpretable": is_interpretable_token_label(most),
                "top_5_tokens_at_position": top5,
            }
        )
    diag = pd.DataFrame(records).sort_values("token_position").reset_index(drop=True)
    diag.to_csv(out_path, index=False)
    return diag


def representative_labels(token_labels: pd.DataFrame) -> dict[int, str]:
    if token_labels.empty:
        return {}
    reps = (
        token_labels.groupby("token_position")["clean_token_text"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return {int(k): normalize_token_label(str(v)) for k, v in reps.items()}


def plot_token_labeled_heatmap(
    agg_df: pd.DataFrame,
    labels_by_pos: dict[int, str],
    value_col: str,
    out_png: Path,
    title: str,
    colorbar_label: str,
    max_label_chars: int,
    top_positions: list[int] | None = None,
) -> None:
    if agg_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"{title} (no data)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    plot_df = agg_df.copy()
    if top_positions is not None:
        plot_df = plot_df[plot_df["token_position"].isin(top_positions)].copy()
        plot_df["token_position"] = pd.Categorical(plot_df["token_position"], categories=top_positions, ordered=True)

    pivot = plot_df.pivot(index="layer", columns="token_position", values=value_col)
    if top_positions is None:
        pivot = pivot.sort_index(axis=1)
    pivot = pivot.sort_index(axis=0)

    flat_vals = pd.Series(pivot.values.ravel())
    vmax = robust_symmetric_vlim(flat_vals)

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.35), 8))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_ylabel("Layer")
    ax.set_xlabel("Token Position")

    x_positions = list(pivot.columns)
    xtick_labels = []
    for pos in x_positions:
        pos_int = int(pos)
        token_label = truncate_label(labels_by_pos.get(pos_int, f"tok_{pos_int}"), max_label_chars)
        xtick_labels.append(f"{pos_int}:{token_label}" if top_positions is not None else token_label)
    ax.set_xticks(range(len(x_positions)))
    ax.set_xticklabels(xtick_labels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    # TODO: optional shaded prompt spans (Context/Question/A/B/C/Answer) can be added later.
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def create_token_labeled_plots(
    raw_df: pd.DataFrame,
    pairs: pd.DataFrame,
    tokenizer,
    out_dir: Path,
    plot_value_col: str,
    num_pair_plots: int,
    top_k_positions: int,
    max_xtick_label_chars: int,
) -> list[Path]:
    out_paths: list[Path] = []
    per_pair_dir = out_dir / "per_pair_bias_effect"
    per_pair_dir.mkdir(parents=True, exist_ok=True)

    token_labels = build_token_label_table(pairs, raw_df, tokenizer)
    labels_diag_path = out_dir / "bbq_resid_pre_bias_effect_token_position_label_diagnostics.csv"
    label_diag = build_label_diagnostics(token_labels, labels_diag_path)
    out_paths.append(labels_diag_path)
    labels_by_pos = representative_labels(token_labels)
    valid_positions = set(
        label_diag[label_diag["is_interpretable"] == True]["token_position"].astype(int).tolist()
    )

    def agg_and_plot(sub_df: pd.DataFrame, suffix: str) -> None:
        agg = sub_df.groupby(["layer", "token_position"], as_index=False)[plot_value_col].mean()
        agg = agg[agg["token_position"].isin(valid_positions)].copy()
        labeled_png = out_dir / f"bbq_resid_pre_bias_effect_token_labeled_heatmap_{suffix}.png"
        plot_token_labeled_heatmap(
            agg_df=agg,
            labels_by_pos=labels_by_pos,
            value_col=plot_value_col,
            out_png=labeled_png,
            title=f"BBQ resid_pre token-labeled ({suffix})",
            colorbar_label="Mean Bias Effect" if plot_value_col == "bias_effect" else f"Mean {plot_value_col}",
            max_label_chars=max_xtick_label_chars,
        )
        out_paths.append(labeled_png)

        pos_importance = (
            agg.groupby("token_position")[plot_value_col].apply(lambda s: s.abs().mean()).reset_index(name="importance")
        )
        top_positions = (
            pos_importance.sort_values("importance", ascending=False)
            .head(top_k_positions)["token_position"]
            .sort_values()
            .astype(int)
            .tolist()
        )
        top_png = out_dir / f"bbq_resid_pre_top_positions_{suffix}.png"
        plot_token_labeled_heatmap(
            agg_df=agg,
            labels_by_pos=labels_by_pos,
            value_col=plot_value_col,
            out_png=top_png,
            title=f"BBQ resid_pre top-{top_k_positions} positions ({suffix})",
            colorbar_label="Mean Bias Effect" if plot_value_col == "bias_effect" else f"Mean {plot_value_col}",
            max_label_chars=max_xtick_label_chars,
            top_positions=top_positions,
        )
        out_paths.append(top_png)

    agg_and_plot(raw_df, "all")
    agg_and_plot(raw_df[raw_df["question_polarity"].astype(str) == "neg"], "neg")
    agg_and_plot(raw_df[raw_df["question_polarity"].astype(str) == "nonneg"], "nonneg")

    pair_meta = pairs.reset_index().rename(columns={"index": "pair_id"})
    n_pair_plots = min(num_pair_plots, len(pair_meta))
    for i in range(n_pair_plots):
        meta = pair_meta.iloc[i]
        pid = int(meta["pair_id"])
        sub = raw_df[raw_df["pair_id"] == pid]
        if sub.empty:
            continue
        sub = sub[sub["token_position"].isin(valid_positions)].copy()
        if sub.empty:
            continue
        pair_labels = token_labels[token_labels["pair_id"] == pid]
        pair_label_map = {
            int(r["token_position"]): normalize_token_label(str(r["clean_token_text"]))
            for _, r in pair_labels.iterrows()
            if int(r["token_position"]) in valid_positions
        }
        agg = sub.groupby(["layer", "token_position"], as_index=False)[plot_value_col].mean()
        out_path = per_pair_dir / (
            f"pair_{pid}_clean_{int(meta['clean_example_id'])}_corrupt_{int(meta['corrupt_example_id'])}.png"
        )
        title = (
            f"pair {pid} | {meta['question_polarity']} | "
            f"{int(meta['clean_example_id'])}->{int(meta['corrupt_example_id'])} | "
            f"{meta['clean_biased_letter']} vs {meta['clean_unknown_letter_metric']}"
        )
        plot_token_labeled_heatmap(
            agg_df=agg,
            labels_by_pos=pair_label_map,
            value_col=plot_value_col,
            out_png=out_path,
            title=title,
            colorbar_label="Bias Effect" if plot_value_col == "bias_effect" else plot_value_col,
            max_label_chars=max_xtick_label_chars,
        )
        out_paths.append(out_path)

    return out_paths


@torch.no_grad()
def run_patching(
    args,
    pairs: pd.DataFrame,
    raw_out_path: Path,
    heatmap_all_csv: Path,
    heatmap_neg_csv: Path,
    heatmap_nonneg_csv: Path,
    heatmap_all_png: Path,
    heatmap_neg_png: Path,
    heatmap_nonneg_png: Path,
) -> pd.DataFrame:
    from transformer_lens import HookedTransformer, utils as tl_utils

    device = resolve_device(args.device)
    if device == "mps" and args.dtype == "bfloat16":
        print("MPS + bfloat16 can be unreliable; switching to float16.")
        args.dtype = "float16"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
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

    print("\nLoading HF model...")
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
    pairs["clean_bias_metric"] = clean_metrics
    pairs["corrupt_bias_metric"] = corrupt_metrics
    pairs["clean_metric"] = clean_metrics
    pairs["corrupt_metric"] = corrupt_metrics
    print_pair_bias_metric_preview(pairs, args.preview_pairs)
    print(f"\nmean clean_bias_metric: {pairs['clean_bias_metric'].mean():.6f}")
    print(f"mean corrupt_bias_metric: {pairs['corrupt_bias_metric'].mean():.6f}")

    raw_header = [
        "pair_id",
        "clean_example_id",
        "corrupt_example_id",
        "question_index",
        "question_polarity",
        "layer",
        "token_position",
        "token_text",
        "token_text_clean",
        "token_text_corrupt",
        "clean_bias_metric",
        "corrupt_bias_metric",
        "patched_bias_metric",
        "bias_effect",
        "clean_metric",
        "corrupt_metric",
        "patched_metric",
        "raw_restoration",
        "normalized_restoration",
    ]

    patch_evaluations = 0
    total_pair_positions = 0
    n_layers = int(model.cfg.n_layers)

    with raw_out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_header)
        writer.writeheader()

        for pair_id, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Patching pairs"):
            clean_prompt = str(row["clean_prompt"])
            corrupt_prompt = str(row["corrupt_prompt"])
            clean_b_tok = letter_token_ids[str(row["clean_biased_letter"])]
            clean_u_tok = letter_token_ids[str(row["clean_unknown_letter_metric"])]
            corrupt_b_tok = letter_token_ids[str(row["corrupt_biased_letter"])]
            corrupt_u_tok = letter_token_ids[str(row["corrupt_unknown_letter_metric"])]

            clean_tokens = tokenizer(clean_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
            corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
            max_position = min(int(clean_tokens.shape[1]), int(corrupt_tokens.shape[1]))
            total_pair_positions += max_position

            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=lambda n: n.endswith("hook_resid_pre")
            )
            corrupt_logits = model(corrupt_tokens)
            clean_bias_metric = metric_from_logits(clean_logits[0, -1, :].float(), clean_b_tok, clean_u_tok)
            corrupt_bias_metric = metric_from_logits(corrupt_logits[0, -1, :].float(), corrupt_b_tok, corrupt_u_tok)

            for layer in range(n_layers):
                act_name = tl_utils.get_act_name("resid_pre", layer)
                clean_layer_act = clean_cache[act_name]
                for pos in range(max_position):
                    clean_tok_id = int(clean_tokens[0, pos].item())
                    corrupt_tok_id = int(corrupt_tokens[0, pos].item())
                    token_text_clean = tokenizer.decode([clean_tok_id]).replace("\n", "\\n")
                    token_text_corrupt = tokenizer.decode([corrupt_tok_id]).replace("\n", "\\n")

                    def patch_fn(act, hook, pos_idx=pos, clean_act=clean_layer_act):
                        act = act.clone()
                        act[:, pos_idx, :] = clean_act[:, pos_idx, :]
                        return act

                    patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(act_name, patch_fn)])
                    # Primary bias-localization effect:
                    # bias_effect > 0: patching clean activation into corrupt increases stereotyped preference.
                    # bias_effect < 0: patching reduces stereotyped preference.
                    patched_bias_metric = metric_from_logits(
                        patched_logits[0, -1, :].float(), corrupt_b_tok, corrupt_u_tok
                    )
                    bias_effect = patched_bias_metric - corrupt_bias_metric

                    # Kept for backward compatibility with prior analyses.
                    raw_restoration = bias_effect
                    normalized_restoration = safe_normalized(
                        raw_restoration, clean_bias_metric, corrupt_bias_metric
                    )

                    writer.writerow(
                        {
                            "pair_id": pair_id,
                            "clean_example_id": row["clean_example_id"],
                            "corrupt_example_id": row["corrupt_example_id"],
                            "question_index": row["question_index"],
                            "question_polarity": row["question_polarity"],
                            "layer": layer,
                            "token_position": pos,
                            "token_text": token_text_corrupt,
                            "token_text_clean": token_text_clean,
                            "token_text_corrupt": token_text_corrupt,
                            "clean_bias_metric": clean_bias_metric,
                            "corrupt_bias_metric": corrupt_bias_metric,
                            "patched_bias_metric": patched_bias_metric,
                            "bias_effect": bias_effect,
                            "clean_metric": clean_bias_metric,
                            "corrupt_metric": corrupt_bias_metric,
                            "patched_metric": patched_bias_metric,
                            "raw_restoration": raw_restoration,
                            "normalized_restoration": normalized_restoration,
                        }
                    )
                    patch_evaluations += 1

            del clean_cache, clean_logits, corrupt_logits
            if device == "cuda":
                torch.cuda.empty_cache()

    raw_df = pd.read_csv(raw_out_path)
    raw_df = ensure_metric_columns(raw_df)
    value_col = args.plot_value_col
    all_df = aggregate_from_raw(raw_df, value_col)
    neg_df = aggregate_from_raw(raw_df[raw_df["question_polarity"].astype(str) == "neg"], value_col)
    nonneg_df = aggregate_from_raw(raw_df[raw_df["question_polarity"].astype(str) == "nonneg"], value_col)
    save_aggregate_heatmap_csv(all_df, heatmap_all_csv, value_col)
    save_aggregate_heatmap_csv(neg_df, heatmap_neg_csv, value_col)
    save_aggregate_heatmap_csv(nonneg_df, heatmap_nonneg_csv, value_col)
    colorbar = "Mean Bias Effect" if value_col == "bias_effect" else f"Mean {value_col}"
    plot_numeric_heatmap(all_df, heatmap_all_png, "BBQ resid_pre Patching (all)", colorbar)
    plot_numeric_heatmap(neg_df, heatmap_neg_png, "BBQ resid_pre Patching (neg)", colorbar)
    plot_numeric_heatmap(nonneg_df, heatmap_nonneg_png, "BBQ resid_pre Patching (nonneg)", colorbar)

    print("\nRun stats:")
    print(f"  pairs processed: {len(pairs)}")
    print(f"  layers processed: {n_layers}")
    print(f"  positions processed (sum over pairs): {total_pair_positions}")
    print(f"  total patch evaluations (pairs*layers*positions): {patch_evaluations}")
    return raw_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or plot BBQ resid_pre activation patching.")
    parser.add_argument("--pairs_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_condition", type=str, default="ambig")
    parser.add_argument("--pair_quality", type=str, default="strict")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--preview_pairs", type=int, default=5)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--raw_csv", type=Path, default=None)
    parser.add_argument("--make_token_labeled_plots", action="store_true")
    parser.add_argument("--num_pair_plots", type=int, default=10)
    parser.add_argument("--top_k_positions", type=int, default=40)
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=["bias_effect", "restoration"],
        default="bias_effect",
        help="Primary plotting/aggregation mode. bias_effect is default.",
    )
    parser.add_argument("--plot_value_col", type=str, default=None)
    parser.add_argument("--max_xtick_label_chars", type=int, default=18)
    args = parser.parse_args()

    if args.plot_value_col is None:
        args.plot_value_col = (
            "bias_effect" if args.metric_mode == "bias_effect" else "normalized_restoration"
        )

    started_at = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_path = args.raw_csv or (args.out_dir / "bbq_resid_pre_bias_effect_raw.csv")
    legacy_raw_out_path = args.out_dir / "bbq_resid_pre_patching_raw.csv"
    heatmap_all_csv = args.out_dir / "bbq_resid_pre_bias_effect_heatmap_all.csv"
    heatmap_neg_csv = args.out_dir / "bbq_resid_pre_bias_effect_heatmap_neg.csv"
    heatmap_nonneg_csv = args.out_dir / "bbq_resid_pre_bias_effect_heatmap_nonneg.csv"
    heatmap_all_png = args.out_dir / "bbq_resid_pre_bias_effect_heatmap_all.png"
    heatmap_neg_png = args.out_dir / "bbq_resid_pre_bias_effect_heatmap_neg.png"
    heatmap_nonneg_png = args.out_dir / "bbq_resid_pre_bias_effect_heatmap_nonneg.png"

    pairs = prepare_pairs(args.pairs_csv, args.pair_quality, args.context_condition, args.max_pairs)
    print(f"Pairs loaded: {len(pairs)}")
    print(f"metric_mode: {args.metric_mode} (plot_value_col={args.plot_value_col})")
    print_pair_preview(pairs, args.preview_pairs)

    if args.plot_only:
        if args.raw_csv is None and (not raw_out_path.exists()) and legacy_raw_out_path.exists():
            raw_out_path = legacy_raw_out_path
        if not raw_out_path.exists():
            raise FileNotFoundError(f"--plot_only requested but raw CSV not found: {raw_out_path}")
        raw_df = ensure_metric_columns(pd.read_csv(raw_out_path))
        if args.plot_value_col not in raw_df.columns:
            raise ValueError(f"plot_value_col {args.plot_value_col!r} not found in raw CSV")
        all_df = aggregate_from_raw(raw_df, args.plot_value_col)
        neg_df = aggregate_from_raw(raw_df[raw_df["question_polarity"].astype(str) == "neg"], args.plot_value_col)
        nonneg_df = aggregate_from_raw(raw_df[raw_df["question_polarity"].astype(str) == "nonneg"], args.plot_value_col)
        save_aggregate_heatmap_csv(all_df, heatmap_all_csv, args.plot_value_col)
        save_aggregate_heatmap_csv(neg_df, heatmap_neg_csv, args.plot_value_col)
        save_aggregate_heatmap_csv(nonneg_df, heatmap_nonneg_csv, args.plot_value_col)
        colorbar = (
            "Mean Bias Effect" if args.plot_value_col == "bias_effect"
            else f"Mean {args.plot_value_col}"
        )
        plot_numeric_heatmap(all_df, heatmap_all_png, "BBQ resid_pre Patching (all)", colorbar)
        plot_numeric_heatmap(neg_df, heatmap_neg_png, "BBQ resid_pre Patching (neg)", colorbar)
        plot_numeric_heatmap(nonneg_df, heatmap_nonneg_png, "BBQ resid_pre Patching (nonneg)", colorbar)
    else:
        raw_df = run_patching(
            args=args,
            pairs=pairs,
            raw_out_path=raw_out_path,
            heatmap_all_csv=heatmap_all_csv,
            heatmap_neg_csv=heatmap_neg_csv,
            heatmap_nonneg_csv=heatmap_nonneg_csv,
            heatmap_all_png=heatmap_all_png,
            heatmap_neg_png=heatmap_neg_png,
            heatmap_nonneg_png=heatmap_nonneg_png,
        )

    written_paths = [
        raw_out_path,
        heatmap_all_csv,
        heatmap_neg_csv,
        heatmap_nonneg_csv,
        heatmap_all_png,
        heatmap_neg_png,
        heatmap_nonneg_png,
    ]

    if args.make_token_labeled_plots:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        token_plot_paths = create_token_labeled_plots(
            raw_df=raw_df,
            pairs=pairs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            plot_value_col=args.plot_value_col,
            num_pair_plots=args.num_pair_plots,
            top_k_positions=args.top_k_positions,
            max_xtick_label_chars=args.max_xtick_label_chars,
        )
        written_paths.extend(token_plot_paths)

    elapsed = time.perf_counter() - started_at
    print("\nWrote:")
    for p in written_paths:
        print(f"  {p}")
    print(f"\nruntime_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
