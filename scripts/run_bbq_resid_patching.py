#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTERS = ["A", "B", "C"]

# Semantic prompt regions, in display order. Cross-pair aggregation by raw token
# position is only valid for the fixed instruction prefix and the final answer token
# (prompts have different lengths/content), so we also aggregate by these spans, which
# align across pairs and templates.
SPAN_ORDER = ["instruction", "context", "question", "option_A", "option_B", "option_C", "answer"]


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


def pair_key_series(df: pd.DataFrame) -> pd.Series:
    """Content-stable per-pair identifier for crash-safe resume.

    Positional `pair_id` (the iterrows index) is NOT stable across reruns with a different
    --max_pairs / --pair_quality / --context_condition or an edited pairs file, so keying
    resume on it would silently skip/mix the wrong pairs. We key on pair *content* instead.
    Note minimal-swap pairs have corrupt_example_id == clean_example_id, so the source id
    alone is not unique — `swap_identities` (plus polarity/context) disambiguates the
    multiple swaps generated from one source example.
    """
    cols = ["clean_example_id", "swap_identities", "question_polarity", "context_condition"]
    have = [c for c in cols if c in df.columns]
    return df[have].astype(str).agg("|".join, axis=1)


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


def assign_spans(tokenizer, prompt: str) -> list[str]:
    """Map each token of `prompt` to a semantic span in SPAN_ORDER using char offsets.

    The prompt is built by a fixed template, so the structural markers below are
    unique and let us segment any prompt regardless of content length. Section-label
    tokens (``Context:``/``Question:``/``A.``/...) are grouped with the section they
    introduce. Because clean/corrupt pairs are token-aligned, computing spans on the
    clean prompt is valid for both members of a pair.
    """
    enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    def find(marker: str, start: int) -> int:
        pos = prompt.find(marker, start)
        return pos if pos >= 0 else len(prompt)

    i_ctx = find("Context:", 0)
    i_q = find("Question:", i_ctx)
    i_a = find("\nA.", i_q)
    i_b = find("\nB.", i_a)
    i_c = find("\nC.", i_b)
    i_ans = find("\nAnswer:", i_c)

    spans: list[str] = []
    for start, _end in offsets:
        if start < i_ctx:
            spans.append("instruction")
        elif start < i_q:
            spans.append("context")
        elif start < i_a:
            spans.append("question")
        elif start < i_b:
            spans.append("option_A")
        elif start < i_c:
            spans.append("option_B")
        elif start < i_ans:
            spans.append("option_C")
        else:
            spans.append("answer")
    return spans


def aggregate_by_span(raw_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-pair SUM of the single-site effect within each (layer, span), then MEAN across pairs.

    The sum is each span's TOTAL causal contribution for that pair: a single causally hot token
    keeps its full effect and inert tokens add ~0, so a concentrated signal is not diluted by span
    length (which a plain mean over all positions would do). Averaging the per-pair sums then makes
    pairs equally weighted regardless of how many tokens their context/options happen to span.
    (Caveat: summing single-site patches is a first-order attribution, not the effect of patching
    the whole span jointly — but it is the right 'where does the effect live' localization summary.)
    """
    df = raw_df.copy()
    df["span"] = pd.Categorical(df["span"], categories=SPAN_ORDER, ordered=True)
    per_pair = df.groupby(["pair_id", "layer", "span"], observed=True)[value_col].sum()
    return (
        per_pair.groupby(["layer", "span"], observed=True).mean()
        .reset_index()
        .rename(columns={value_col: "value"})
        .sort_values(["layer", "span"])
        .reset_index(drop=True)
    )


def plot_span_heatmap(agg_df: pd.DataFrame, png_path: Path, title: str, colorbar_label: str) -> None:
    if agg_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"{title} (no data)")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return

    present = [s for s in SPAN_ORDER if s in set(agg_df["span"].astype(str))]
    pivot = (
        agg_df.assign(span=agg_df["span"].astype(str))
        .pivot(index="layer", columns="span", values="value")
        .reindex(columns=present)
        .sort_index(axis=0)
    )
    vmax = robust_symmetric_vlim(pd.Series(pivot.values.ravel()))

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.1), 8))
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
    ax.set_xlabel("Prompt span")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_span_identity_heatmap(
    sub_df: pd.DataFrame, value_col: str, png_path: Path, title: str, colorbar_label: str
) -> None:
    """Two panels (swapped-identity tokens vs shared scaffold tokens), layer x span,
    on a shared color scale. Shows whether effect is carried by the identity tokens
    themselves or by propagation into the shared tokens."""
    df = sub_df.copy()
    df["span"] = pd.Categorical(df["span"], categories=SPAN_ORDER, ordered=True)
    # Same per-pair SUM then mean-across-pairs reduction as aggregate_by_span, split by whether the
    # token was a swapped identity token. (Identity + scaffold panels sum back to the main heatmap.)
    per_pair = df.groupby(["pair_id", "layer", "span", "is_identity_token"], observed=True)[value_col].sum()
    g = (
        per_pair.groupby(["layer", "span", "is_identity_token"], observed=True).mean()
        .reset_index()
    )
    if g.empty:
        return
    layers = sorted(int(x) for x in df["layer"].unique())
    vmax = robust_symmetric_vlim(g[value_col])

    fig, axes = plt.subplots(1, 2, figsize=(2 * max(5, len(SPAN_ORDER) * 0.9), 8), sharey=True)
    im = None
    for ax, flag, panel in [(axes[0], 1, "swapped identity tokens"), (axes[1], 0, "shared scaffold tokens")]:
        panel_df = g[g["is_identity_token"] == flag]
        present = [s for s in SPAN_ORDER if s in set(panel_df["span"].astype(str))]
        pivot = (
            panel_df.assign(span=panel_df["span"].astype(str))
            .pivot(index="layer", columns="span", values=value_col)
            .reindex(index=layers, columns=present)
        )
        im = ax.imshow(
            pivot.values, aspect="auto", origin="lower", cmap="coolwarm",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(panel)
        ax.set_xlabel("Prompt span")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right")
    axes[0].set_ylabel("Layer")
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels([str(y) for y in layers])
    fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(colorbar_label)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_span_outputs(raw_df: pd.DataFrame, out_dir: Path, value_col: str) -> list[Path]:
    """Canonical cross-pair-valid aggregation: by semantic span, and by whether each
    position is a swapped identity token. Raw token-position aggregation across pairs is
    intentionally not produced (positions don't align across variable-length prompts)."""
    if "span" not in raw_df.columns:
        print("  [span] raw CSV has no 'span' column; skipping span aggregation (rerun patching to add it).")
        return []

    colorbar = "Mean Bias Effect" if value_col == "bias_effect" else f"Mean {value_col}"
    has_identity = "is_identity_token" in raw_df.columns
    paths: list[Path] = []
    splits = [
        ("all", raw_df),
        ("neg", raw_df[raw_df["question_polarity"].astype(str) == "neg"]),
        ("nonneg", raw_df[raw_df["question_polarity"].astype(str) == "nonneg"]),
    ]
    for suffix, sub in splits:
        agg = aggregate_by_span(sub, value_col)
        csv_path = out_dir / f"bbq_resid_pre_bias_effect_span_heatmap_{suffix}.csv"
        png_path = out_dir / f"bbq_resid_pre_bias_effect_span_heatmap_{suffix}.png"
        agg.rename(columns={"value": f"mean_{value_col}"}).to_csv(csv_path, index=False)
        plot_span_heatmap(agg, png_path, f"BBQ resid_pre by span ({suffix})", colorbar)
        paths.extend([csv_path, png_path])

        if has_identity and not sub.empty:
            id_png = out_dir / f"bbq_resid_pre_bias_effect_span_identity_{suffix}.png"
            plot_span_identity_heatmap(
                sub, value_col, id_png, f"BBQ resid_pre by span x identity ({suffix})", colorbar
            )
            paths.append(id_png)

    # Identity-resolved table (all polarities) backing the span x identity plots.
    if has_identity:
        df = raw_df.copy()
        df["span"] = pd.Categorical(df["span"], categories=SPAN_ORDER, ordered=True)
        ident = (
            df.groupby(["layer", "span", "is_identity_token"], as_index=False, observed=True)[value_col]
            .mean()
            .rename(columns={value_col: f"mean_{value_col}"})
            .sort_values(["layer", "span", "is_identity_token"])
            .reset_index(drop=True)
        )
        ident_path = out_dir / "bbq_resid_pre_bias_effect_by_span_identity.csv"
        ident.to_csv(ident_path, index=False)
        paths.append(ident_path)
    return paths


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


def nonzero_positions(agg_df: pd.DataFrame, value_col: str) -> set[int]:
    """Positions with any non-zero effect across layers.

    In a token-aligned minimal-swap pair, every token before the first swapped
    identity token is identical in clean and corrupt, so patching there is exactly
    0 across all layers. Those structurally-dead columns (the instruction prefix and
    leading context) carry no information and are dropped from the labeled plots.
    """
    per_pos = agg_df.groupby("token_position")[value_col].apply(lambda s: float(s.abs().max()))
    return {int(pos) for pos, vmax in per_pos.items() if vmax > 0.0}


def plot_token_labeled_heatmap(
    agg_df: pd.DataFrame,
    labels_by_pos: dict[int, str],
    value_col: str,
    out_png: Path,
    title: str,
    colorbar_label: str,
    max_label_chars: int,
    top_positions: list[int] | None = None,
    identity_positions: set[int] | None = None,
    boxes: list[tuple[list[int], str, str]] | None = None,
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

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.32), 8))
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
    identity_positions = identity_positions or set()
    ax.set_xlabel("Token (swapped identity tokens in red)" if identity_positions else "Token")

    x_positions = list(pivot.columns)
    xtick_labels = []
    for pos in x_positions:
        pos_int = int(pos)
        token_label = truncate_label(labels_by_pos.get(pos_int, f"tok_{pos_int}"), max_label_chars)
        xtick_labels.append(f"{pos_int}:{token_label}" if top_positions is not None else token_label)
    ax.set_xticks(range(len(x_positions)))
    tick_objs = ax.set_xticklabels(xtick_labels, rotation=75, ha="right", fontsize=7)
    # Highlight the swapped identity tokens (the causal drivers of bias_effect).
    for tick, pos in zip(tick_objs, x_positions):
        if int(pos) in identity_positions:
            tick.set_color("crimson")
            tick.set_fontweight("bold")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])

    # Outline token groups (e.g. the stereotyped identity in the context and in its
    # answer option) so the context->answer mapping is easy to read.
    pos_to_col = {int(p): i for i, p in enumerate(x_positions)}
    n_layers = len(pivot.index)
    legend_handles: dict[str, Rectangle] = {}
    for positions, color, label in boxes or []:
        cols = sorted(pos_to_col[p] for p in positions if p in pos_to_col)
        if not cols:
            continue
        rect = Rectangle(
            (cols[0] - 0.5, -0.5), (cols[-1] - cols[0]) + 1.0, n_layers,
            fill=False, edgecolor=color, linewidth=2.5, zorder=5,
        )
        ax.add_patch(rect)
        legend_handles.setdefault(label, Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=2.5))
    if legend_handles:
        ax.legend(legend_handles.values(), legend_handles.keys(), loc="upper left", fontsize=8, framealpha=0.9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
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
    max_xtick_label_chars: int,
) -> list[Path]:
    out_paths: list[Path] = []
    per_pair_dir = out_dir / "per_pair_bias_effect"
    per_pair_dir.mkdir(parents=True, exist_ok=True)

    token_labels = build_token_label_table(pairs, raw_df, tokenizer)
    labels_diag_path = out_dir / "bbq_resid_pre_bias_effect_token_position_label_diagnostics.csv"
    build_label_diagnostics(token_labels, labels_diag_path)
    out_paths.append(labels_diag_path)

    # NOTE: cross-pair aggregates by raw token position are intentionally NOT produced
    # here. Different pairs have different content at each position index, so averaging
    # by position mislabels and mixes tokens (the "most common token" at a position is
    # usually punctuation like ':' or '.'). The valid cross-pair aggregate is the span
    # heatmap (see write_span_outputs); the per-pair plots below are valid because
    # positions align within a single pair.

    pair_meta = pairs.reset_index().rename(columns={"index": "pair_id"})
    n_pair_plots = min(num_pair_plots, len(pair_meta))
    for i in range(n_pair_plots):
        meta = pair_meta.iloc[i]
        pid = int(meta["pair_id"])
        sub = raw_df[raw_df["pair_id"] == pid]
        if sub.empty:
            continue
        agg = sub.groupby(["layer", "token_position"], as_index=False)[plot_value_col].mean()
        # Keep only positions that carry signal; everything before the first swapped
        # identity token is exactly 0 by construction and adds dead columns.
        keep = nonzero_positions(agg, plot_value_col)
        if not keep:
            continue
        agg = agg[agg["token_position"].isin(keep)].copy()
        pair_labels = token_labels[token_labels["pair_id"] == pid]
        pair_label_map = {
            int(r["token_position"]): normalize_token_label(str(r["clean_token_text"]))
            for _, r in pair_labels.iterrows()
        }
        identity_positions = set()
        boxes: list[tuple[list[int], str, str]] = []
        if {"span", "is_identity_token", "token_text_clean"} <= set(sub.columns):
            identity_positions = {
                int(p)
                for p in sub.loc[sub["is_identity_token"] == 1, "token_position"].unique()
                if int(p) in keep
            }
            # Outline BOTH swapped identities so it's unambiguous which tokens are which:
            #  - the stereotyped TARGET (green; from target_letters, polarity-independent)
            #  - the DISTRACTOR it is swapped against (orange).
            # Each identity is boxed where it appears in the context and as its answer option.
            pos_info = sub.drop_duplicates("token_position")
            pos_info = pos_info[pos_info["token_position"].isin(keep)]

            def boxes_for_option(opt_span: str, color: str, label: str) -> list[tuple[list[int], str, str]]:
                opt = pos_info[(pos_info["span"] == opt_span) & (pos_info["is_identity_token"] == 1)]
                opt_positions = [int(p) for p in opt["token_position"]]
                texts = set(opt["token_text_clean"].astype(str))
                ctx = pos_info[
                    (pos_info["span"] == "context")
                    & (pos_info["is_identity_token"] == 1)
                    & (pos_info["token_text_clean"].astype(str).isin(texts))
                ]
                ctx_positions = [int(p) for p in ctx["token_position"]]
                made = []
                if ctx_positions:
                    made.append((ctx_positions, color, label))
                if opt_positions:
                    made.append((opt_positions, color, label))
                return made

            def meta_group(key: str, fallback: str) -> str:
                v = meta.get(key)
                return str(v) if isinstance(v, str) and v.strip() else fallback

            target_letter = first_letter(str(meta.get("clean_target_letters", "")))
            named_opt_spans = sorted(
                pos_info.loc[
                    (pos_info["is_identity_token"] == 1)
                    & (pos_info["span"].astype(str).str.startswith("option_")),
                    "span",
                ].astype(str).unique()
            )
            target_span = f"option_{target_letter}" if target_letter in LETTERS else None
            distractor_span = next((s for s in named_opt_spans if s != target_span), None)
            if target_span in named_opt_spans:
                boxes += boxes_for_option(
                    target_span, "#00b050", f"target — {meta_group('target_identity_group', 'stereotyped')}"
                )
            if distractor_span:
                boxes += boxes_for_option(
                    distractor_span, "#e07000", f"distractor — {meta_group('distractor_identity_group', 'reference')}"
                )
        out_path = per_pair_dir / (
            f"pair_{pid}_clean_{int(meta['clean_example_id'])}_corrupt_{int(meta['corrupt_example_id'])}.png"
        )
        title = (
            f"pair {pid} | {meta['question_polarity']} | "
            f"{int(meta['clean_example_id'])}->{int(meta['corrupt_example_id'])} | "
            f"readout {meta['clean_biased_letter']} vs {meta['clean_unknown_letter_metric']} | "
            f"target {first_letter(str(meta.get('clean_target_letters', '')))}"
        )
        plot_token_labeled_heatmap(
            agg_df=agg,
            labels_by_pos=pair_label_map,
            value_col=plot_value_col,
            out_png=out_path,
            title=title,
            colorbar_label="Bias Effect" if plot_value_col == "bias_effect" else plot_value_col,
            max_label_chars=max_xtick_label_chars,
            identity_positions=identity_positions,
            boxes=boxes,
        )
        out_paths.append(out_path)

    return out_paths


@torch.no_grad()
def run_patching(
    args,
    pairs: pd.DataFrame,
    raw_out_path: Path,
) -> pd.DataFrame:
    from transformer_lens import HookedTransformer, utils as tl_utils

    pairs = pairs.copy()
    pairs["pair_key"] = pair_key_series(pairs)  # content-stable id for crash-safe resume

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
        "pair_key",
        "clean_example_id",
        "corrupt_example_id",
        "question_index",
        "question_polarity",
        "layer",
        "token_position",
        "span",
        "is_identity_token",
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
    patch_batch_size = max(1, int(args.patch_batch_size))

    # --- Resume / checkpoint: skip pairs already written; per-pair atomic append. ---
    # Resume is keyed on the content-stable `pair_key`, NOT the positional pair_id, so it
    # stays correct even when this run uses different --max_pairs/filters or a reordered
    # pairs file. Legacy raw CSVs (written before pair_key existed) fall back to the old
    # position-based resume with a loud warning.
    done_keys: set[str] = set()
    done_pair_ids: set[int] = set()
    use_key_resume = False
    resume = raw_out_path.exists() and not args.overwrite
    if resume:
        try:
            existing = pd.read_csv(raw_out_path)
        except Exception:
            existing = pd.DataFrame()
        if "pair_key" in existing.columns and existing["pair_key"].notna().any():
            use_key_resume = True
            ordered = existing["pair_key"].astype(str)
            # The pair whose rows are physically last may be half-written from a crash:
            # drop it (every occurrence) and redo it. Order/duplicates don't matter.
            partial = ordered.iloc[-1]
            done_keys = set(ordered) - {partial}
            existing[ordered.isin(done_keys)].reindex(columns=raw_header).to_csv(
                raw_out_path, index=False
            )
            print(f"\nResuming from {raw_out_path}: {len(done_keys)} pairs already complete; "
                  f"redoing pair_key={partial} and continuing.")
            f = raw_out_path.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=raw_header)
        else:
            present = sorted(int(p) for p in existing.get("pair_id", pd.Series([], dtype=int)).dropna().unique())
            if present:
                # Legacy file without pair_key: position-based resume (only safe if this run's
                # --max_pairs/--pair_quality/--context_condition and the pairs file are identical).
                done_pair_ids = set(present[:-1])
                existing[existing["pair_id"].isin(done_pair_ids)].reindex(columns=raw_header).to_csv(
                    raw_out_path, index=False
                )
                print(f"\nWARNING: resuming a legacy raw without pair_key by POSITION "
                      f"({len(done_pair_ids)} pairs); this is only correct if the pairs file and "
                      f"--max_pairs/--pair_quality/--context_condition are unchanged. Use --overwrite "
                      f"to start fresh.\nRedoing pair {present[-1]} and continuing.")
                f = raw_out_path.open("a", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=raw_header)
            else:
                resume = False
    if not resume:
        f = raw_out_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=raw_header)
        writer.writeheader()

    try:
        for pair_id, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Patching pairs"):
            pair_id = int(pair_id)
            pair_key = str(row["pair_key"])
            if (pair_key in done_keys) if use_key_resume else (pair_id in done_pair_ids):
                continue
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

            spans = assign_spans(tokenizer, clean_prompt)

            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens, names_filter=lambda n: n.endswith("hook_resid_pre")
            )
            corrupt_logits = model(corrupt_tokens)
            clean_bias_metric = metric_from_logits(clean_logits[0, -1, :].float(), clean_b_tok, clean_u_tok)
            corrupt_bias_metric = metric_from_logits(corrupt_logits[0, -1, :].float(), corrupt_b_tok, corrupt_u_tok)

            pair_rows: list[dict[str, Any]] = []
            for layer in range(n_layers):
                act_name = tl_utils.get_act_name("resid_pre", layer)
                clean_layer_act = clean_cache[act_name]  # [1, T, d]
                # Patch positions in batches: B copies of the corrupt prompt, each row
                # patched at its own position, so one forward yields B position results.
                for chunk_start in range(0, max_position, patch_batch_size):
                    chunk = list(range(chunk_start, min(chunk_start + patch_batch_size, max_position)))
                    chunk_t = torch.tensor(chunk, device=device)
                    batched = corrupt_tokens.repeat(len(chunk), 1)

                    def patch_fn(act, hook, chunk_t=chunk_t, clean_act=clean_layer_act):
                        act = act.clone()
                        rows = torch.arange(act.shape[0], device=act.device)
                        act[rows, chunk_t, :] = clean_act[0, chunk_t, :].to(act.dtype)
                        return act

                    patched_logits = model.run_with_hooks(batched, fwd_hooks=[(act_name, patch_fn)])
                    last = patched_logits[:, -1, :].float()  # [B, vocab]
                    for i, pos in enumerate(chunk):
                        clean_tok_id = int(clean_tokens[0, pos].item())
                        corrupt_tok_id = int(corrupt_tokens[0, pos].item())
                        patched_bias_metric = float((last[i, corrupt_b_tok] - last[i, corrupt_u_tok]).item())
                        bias_effect = patched_bias_metric - corrupt_bias_metric
                        normalized_restoration = safe_normalized(
                            bias_effect, clean_bias_metric, corrupt_bias_metric
                        )
                        token_text_corrupt = tokenizer.decode([corrupt_tok_id]).replace("\n", "\\n")
                        pair_rows.append(
                            {
                                "pair_id": pair_id,
                                "pair_key": pair_key,
                                "clean_example_id": row["clean_example_id"],
                                "corrupt_example_id": row["corrupt_example_id"],
                                "question_index": row["question_index"],
                                "question_polarity": row["question_polarity"],
                                "layer": layer,
                                "token_position": pos,
                                "span": spans[pos] if pos < len(spans) else "answer",
                                "is_identity_token": int(clean_tok_id != corrupt_tok_id),
                                "token_text": token_text_corrupt,
                                "token_text_clean": tokenizer.decode([clean_tok_id]).replace("\n", "\\n"),
                                "token_text_corrupt": token_text_corrupt,
                                "clean_bias_metric": clean_bias_metric,
                                "corrupt_bias_metric": corrupt_bias_metric,
                                "patched_bias_metric": patched_bias_metric,
                                "bias_effect": bias_effect,
                                "clean_metric": clean_bias_metric,
                                "corrupt_metric": corrupt_bias_metric,
                                "patched_metric": patched_bias_metric,
                                "raw_restoration": bias_effect,
                                "normalized_restoration": normalized_restoration,
                            }
                        )
                        patch_evaluations += 1

            # Atomic per-pair write + flush → safe to resume after a crash.
            writer.writerows(pair_rows)
            f.flush()

            del clean_cache, clean_logits, corrupt_logits
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        f.close()

    raw_df = pd.read_csv(raw_out_path)
    raw_df = ensure_metric_columns(raw_df)

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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the baseline-metric pass.")
    parser.add_argument(
        "--patch_batch_size",
        type=int,
        default=16,
        help="Number of token positions patched per forward pass (batched over positions within a "
        "layer). Higher = faster on CUDA, more memory. Equivalent to 1 up to floating-point "
        "batching nondeterminism (~1e-5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start the raw CSV fresh. Default resumes from an existing raw CSV (skips completed pairs).",
    )
    parser.add_argument("--context_condition", type=str, default="ambig")
    parser.add_argument("--pair_quality", type=str, default="strict")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--preview_pairs", type=int, default=5)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--raw_csv", type=Path, default=None)
    parser.add_argument("--make_token_labeled_plots", action="store_true")
    parser.add_argument("--num_pair_plots", type=int, default=10)
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
    else:
        raw_df = run_patching(args=args, pairs=pairs, raw_out_path=raw_out_path)

    written_paths = [raw_out_path]

    # Canonical cross-pair-valid aggregation: by semantic span (and identity-token flag).
    written_paths.extend(write_span_outputs(raw_df, args.out_dir, args.plot_value_col))

    if args.make_token_labeled_plots:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        token_plot_paths = create_token_labeled_plots(
            raw_df=raw_df,
            pairs=pairs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            plot_value_col=args.plot_value_col,
            num_pair_plots=args.num_pair_plots,
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
