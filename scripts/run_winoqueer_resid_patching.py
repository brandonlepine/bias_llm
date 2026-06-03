#!/usr/bin/env python3
"""Per-token, per-layer residual-stream activation patching for WinoQueer.

Question: does injecting the QUEER residual state into the CONTROL run make the model assign
the stereotype continuation (the predicate) a higher probability — i.e. does the queer state
*create* the bias, and where (which layer / token) does it live?

    queer_variant   (source) = sent_x = prefix_x + continuation   (LGBTQ identity)
    control_variant (target) = sent_y = prefix_y + continuation    (straight/cisgender control)

The two prompts differ only in the identity phrase, so they share a common token PREFIX and a
common token SUFFIX; the single contiguous differing span in the middle is the identity. We:
  - cache queer `resid_pre` activations once,
  - for each (layer, control token position) patch the queer activation at the ALIGNED source
    position into the control run, and re-score the continuation,
  - metric: bias_effect = patched_continuation_avg_logp - control_continuation_avg_logp
            (>0 => injecting the queer state raised the stereotype continuation's probability).

Token alignment (stringent):
  P = longest common token prefix, S = longest common token suffix of (queer_ids, control_ids).
  Everything in [0, P) is identical (1:1, and bitwise-identical activations -> zero effect).
  Everything in [P, end) is shifted by delta = len_queer - len_control, which:
    * end-aligns the shared post-identity scaffold and the continuation EXACTLY (so the readout
      tokens correspond perfectly), and
    * maps each control identity token to the end-aligned queer identity token (clamped into the
      queer identity span). The identity is the only span where lengths differ.
  spans: shared_pre / identity / shared_post / continuation. We also patch the WHOLE identity
  span at once per layer (token_position = -1, span = "identity_all").
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SPAN_ORDER = ["shared_pre", "identity", "shared_post", "continuation"]


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def common_prefix_len(a: list[int], b: list[int]) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def common_suffix_len(a: list[int], b: list[int]) -> int:
    k = 0
    while k < len(a) and k < len(b) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def align_pair(tokenizer, sent_x: str, sent_y: str, prefix_y: str) -> dict[str, Any] | None:
    """Token-level alignment of the control prompt (sent_y) to the queer prompt (sent_x).

    Returns None with a reason if the pair is not a clean single-identity-span difference.
    """
    q_ids = tokenizer(sent_x, add_special_tokens=True)["input_ids"]
    c_ids = tokenizer(sent_y, add_special_tokens=True)["input_ids"]
    len_q, len_c = len(q_ids), len(c_ids)

    P = common_prefix_len(q_ids, c_ids)
    S = common_suffix_len(q_ids, c_ids)
    # Prevent prefix/suffix from overlapping (cap so the differing span is well defined).
    S = min(S, len_q - P, len_c - P)
    Lx = len_q - P - S   # queer identity-span token count
    Ly = len_c - P - S   # control identity-span token count
    if Lx <= 0 or Ly <= 0:
        return {"ok": False, "reason": "no contiguous identity span (lengths)"}

    # Continuation readout positions in the control prompt: tokens after prefix_y.
    pref_c_ids = tokenizer(prefix_y, add_special_tokens=True)["input_ids"]
    cont_start_c = common_prefix_len(c_ids, pref_c_ids)
    cont_count = len_c - cont_start_c
    if cont_count <= 0:
        return {"ok": False, "reason": "empty continuation in control"}
    # The continuation must lie inside the shared suffix (i.e. be identical in both prompts).
    if cont_start_c < P + Ly or cont_start_c < len_c - S:
        # continuation begins before the post-identity shared region -> not cleanly aligned
        return {"ok": False, "reason": "continuation not within shared suffix"}

    delta = len_q - len_c  # = Lx - Ly
    # Source (queer) position for each control position, and span label.
    source_pos: list[int] = []
    spans: list[str] = []
    for c in range(len_c):
        if c < P:
            q = c
            span = "shared_pre"
        else:
            q = min(max(c + delta, P), len_q - 1)
            if c < P + Ly:
                span = "identity"
            elif c < cont_start_c:
                span = "shared_post"
            else:
                span = "continuation"
        source_pos.append(q)
        spans.append(span)

    identity_control_positions = list(range(P, P + Ly))
    identity_source_positions = [min(max(c + delta, P), len_q - 1) for c in identity_control_positions]

    return {
        "ok": True,
        "q_ids": q_ids,
        "c_ids": c_ids,
        "source_pos": source_pos,
        "spans": spans,
        "cont_start_c": cont_start_c,
        "cont_count": cont_count,
        "P": P, "S": S, "Lx": Lx, "Ly": Ly,
        "identity_control_positions": identity_control_positions,
        "identity_source_positions": identity_source_positions,
    }


def continuation_logp(logits: torch.Tensor, target_ids: torch.Tensor, cont_start: int) -> torch.Tensor:
    """Sum of log P(continuation tokens) per batch row. `logits` [B,T,V]; `target_ids` [T] is the
    (shared) sequence being scored; continuation = positions [cont_start, T)."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = target_ids[1:]  # [T-1]
    tok_lp = lp[:, :-1, :].gather(-1, tgt.view(1, -1, 1).expand(lp.size(0), -1, 1)).squeeze(-1)  # [B,T-1]
    seg = tok_lp[:, cont_start - 1 : target_ids.shape[0] - 1]  # continuation token logprobs
    return seg.sum(dim=1)


RAW_HEADER = [
    "pair_id", "row_id", "Gender_ID_x", "Gender_ID_y", "predicate", "predicate_label_provisional",
    "layer", "token_position", "span", "source_position", "is_identity_token",
    "token_text_control", "token_text_source",
    "control_cont_avg_logp", "queer_cont_avg_logp", "patched_cont_avg_logp",
    "bias_effect", "normalized_restoration",
]


def safe_norm(num: float, denom: float) -> float:
    return num / denom if abs(denom) > 1e-8 else float("nan")


@torch.no_grad()
def run_patching(args, pairs: pd.DataFrame, raw_out_path: Path) -> pd.DataFrame:
    from transformer_lens import HookedTransformer, utils as tl_utils

    device = resolve_device(args.device)
    if device == "mps" and args.dtype == "bfloat16":
        print("MPS + bfloat16 can be unreliable; switching to float16.")
        args.dtype = "float16"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"\nDevice: {device} | dtype: {args.dtype} | patch_batch_size: {args.patch_batch_size}")

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
    n_layers = int(model.cfg.n_layers)
    pbs = max(1, int(args.patch_batch_size))

    # ---- resume ----
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

    skipped = 0
    try:
        for pair_id, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Patching pairs"):
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
            len_c = c_ids.shape[1]
            cont_start = aln["cont_start_c"]
            cont_count = aln["cont_count"]
            source_pos = aln["source_pos"]
            spans = aln["spans"]

            # Cache queer resid_pre; control + queer baselines.
            _, queer_cache = model.run_with_cache(q_ids, names_filter=lambda n: n.endswith("hook_resid_pre"))
            control_logits = model(c_ids)
            control_cont_logp = float(continuation_logp(control_logits, c_ids_flat, cont_start)[0].item())
            queer_logits = model(q_ids)
            queer_cont_logp = float(continuation_logp(queer_logits, q_ids[0], len(aln["q_ids"]) - cont_count)[0].item())
            control_avg = control_cont_logp / cont_count
            queer_avg = queer_cont_logp / cont_count
            denom = queer_avg - control_avg

            tok_text_control = [tokenizer.decode([t]).replace("\n", "\\n") for t in aln["c_ids"]]
            tok_text_source = [tokenizer.decode([aln["q_ids"][source_pos[c]]]).replace("\n", "\\n") for c in range(len_c)]

            pair_rows: list[dict[str, Any]] = []

            def record(layer, pos, span, src, patched_avg):
                bias_effect = patched_avg - control_avg
                pair_rows.append({
                    "pair_id": pair_id, "row_id": row.get("row_id"),
                    "Gender_ID_x": row.get("Gender_ID_x"), "Gender_ID_y": row.get("Gender_ID_y"),
                    "predicate": row.get("predicate"),
                    "predicate_label_provisional": row.get("predicate_label_provisional"),
                    "layer": layer, "token_position": pos, "span": span, "source_position": src,
                    "is_identity_token": int(span == "identity"),
                    "token_text_control": tok_text_control[pos] if pos >= 0 else "",
                    "token_text_source": tok_text_source[pos] if pos >= 0 else "",
                    "control_cont_avg_logp": control_avg, "queer_cont_avg_logp": queer_avg,
                    "patched_cont_avg_logp": patched_avg, "bias_effect": bias_effect,
                    "normalized_restoration": safe_norm(bias_effect, denom),
                })

            for layer in range(n_layers):
                act_name = tl_utils.get_act_name("resid_pre", layer)
                qact = queer_cache[act_name]  # [1, len_q, d]

                # Per-position sweep (batched over control positions).
                for chunk_start in range(0, len_c, pbs):
                    chunk = list(range(chunk_start, min(chunk_start + pbs, len_c)))
                    ctrl_pos = torch.tensor(chunk, device=device)
                    src_pos = torch.tensor([source_pos[c] for c in chunk], device=device)
                    batched = c_ids.repeat(len(chunk), 1)

                    def hook(act, hook, ctrl_pos=ctrl_pos, src_pos=src_pos, qact=qact):
                        act = act.clone()
                        rows = torch.arange(act.shape[0], device=act.device)
                        act[rows, ctrl_pos, :] = qact[0, src_pos, :].to(act.dtype)
                        return act

                    patched_logits = model.run_with_hooks(batched, fwd_hooks=[(act_name, hook)])
                    sums = continuation_logp(patched_logits, c_ids_flat, cont_start)  # [b]
                    for i, c in enumerate(chunk):
                        record(layer, c, spans[c], source_pos[c], float(sums[i].item()) / cont_count)

                # Whole-identity-span patch (all identity control positions at once).
                id_ctrl = aln["identity_control_positions"]
                id_src = aln["identity_source_positions"]
                if id_ctrl:
                    ctrl_pos = torch.tensor(id_ctrl, device=device)
                    src_pos = torch.tensor(id_src, device=device)

                    def hook_all(act, hook, ctrl_pos=ctrl_pos, src_pos=src_pos, qact=qact):
                        act = act.clone()
                        act[0, ctrl_pos, :] = qact[0, src_pos, :].to(act.dtype)
                        return act

                    patched_logits = model.run_with_hooks(c_ids, fwd_hooks=[(act_name, hook_all)])
                    s = float(continuation_logp(patched_logits, c_ids_flat, cont_start)[0].item()) / cont_count
                    record(layer, -1, "identity_all", -1, s)

            writer.writerows(pair_rows)
            f.flush()
            del queer_cache
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        f.close()

    print(f"\nSkipped (not cleanly alignable): {skipped} / {len(pairs)}")
    return pd.read_csv(raw_out_path)


def aggregate_by_span(raw_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = raw_df[raw_df["span"].isin(SPAN_ORDER)].copy()
    df["span"] = pd.Categorical(df["span"], categories=SPAN_ORDER, ordered=True)
    return (
        df.groupby(["layer", "span"], as_index=False, observed=True)[value_col].mean()
        .rename(columns={value_col: "value"}).sort_values(["layer", "span"]).reset_index(drop=True)
    )


def robust_vlim(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return max(float(finite.abs().quantile(0.95)), 1e-6) if not finite.empty else 1.0


def plot_span_heatmap(agg: pd.DataFrame, png: Path, title: str) -> None:
    if agg.empty:
        return
    present = [s for s in SPAN_ORDER if s in set(agg["span"].astype(str))]
    pivot = agg.assign(span=agg["span"].astype(str)).pivot(index="layer", columns="span", values="value").reindex(columns=present).sort_index()
    vmax = robust_vlim(pd.Series(pivot.values.ravel()))
    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.2), 8))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title); ax.set_xlabel("prompt span"); ax.set_ylabel("layer")
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(int(y)) for y in pivot.index])
    fig.colorbar(im, ax=ax).set_label("mean bias_effect (queer→control, continuation avg logp)")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)


def plot_pair_heatmap(sub: pd.DataFrame, png: Path, title: str, trim_dead: bool = False) -> None:
    """Per-(layer, token) bias_effect for one pair. Shows the FULL control sentence so the
    whole data instance is visible. x-tick labels are color-coded by span:
      green/bold = identity slot (labeled control←injected_queer),
      blue       = continuation = the predicate (the readout being scored),
      grey       = the read-out position (last token before the predicate),
      black      = other scaffold.
    """
    sub = sub[sub["token_position"] >= 0].copy()
    if sub.empty:
        return
    if trim_dead:
        imp = sub.groupby("token_position")["bias_effect"].apply(lambda s: float(s.abs().max()))
        gmax = max(float(imp.max()), 1e-9)
        keep = set(imp[imp >= 0.02 * gmax].index.astype(int)) | set(sub.loc[sub["is_identity_token"] == 1, "token_position"].astype(int))
        sub = sub[sub["token_position"].isin(keep)]
    pivot = sub.pivot_table(index="layer", columns="token_position", values="bias_effect").sort_index(axis=0).sort_index(axis=1)
    meta = sub.drop_duplicates("token_position").set_index("token_position")
    cols = [int(c) for c in pivot.columns]
    id_pos = set(sub.loc[sub["is_identity_token"] == 1, "token_position"].astype(int))
    span_of = {c: str(meta.loc[c, "span"]) for c in cols}
    # The read-out position = the last token BEFORE the first continuation token.
    cont_positions = sorted(c for c in cols if span_of[c] == "continuation")
    readout_pos = (cont_positions[0] - 1) if cont_positions else None
    # Full queer identity name (the injected entity) — clearer than the literal last sub-token.
    queer_id = str(sub["Gender_ID_x"].iloc[0]).strip()

    def lab(c: int) -> str:
        ct = str(meta.loc[c, "token_text_control"]).strip()
        if c in id_pos:  # control identity slot, filled with the queer identity's state
            return f"{ct}←{queer_id}"
        return ct

    vmax = robust_vlim(pd.Series(pivot.values.ravel()))
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.52), 8))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("layer")
    ax.set_xlabel("full control sentence — green=identity (control←injected queer), blue=predicate (scored), grey=read-out position")
    ax.set_xticks(range(len(cols)))
    ticks = ax.set_xticklabels([lab(c) for c in cols], rotation=55, ha="right", fontsize=8)
    for t, c in zip(ticks, cols):
        if c in id_pos:
            t.set_color("#00a000"); t.set_fontweight("bold")
        elif span_of[c] == "continuation":
            t.set_color("#1f77b4"); t.set_fontweight("bold")
        elif c == readout_pos:
            t.set_color("#777777")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([str(int(y)) for y in pivot.index])
    id_cols = sorted(i for i, c in enumerate(cols) if c in id_pos)
    if id_cols:
        ax.add_patch(Rectangle((id_cols[0] - 0.5, -0.5), (id_cols[-1] - id_cols[0]) + 1.0, len(pivot.index),
                               fill=False, edgecolor="#00b050", linewidth=2.5))
    fig.colorbar(im, ax=ax).set_label("bias_effect = Δ logP(predicate) from injecting queer state at this token/layer")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)


def make_outputs(raw_df: pd.DataFrame, out_dir: Path, num_pair_plots: int, trim_dead: bool = False):
    """Aggregates + plots from a raw patching CSV (used by both full runs and --plot_only)."""
    span_csv = out_dir / "resid_pre_span_heatmap.csv"
    span_png = out_dir / "resid_pre_span_heatmap.png"
    agg = aggregate_by_span(raw_df, "bias_effect")
    agg.rename(columns={"value": "mean_bias_effect"}).to_csv(span_csv, index=False)
    plot_span_heatmap(agg, span_png, "WinoQueer resid_pre patching (queer→control) by span — ALL predicates")
    extra: list[Path] = []

    # Per predicate-label span heatmaps (compare how stereotype categories localize).
    if "predicate_label_provisional" in raw_df.columns:
        per_label_dir = out_dir / "per_label_span"
        per_label_dir.mkdir(parents=True, exist_ok=True)
        for label, g in raw_df.groupby("predicate_label_provisional"):
            lab = str(label).strip()
            if not lab or lab.lower() == "nan":
                continue
            safe = "".join(ch if ch.isalnum() else "_" for ch in lab)[:48]
            a = aggregate_by_span(g, "bias_effect")
            a.rename(columns={"value": "mean_bias_effect"}).to_csv(per_label_dir / f"span_{safe}.csv", index=False)
            plot_span_heatmap(a, per_label_dir / f"span_{safe}.png",
                              f"{lab}  (n={g['pair_id'].nunique()} pairs) — resid_pre by span")
            extra += [per_label_dir / f"span_{safe}.csv", per_label_dir / f"span_{safe}.png"]

    id_all = raw_df[raw_df["span"] == "identity_all"].groupby("layer", as_index=False)["bias_effect"].mean()
    id_all_csv = out_dir / "resid_pre_identity_span_by_layer.csv"
    id_all.rename(columns={"bias_effect": "mean_bias_effect"}).to_csv(id_all_csv, index=False)

    # Per-pair plots: ONE representative (highest-bias) pair per DISTINCT predicate, so the
    # spot-checks span many stereotypes instead of repeating the single top predicate.
    per_pair_dir = out_dir / "per_pair"
    per_pair_dir.mkdir(parents=True, exist_ok=True)
    first = raw_df.drop_duplicates("pair_id").set_index("pair_id")
    seen: set[str] = set()
    chosen: list[int] = []
    for pid in sorted(raw_df["pair_id"].unique()):
        pred = str(first.loc[pid, "predicate"])
        if pred in seen:
            continue
        seen.add(pred)
        chosen.append(int(pid))
        if len(chosen) >= num_pair_plots:
            break
    plot_paths = []
    for pid in chosen:
        sub = raw_df[raw_df["pair_id"] == pid]
        m = sub.iloc[0]
        safe_pred = "".join(ch if ch.isalnum() else "_" for ch in str(m["predicate"]))[:24]
        out = per_pair_dir / f"pair{int(pid)}_{m['Gender_ID_x']}_to_{m['Gender_ID_y']}_{safe_pred}.png"
        title = f"inject {m['Gender_ID_x']} state into '{m['Gender_ID_y']}' prompt  →  Δ logP(predicate={m['predicate']!r})"
        plot_pair_heatmap(sub, out, title, trim_dead=trim_dead)
        plot_paths.append(out)
    return [span_csv, span_png, id_all_csv] + extra + plot_paths, agg, id_all


def main() -> None:
    parser = argparse.ArgumentParser(description="WinoQueer per-token per-layer resid_pre patching (queer→control).")
    parser.add_argument("--pairs_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip the model; rebuild aggregates/plots from an existing raw CSV in --out_dir.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_per_predicate", type=int, default=None,
                        help="Cap rows per predicate (highest bias_score kept) for diversity "
                        "across predicates. Applied before --max_pairs.")
    parser.add_argument("--patch_batch_size", type=int, default=32)
    parser.add_argument("--num_pair_plots", type=int, default=10)
    parser.add_argument("--trim_dead", action="store_true",
                        help="Trim structurally-zero scaffold columns from per-pair plots. Default: show the full sentence.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_resort", action="store_true",
                        help="Consume --pairs_csv in file order (no bias_score re-sort / per-predicate cap). "
                             "Use with a pre-frozen cohort so pair_id == cohort row order.")
    args = parser.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "resid_pre_patching_raw.csv"

    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"--plot_only needs an existing raw CSV at {raw_path}")
        raw_df = pd.read_csv(raw_path)
        print(f"plot_only: {raw_df['pair_id'].nunique()} pairs, {raw_df['predicate'].nunique()} predicates from {raw_path}")
    else:
        if args.pairs_csv is None:
            parser.error("--pairs_csv is required unless --plot_only is set")
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
            n_pred = pairs["predicate"].nunique() if "predicate" in pairs.columns else "?"
            print(f"Pairs: {len(pairs)} across {n_pred} predicates "
                  f"(max_per_predicate={args.max_per_predicate}, max_pairs={args.max_pairs})")
        raw_df = run_patching(args, pairs, raw_path)

    out_paths, agg, id_all = make_outputs(raw_df, args.out_dir, args.num_pair_plots, trim_dead=args.trim_dead)

    print("\nWrote:")
    for p in [raw_path] + out_paths:
        print(f"  {p}")
    print("\nMean bias_effect by span (averaged over layers):")
    print(agg.groupby("span", observed=True)["value"].mean().reindex(SPAN_ORDER).to_string())
    print("\nIdentity-span-all bias_effect by layer (injecting the whole queer identity):")
    print(id_all.rename(columns={"bias_effect": "mean_bias_effect"}).to_string(index=False))
    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
