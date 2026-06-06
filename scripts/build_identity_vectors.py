#!/usr/bin/env python3
"""Estimate IDENTITY-only directions (v_identity) from the identity-mention dataset.

Motivation: v_bias (run_winoqueer_steering_sweep.py --save_vectors) entangles two things — the
representation of the IDENTITY itself and its STEREOTYPE association. The identity-only dataset
(data/mi_identity_prompts.csv) mentions each identity WITHOUT any stereotype continuation, so a
contrast over it isolates the identity direction. Subtracting it from v_bias (residualize_vectors.py)
gives a rough v_stereotype = v_bias - v_identity.

What it does, in ONE model pass:
  1. For every prompt, locate the identity token span via the template's {form} placeholder
     (prefix/suffix split -> char span -> token span via the fast tokenizer's offset mapping) and
     VALIDATE the alignment (recovered tokens must cover form_used). Misalignments are counted and
     skipped, never silently mis-read.
  2. Read resid_pre[L] under THREE conventions, accumulating per-identity centroids:
       - identity (PRIMARY): mean over the identity token span — the cleanest identity signal.
       - last: the last real token (note: usually the sentence period — kept only for comparison).
       - mean: mean over all real tokens.
  3. Dump per-identity centroids (all identities in scope) for reuse, and build contrast vectors
     v_identity[L] = mean(target centroids) - mean(reference centroids) per configured axis, saved
     in the SAME .pt schema as v_bias so they're directly subtractable AND loadable by
     run_bbq_steering_transfer.py.

Contrast defaults match the WinoQueer-aligned axes; override with --target/--reference for one axis.

Run (POD/GPU): see scripts/pod_identity_vectors.sh.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify_biased_bbq_items import resolve_device  # noqa: E402
from run_bbq_steering_transfer import load_model       # noqa: E402
from transformer_lens import utils as tl_utils         # noqa: E402

# Axis -> (target labels, reference labels). canonical_label values from mi_identity_prompts.csv.
# SO target = the WinoQueer identity set (gay/lesbian/bi/pan/ace) vs the cis-het reference; gender
# target = trans/NB vs cis. Kept matched to v_bias's axes so the subtraction is apples-to-apples.
DEFAULT_CONTRASTS = {
    "sexual_orientation": (
        ["gay", "lesbian", "bisexual", "pansexual", "asexual"],
        ["straight", "heterosexual"],
    ),
    "gender_identity": (
        ["transgender", "transgender man", "transgender woman", "nonbinary"],
        ["cisgender", "cisgender man", "cisgender woman"],
    ),
}
VARIANTS = ("identity", "last", "mean")


def char_span(template_text: str, prompt: str) -> tuple[int, int]:
    """Char [start, end) of the substituted {form} in the prompt (prefix/suffix are literal)."""
    i = template_text.index("{form}")
    pre, suf = template_text[:i], template_text[i + len("{form}"):]
    return len(pre), len(prompt) - len(suf)


def identity_token_idx(offsets, cs: int, ce: int) -> list[int]:
    """Token indices whose char offsets overlap [cs, ce); drops zero-length special/pad tokens."""
    return [j for j, (a, b) in enumerate(offsets) if b > a and a < ce and b > cs]


@torch.no_grad()
def collect_centroids(model, tokenizer, df, device, batch_size):
    """One pass over the prompts -> per-(label, variant) summed readout vectors + counts.

    Returns: sums[label][variant] -> Tensor[n_layers, d_model] (float32), counts[label] -> int,
    and a diagnostics dict (n_misaligned, examples).
    """
    n_layers, d_model = int(model.cfg.n_layers), int(model.cfg.d_model)
    names = [tl_utils.get_act_name("resid_pre", L) for L in range(n_layers)]
    nf = lambda n: n.endswith("hook_resid_pre")

    rows = df.to_dict("records")
    sums: dict[str, dict[str, torch.Tensor]] = {}
    counts: dict[str, int] = {}
    misaligned = 0
    mis_examples: list[str] = []

    def acc(label, variant, vec):
        d = sums.setdefault(label, {v: torch.zeros(n_layers, d_model, dtype=torch.float32) for v in VARIANTS})
        d[variant] += vec

    for s in tqdm(range(0, len(rows), batch_size), desc="Identity readouts"):
        batch = rows[s:s + batch_size]
        prompts = [r["prompt"] for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True,
                        return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        tokens = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        # Resolve identity token spans up front; flag any prompt whose alignment fails.
        spans = []
        for bi, r in enumerate(batch):
            cs, ce = char_span(r["template_text"], r["prompt"])
            idx = identity_token_idx(offsets[bi].tolist(), cs, ce)
            recovered = r["prompt"][cs:ce].strip().lower()
            if not idx or recovered != str(r["form_used"]).strip().lower():
                spans.append(None)
            else:
                spans.append(idx)

        _, cache = model.run_with_cache(tokens, names_filter=nf)
        last_pos = (attn.sum(dim=1) - 1)
        real = attn.bool()

        for bi, r in enumerate(batch):
            label = r["canonical_label"]
            idx = spans[bi]
            if idx is None:
                misaligned += 1
                if len(mis_examples) < 12:
                    mis_examples.append(f"{r['prompt_id']}: {r['prompt']!r} (form={r['form_used']!r})")
                continue
            idx_t = torch.tensor(idx, device=device)
            lp = int(last_pos[bi].item())
            rmask = real[bi]
            id_vec = torch.empty(n_layers, d_model, dtype=torch.float32)
            last_vec = torch.empty(n_layers, d_model, dtype=torch.float32)
            mean_vec = torch.empty(n_layers, d_model, dtype=torch.float32)
            for L, name in enumerate(names):
                act = cache[name][bi]                       # [seq, d_model]
                id_vec[L] = act[idx_t].mean(0).float().cpu()
                last_vec[L] = act[lp].float().cpu()
                mean_vec[L] = act[rmask].mean(0).float().cpu()
            acc(label, "identity", id_vec)
            acc(label, "last", last_vec)
            acc(label, "mean", mean_vec)
            counts[label] = counts.get(label, 0) + 1
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

    centroids = {lab: {v: sums[lab][v] / counts[lab] for v in VARIANTS} for lab in counts}
    diag = {"n_misaligned": misaligned, "examples": mis_examples,
            "n_layers": n_layers, "d_model": d_model}
    return centroids, counts, diag


def contrast_vector(centroids, target, reference, variant):
    """v[L] = mean over target labels of centroid - mean over reference labels of centroid."""
    miss = [l for l in target + reference if l not in centroids]
    if miss:
        raise SystemExit(f"Labels not found in identity data: {miss}\nHave: {sorted(centroids)}")
    tgt = torch.stack([centroids[l][variant] for l in target]).mean(0)
    ref = torch.stack([centroids[l][variant] for l in reference]).mean(0)
    return tgt - ref


def save_contrast(path, centroids, counts, axis, target, reference, diag, tl_model_name, src):
    """Write a v_bias-compatible .pt: top-level vectors/norms = PRIMARY (identity) variant;
    last/mean variants retained under 'variants' for position-matched comparisons later."""
    vecs = {v: contrast_vector(centroids, target, reference, v) for v in VARIANTS}
    primary = vecs["identity"]
    n_layers = primary.shape[0]
    blob = {
        "vectors": primary,                              # [n_layers, d_model] float32  (PRIMARY)
        "norms": primary.norm(dim=1),                    # [n_layers]
        "vector_position": "identity",                   # readout convention of the PRIMARY vector
        "variants": vecs,                                # {identity,last,mean} -> [n_layers, d_model]
        "tl_model_name": tl_model_name,
        "n_layers": int(n_layers),
        "n_train_pairs": int(sum(counts[l] for l in target + reference)),  # name kept for schema compat
        "train_identity_counts": {l: counts[l] for l in target + reference},
        "axis": axis,
        "target_labels": target,
        "reference_labels": reference,
        "source": "identity_only",
        "pairs_csv": str(src),
        "n_misaligned": diag["n_misaligned"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    nm = blob["norms"]
    print(f"  saved {path}  | primary(identity) per-layer norm [{nm.min():.2f}, {nm.max():.2f}] "
          f"| target n={sum(counts[l] for l in target)} ref n={sum(counts[l] for l in reference)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build identity-only direction vectors (v_identity).")
    ap.add_argument("--identity_csv", type=Path, default=Path("data/mi_identity_prompts.csv"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--axes", type=str, default="sexual_orientation,gender_identity",
                    help="comma list, or 'all' to compute centroids for every axis in the CSV")
    ap.add_argument("--target", type=str, default=None,
                    help="override target labels (comma) — only valid with a single --axes")
    ap.add_argument("--reference", type=str, default=None, help="override reference labels (comma)")
    ap.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_per_label", type=int, default=None, help="cap prompts/identity (debug)")
    args = ap.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    df = pd.read_csv(args.identity_csv)
    axes = sorted(df["axis"].unique()) if args.axes == "all" else [a.strip() for a in args.axes.split(",")]
    df = df[df["axis"].isin(axes)].reset_index(drop=True)
    if args.max_per_label is not None:
        df = df.groupby("canonical_label", sort=False, group_keys=False).head(args.max_per_label)
    print(f"Identity prompts: {len(df)} across axes {axes} "
          f"({df['canonical_label'].nunique()} identities)")

    model, tokenizer = load_model(args.model_path, args.tl_model_name, device, dtype)
    centroids, counts, diag = collect_centroids(model, tokenizer, df, device, args.batch_size)

    print(f"\nAlignment: {diag['n_misaligned']} misaligned / {len(df)} prompts "
          f"({100*diag['n_misaligned']/max(len(df),1):.2f}%)")
    for ex in diag["examples"]:
        print(f"   MISALIGNED {ex}")

    # Per-identity centroids (all variants) — reusable for any future contrast without re-running.
    cpath = args.out_dir / "identity_centroids.pt"
    torch.save({"centroids": centroids, "counts": counts, "variants": list(VARIANTS),
                "axes": axes, "tl_model_name": args.tl_model_name,
                "n_layers": diag["n_layers"], "d_model": diag["d_model"],
                "identity_csv": str(args.identity_csv)}, cpath)
    print(f"\nSaved per-identity centroids ({len(counts)} identities) -> {cpath}")

    # Contrast vectors.
    if args.target or args.reference:
        if len(axes) != 1:
            raise SystemExit("--target/--reference override requires exactly one --axes")
        contrasts = {axes[0]: ([t.strip() for t in args.target.split(",")],
                               [r.strip() for r in args.reference.split(",")])}
    else:
        contrasts = {a: DEFAULT_CONTRASTS[a] for a in axes if a in DEFAULT_CONTRASTS}
        skipped = [a for a in axes if a not in DEFAULT_CONTRASTS]
        if skipped:
            print(f"No default contrast for {skipped} — centroids saved; pass --target/--reference "
                  f"to build a vector for one of these.")

    print("\nContrast vectors (v_identity):")
    for axis, (tgt, ref) in contrasts.items():
        save_contrast(args.out_dir / f"v_identity_{axis}.pt", centroids, counts, axis, tgt, ref,
                      diag, args.tl_model_name, args.identity_csv)

    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
