#!/usr/bin/env python3
"""WinoQueer activation-steering sweep — turn the bias circuit into a single steerable direction.

The patching/ablation work localized the bias to specific components. This asks the complementary
question: is there a single fixed *bias direction* in the residual stream that linearly controls the
stereotype — one we can ADD to neutralize-context prompts to induce bias, or SUBTRACT from queer
prompts to de-bias? A direction that generalizes to held-out pairs is evidence of a linear
"queerness / bias" feature, and the best steering layer should line up with the resid-patching
layer profile.

Steering vector (difference-of-means, per layer), built on a TRAIN split:
    v_L = mean over train pairs of [ resid_pre_L(queer)[readout] - resid_pre_L(control)[readout] ]
The readout token is the position whose next-token IS the first continuation token (end-aligned, and
the same token in both prompts since it lies in the shared suffix), so v_L isolates the queer-vs-
control state at the decision point — exactly the quantity run_winoqueer_resid_patching.py patches.

Sweep (evaluated on a HELD-OUT test split):
    layer L  x  coefficient alpha   (raw units of v_L: alpha=1 ~ inject one full avg difference,
                                     alpha<0 = subtract = de-bias)
    inject alpha * v_L at the readout + continuation positions of:
      - CONTROL prompts  -> "induce": does +alpha raise the stereotype toward the queer level?
      - QUEER prompts    -> "remove": does -alpha lower it toward the control level?
    metric: bias_fraction = (steered_cont_avg_logp - control_base) / (queer_base - control_base)
            (0 = control-like / unbiased, 1 = queer-like / fully biased), per pair's own baselines.

Rigor controls:
  - RANDOM direction (norm-matched to v_L, per layer): must NOT systematically move bias -> proves
    the effect is the DIRECTION, not the injected norm.
  - FLUENCY guard: KL(steered_next_token || unsteered_next_token) at the readout position, so we can
    find the de-biasing sweet spot before large alpha wrecks the distribution.

Outputs: long raw CSV + layer x alpha heatmaps (induce / remove), best-layer steering curves
(real vs random), the steering frontier (bias change vs KL fluency cost), and a cross-check of the
best steering layer against the resid-patching layer profile (if provided).
"""
from __future__ import annotations

import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_winoqueer_head_ablation import load_model, prep_pair  # noqa: E402
from run_winoqueer_resid_patching import continuation_logp  # noqa: E402


RAW_HEADER = [
    "row_id", "pair_id", "eval_set", "kind", "rand_seed", "layer", "alpha",
    "Gender_ID_x", "Gender_ID_y", "predicate", "predicate_label_provisional",
    "control_base", "queer_base", "steered_cont_avg_logp", "bias_fraction", "kl_readout",
]


def readout_and_cont(p, eval_set):
    """Injection positions for one prompt: the readout token + every continuation token."""
    if eval_set == "control":
        ids, cont_start, length = p["c_ids"], p["c_cont_start"], p["len_c"]
    else:
        ids, cont_start, length = p["q_ids"], p["q_cont_start"], p["len_q"]
    readout = cont_start - 1
    pos = torch.arange(readout, length, device=ids.device)  # readout + continuation
    return ids, cont_start, p["cont_count"], pos, readout


@torch.no_grad()
def build_vectors(model, tokenizer, device, train_pairs, position, n_layers):
    """Difference-of-means steering vector per layer from the train split."""
    from transformer_lens import utils as tl_utils
    names = [tl_utils.get_act_name("resid_pre", L) for L in range(n_layers)]
    nf = lambda n: n.endswith("hook_resid_pre")
    acc = torch.zeros(n_layers, model.cfg.d_model, dtype=torch.float32)
    used = 0
    for _, row in tqdm(train_pairs.iterrows(), total=len(train_pairs), desc="Build vectors"):
        p = prep_pair(tokenizer, row, device)
        if p is None:
            continue
        _, qc = model.run_with_cache(p["q_ids"], names_filter=nf)
        _, cc = model.run_with_cache(p["c_ids"], names_filter=nf)
        q_read, c_read = p["q_cont_start"] - 1, p["c_cont_start"] - 1
        for L, name in enumerate(names):
            if position == "identity":
                qv = qc[name][0, p["aln"]["P"]:p["aln"]["P"] + p["aln"]["Lx"]].mean(0)
                cpos = p["aln"]["identity_control_positions"]
                cv = cc[name][0, cpos].mean(0)
            elif position == "mean":
                qv, cv = qc[name][0].mean(0), cc[name][0].mean(0)
            else:  # readout (default)
                qv, cv = qc[name][0, q_read], cc[name][0, c_read]
            acc[L] += (qv - cv).float().cpu()
        used += 1
        del qc, cc
    vectors = acc / max(used, 1)
    norms = vectors.norm(dim=1)
    print(f"Built vectors from {used} train pairs | per-layer norm range "
          f"[{norms.min():.2f}, {norms.max():.2f}]")
    return vectors, norms


def random_matched(vectors, norms, seed_offset=0):
    """Random directions, norm-matched per layer (deterministic without global RNG)."""
    g = torch.Generator().manual_seed(1234 + seed_offset)
    r = torch.randn(vectors.shape, generator=g)
    r = r / r.norm(dim=1, keepdim=True) * norms.unsqueeze(1)
    return r


@torch.no_grad()
def eval_pair(model, p, eval_set, vectors, rand_vectors_list, alphas, layers, kinds, device):
    """For one pair: sweep (kind, layer, alpha). One batched forward per (kind, seed, layer).

    The "random" control averages over rand_vectors_list (several norm-matched random directions);
    a single random draw could coincidentally align with the bias direction, so we use several and
    let the downstream median pool over (pairs × seeds)."""
    from transformer_lens import utils as tl_utils
    ids, cont_start, cont_count, pos, readout = readout_and_cont(p, eval_set)
    # per-pair baselines (both prompts, regardless of which we steer)
    control_base = float(continuation_logp(model(p["c_ids"]), p["c_ids"][0], p["c_cont_start"])[0].item()) / p["cont_count"]
    queer_base = float(continuation_logp(model(p["q_ids"]), p["q_ids"][0], p["q_cont_start"])[0].item()) / p["cont_count"]
    denom = queer_base - control_base
    if abs(denom) < 1e-6:
        return []

    a_t = torch.tensor(alphas, device=device)
    i0 = int(np.argmin(np.abs(np.array(alphas))))  # index of alpha closest to 0 (reference)
    B = len(alphas)
    out = []
    for kind in kinds:
        srcs = [vectors] if kind == "real" else rand_vectors_list
        for si, src in enumerate(srcs):
            rseed = -1 if kind == "real" else si
            for L in layers:
                name = tl_utils.get_act_name("resid_pre", L)
                vec = src[L].to(device)

                def hook(act, hook, pos=pos, vec=vec, a_t=a_t):
                    act = act.clone()
                    act[:, pos, :] = act[:, pos, :] + a_t.view(-1, 1, 1).to(act.dtype) * vec.to(act.dtype)
                    return act

                logits = model.run_with_hooks(ids.repeat(B, 1), fwd_hooks=[(name, hook)])
                sums = continuation_logp(logits, ids[0], cont_start) / cont_count  # [B]
                # KL(steered || unsteered) at the readout position
                lp = torch.log_softmax(logits[:, readout, :].float(), dim=-1)  # [B,V]
                ref = lp[i0:i0 + 1]
                kl = (lp.exp() * (lp - ref)).sum(-1)  # [B]
                for j, a in enumerate(alphas):
                    steered = float(sums[j].item())
                    out.append({
                        "row_id": p_rowid(p), "pair_id": None, "eval_set": eval_set, "kind": kind,
                        "rand_seed": rseed, "layer": L, "alpha": a,
                        "Gender_ID_x": p["meta"]["Gender_ID_x"], "Gender_ID_y": p["meta"]["Gender_ID_y"],
                        "predicate": p["meta"]["predicate"], "predicate_label_provisional": p["meta"]["predicate_label_provisional"],
                        "control_base": control_base, "queer_base": queer_base,
                        "steered_cont_avg_logp": steered,
                        "bias_fraction": (steered - control_base) / denom,
                        "kl_readout": float(kl[j].item()),
                    })
    return out


def p_rowid(p):
    return p["meta"].get("row_id")


def attach_meta(prep, row):
    prep["meta"] = {k: row.get(k) for k in
                    ["row_id", "Gender_ID_x", "Gender_ID_y", "predicate", "predicate_label_provisional"]}
    return prep


def stratified_split(pairs, train_frac, seed=0):
    """Predicate-stratified train/test split, shuffled within each predicate group (seeded).

    Pairs arrive sorted by bias_score upstream; without the shuffle the first `cut` rows of each
    group (= the highest-bias pairs) would all go to train and the lower-bias ones to test, a
    systematic distribution shift. Shuffling within the group removes that while staying
    deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, g in pairs.groupby("predicate", sort=True):
        idx = list(g.index)
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_frac))
        if len(idx) >= 2 and 0.0 < train_frac < 1.0:
            cut = min(max(cut, 1), len(idx) - 1)  # both splits get >=1 when the group allows
        train_idx += idx[:cut]
        test_idx += idx[cut:]
    return pairs.loc[train_idx], pairs.loc[test_idx]


# ----------------------------------------------------------------------------- plots
def heatmap(piv, png, title, cbar, center1=False):
    fig, ax = plt.subplots(figsize=(max(8, piv.shape[1] * 0.7), max(7, piv.shape[0] * 0.28)))
    if center1:
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
    else:
        v = max(float(np.nanmax(np.abs(piv.values))), 1e-9)
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(title, fontsize=12); ax.set_xlabel("alpha (coefficient on v_L)"); ax.set_ylabel("layer")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([str(int(i)) for i in piv.index], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(cbar)
    fig.tight_layout(); fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)


def make_outputs(raw: pd.DataFrame, out_dir: Path, patch_profile_csv: Path | None, kl_budget: float = 0.5):
    paths = []
    real = raw[raw.kind == "real"]
    # 1. layer x alpha heatmaps of bias_fraction, per eval_set. We aggregate across pairs with
    #    the MEDIAN: bias_fraction = effect / (queer_base - control_base) is a per-pair ratio,
    #    so a pair with a tiny baseline gap can blow up its ratio and dominate a mean; the
    #    median is robust to that heavy tail. (KL stays a mean — it has no small denominator.)
    best = {}
    for es in ["control", "queer"]:
        sub = real[real.eval_set == es]
        if sub.empty:
            continue
        piv = sub.groupby(["layer", "alpha"])["bias_fraction"].median().unstack("alpha").sort_index()
        piv.to_csv(out_dir / f"steering_layer_alpha_{es}.csv")
        title = ("INDUCE: add +alpha*v to CONTROL prompts (1=fully biased)" if es == "control"
                 else "REMOVE: add alpha*v to QUEER prompts (alpha<0 de-biases)")
        heatmap(piv, out_dir / f"steering_layer_alpha_{es}.png", f"WinoQueer steering — {title}",
                "median bias_fraction", center1=True)
        paths += [out_dir / f"steering_layer_alpha_{es}.csv", out_dir / f"steering_layer_alpha_{es}.png"]
        # best layer for this regime — closest to target AMONG fluency-safe points (KL<=budget)
        if es == "control":   # induce toward 1 via alpha>0
            cand, target = sub[sub.alpha > 0], 1.0
        else:                 # de-bias toward 0 via alpha<0
            cand, target = sub[sub.alpha < 0], 0.0
        g = cand.groupby(["layer", "alpha"]).agg(bias=("bias_fraction", "median"), kl=("kl_readout", "mean"))
        safe = g[g["kl"] <= kl_budget]
        pick = safe if not safe.empty else g  # fall back if nothing under budget
        if not pick.empty:
            best[es] = pick["bias"].sub(target).abs().idxmin()  # (layer, alpha)

    # 2. best-layer steering curves (real vs random), per eval_set
    for es in ["control", "queer"]:
        sub = raw[raw.eval_set == es]
        if sub.empty or es not in best:
            continue
        bl = int(best[es][0])
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for kind, color in [("real", "#c0392b"), ("random", "#888888")]:
            grp = sub[(sub.kind == kind) & (sub.layer == bl)].groupby("alpha")["bias_fraction"]
            med = grp.median()
            if med.empty:
                continue
            q1, q3 = grp.quantile(0.25), grp.quantile(0.75)  # robust IQR band (matches median)
            ax.plot(med.index, med.values, "-o", color=color, lw=2, ms=4, label=f"{kind} v")
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.2)
        ax.axhline(0, color="#2c7fb8", ls=":", lw=1, label="control baseline (unbiased)")
        ax.axhline(1, color="#c0392b", ls=":", lw=1, label="queer baseline (biased)")
        ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("alpha"); ax.set_ylabel("bias_fraction"); ax.set_title(f"Steering curve @ best layer L{bl} — eval on {es} prompts")
        ax.legend(fontsize=8)
        p = out_dir / f"steering_curve_{es}_L{bl}.png"
        fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 3. steering frontier: bias change vs KL fluency cost (real, all layers)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    fr = real.groupby(["eval_set", "layer", "alpha"]).agg(bias=("bias_fraction", "median"), kl=("kl_readout", "mean")).reset_index()
    for es, marker in [("control", "o"), ("queer", "s")]:
        s = fr[fr.eval_set == es]
        sc = ax.scatter(s["kl"], s["bias"], c=s["layer"], cmap="viridis", s=22, marker=marker, alpha=0.8, label=f"eval={es}")
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("layer")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.axhline(0, color="#2c7fb8", ls=":", lw=1); ax.axhline(1, color="#c0392b", ls=":", lw=1)
    ax.set_xlabel("KL(steered || unsteered) at readout  (fluency cost →)")
    ax.set_ylabel("median bias_fraction")
    ax.set_title("Steering frontier — bias control vs distribution distortion\n(want big bias move at low KL)")
    ax.legend(fontsize=8)
    p = out_dir / "steering_frontier.png"
    fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 4. cross-check best steering layer vs resid-patching layer profile
    if patch_profile_csv and Path(patch_profile_csv).exists():
        prof = pd.read_csv(patch_profile_csv)
        col = next((c for c in ["mean_bias_effect", "suff_pos_mass", "value"] if c in prof.columns), None)
        if col and "layer" in prof.columns:
            pl = prof.groupby("layer")[col].mean()
            steer = real[real.eval_set == "control"]
            sl = steer[steer.alpha > 0].groupby("layer")["bias_fraction"].max()
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(sl.index, sl.values, "-o", color="#c0392b", lw=2, ms=4, label="max induce bias_fraction (steering)")
            ax1.set_xlabel("layer"); ax1.set_ylabel("steering induce strength", color="#c0392b")
            ax2 = ax1.twinx()
            ax2.plot(pl.index, pl.values, "-s", color="#2c7fb8", lw=2, ms=4, label="resid-patching effect")
            ax2.set_ylabel("patching layer profile", color="#2c7fb8")
            ax1.set_title("Does the best steering layer match where patching localized the bias?")
            p = out_dir / "steering_vs_patching_layer.png"
            fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths, best


def main() -> None:
    ap = argparse.ArgumentParser(description="WinoQueer activation-steering sweep (difference-of-means).")
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--max_pairs", type=int, default=400, help="total pairs (split into train/test)")
    ap.add_argument("--max_per_predicate", type=int, default=None)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--vector_position", choices=["readout", "identity", "mean"], default="readout")
    ap.add_argument("--alphas", type=str, default="-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3")
    ap.add_argument("--layers", type=str, default=None, help="comma list; default = all layers")
    ap.add_argument("--eval_sets", type=str, default="control,queer")
    ap.add_argument("--no_random", action="store_true", help="skip the norm-matched random control")
    ap.add_argument("--n_random_seeds", type=int, default=5,
                    help="number of norm-matched random directions to average for the control")
    ap.add_argument("--kl_budget", type=float, default=0.5,
                    help="max readout KL for a steering point to count as fluency-safe when picking the best operating point")
    ap.add_argument("--patch_profile_csv", type=Path, default=None,
                    help="resid-patching layer profile CSV for the layer cross-check plot")
    ap.add_argument("--save_vectors", type=Path, default=None,
                    help="persist the learned per-layer difference-of-means vectors (.pt) so they "
                         "can be applied OOD (e.g. to the BBQ QA task by run_bbq_steering_transfer.py)")
    ap.add_argument("--load_vectors", type=Path, default=None,
                    help="evaluate an EXTERNAL saved vector (.pt) instead of building one from the "
                         "cohort — applies v_identity / v_stereotype / a foreign v_bias to this "
                         "continuation task. Skips the train split (evaluates on all matched pairs).")
    ap.add_argument("--axis", type=str, default=None,
                    help="filter the cohort to one axis (e.g. sexual_orientation) so the vector is "
                         "applied only to matching pairs, not blindly across axes.")
    ap.add_argument("--identity", type=str, default=None,
                    help="comma list of identities to filter to (matched application).")
    ap.add_argument("--identity_col", type=str, default=None,
                    help="column the --identity filter keys on; default auto: `identity` (WinoQueer) "
                         "else `block` (combined BBQ+CrowS cohort).")
    ap.add_argument("--vectors_only", action="store_true",
                    help="build + --save_vectors from all matched pairs, then exit (skip the alpha "
                         "sweep). Cheap way to mint per-identity v_bias.")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "steering_sweep_raw.csv"

    if args.plot_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"--plot_only needs {raw_path}")
        raw = pd.read_csv(raw_path)
    else:
        alphas = [float(x) for x in args.alphas.split(",")]
        eval_sets = [s.strip() for s in args.eval_sets.split(",")]
        kinds = ["real"] if args.no_random else ["real", "random"]

        pairs = pd.read_csv(args.pairs_csv).sort_values("bias_score", ascending=False)
        # Identity-MATCHED application: restrict to the axis/identity the vector is meant for so we
        # don't blindly steer unrelated pairs (e.g. a sexual_orientation vector on race pairs).
        if args.axis is not None and "axis" in pairs.columns:
            pairs = pairs[pairs["axis"].astype(str) == args.axis]
        if args.identity is not None:
            idcol = args.identity_col or ("identity" if "identity" in pairs.columns else "block")
            if idcol not in pairs.columns:
                raise SystemExit(f"--identity given but column {idcol!r} not in cohort {list(pairs.columns)}")
            keep = {s.strip() for s in args.identity.split(",")}
            pairs = pairs[pairs[idcol].astype(str).isin(keep)]
        if args.max_per_predicate is not None and "predicate" in pairs.columns:
            pairs = pairs.groupby("predicate", sort=False, group_keys=False).head(args.max_per_predicate).sort_values("bias_score", ascending=False)
        pairs = pairs.head(args.max_pairs).reset_index(drop=True)
        if len(pairs) == 0:
            raise SystemExit(f"No pairs after filtering (axis={args.axis}, identity={args.identity}).")

        model, tokenizer, device = load_model(args)
        n_layers = int(model.cfg.n_layers)
        layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))

        if args.load_vectors is not None:
            # Evaluate an external vector: no train split, no build, no save — score every matched pair.
            blob = torch.load(args.load_vectors, map_location="cpu")
            vectors, norms = blob["vectors"].float(), blob["norms"].float()
            if vectors.shape[0] != n_layers:
                raise SystemExit(f"vector n_layers {vectors.shape[0]} != model n_layers {n_layers}")
            train_pairs, test_pairs = pairs.iloc[0:0], pairs
            print(f"LOADED vector {args.load_vectors} (pos={blob.get('vector_position')}, "
                  f"axis={blob.get('axis')}) | eval pairs={len(test_pairs)} (axis={args.axis}) | "
                  f"alphas={alphas} | eval_sets={eval_sets} | kinds={kinds}")
        else:
            # vectors_only: build from ALL matched pairs (no held-out test) and save — used to mint
            # per-identity v_bias cheaply without paying for the full alpha sweep.
            train_pairs, test_pairs = ((pairs, pairs.iloc[0:0]) if args.vectors_only
                                       else stratified_split(pairs, args.train_frac))
            print(f"Pairs: {len(pairs)} -> train {len(train_pairs)} / test {len(test_pairs)} | "
                  f"alphas={alphas} | eval_sets={eval_sets} | kinds={kinds} | vector_position={args.vector_position}")
            vectors, norms = build_vectors(model, tokenizer, device, train_pairs, args.vector_position, n_layers)
        if args.save_vectors and args.load_vectors is None:
            # Persist the directions (+ provenance) so they can be injected into OTHER tasks. The
            # vector lives in resid_pre[L] space; applying it to BBQ's resid_pre[L] is the OOD test.
            args.save_vectors.parent.mkdir(parents=True, exist_ok=True)
            # Identity-label column for provenance: WinoQueer uses Gender_ID_x, the combined
            # BBQ/CrowS cohorts use Group_x. Fall back gracefully so non-WQ cohorts don't crash.
            idcol = next((c for c in ("Gender_ID_x", "Group_x") if c in train_pairs.columns), None)
            torch.save({
                "vectors": vectors,                       # [n_layers, d_model], float32
                "norms": norms,                           # [n_layers]
                "vector_position": args.vector_position,
                "tl_model_name": args.tl_model_name,
                "n_layers": n_layers,
                "n_train_pairs": int(len(train_pairs)),
                "train_identity_counts": (train_pairs[idcol].value_counts().to_dict() if idcol else {}),
                "pairs_csv": str(args.pairs_csv),
            }, args.save_vectors)
            print(f"Saved steering vectors -> {args.save_vectors}")
        if args.vectors_only:
            print(f"vectors_only: built+saved from {len(train_pairs)} pairs; skipping the sweep.")
            print(f"runtime_seconds: {time.perf_counter() - started:.2f}")
            return
        # Several norm-matched random directions (not one) so the control isn't at the mercy of a
        # single draw coincidentally aligning with the bias direction.
        rand_vectors_list = ([random_matched(vectors, norms, seed_offset=s) for s in range(args.n_random_seeds)]
                             if "random" in kinds else [])

        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_HEADER); writer.writeheader()
            skipped = 0
            for pid, row in tqdm(test_pairs.reset_index(drop=True).iterrows(), total=len(test_pairs), desc="Steering sweep"):
                p = prep_pair(tokenizer, row, device)
                if p is None:
                    skipped += 1
                    continue
                p = attach_meta(p, row)
                for es in eval_sets:
                    rows = eval_pair(model, p, es, vectors, rand_vectors_list, alphas, layers, kinds, device)
                    for r in rows:
                        r["pair_id"] = int(pid)
                    writer.writerows(rows)
                f.flush()
                if device == "cuda":
                    torch.cuda.empty_cache()
            print(f"Skipped (unalignable): {skipped}/{len(test_pairs)}")
        raw = pd.read_csv(raw_path)

    out_paths, best = make_outputs(raw, args.out_dir, args.patch_profile_csv, kl_budget=args.kl_budget)
    print("\nWrote:")
    for p in [raw_path] + out_paths:
        print(f"  {p}")
    print("\nBest steering operating points (layer, alpha):")
    for es, (L, a) in best.items():
        regime = "induce→1" if es == "control" else "de-bias→0"
        sub = raw[(raw.kind == "real") & (raw.eval_set == es) & (raw.layer == L) & (raw.alpha == a)]
        print(f"  eval={es} ({regime}): L{int(L)} alpha={a:g} -> bias_fraction(median)={sub['bias_fraction'].median():.3f}, "
              f"KL={sub['kl_readout'].mean():.3f}")
    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
