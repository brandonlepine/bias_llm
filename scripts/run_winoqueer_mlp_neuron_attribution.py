#!/usr/bin/env python3
"""Per-NEURON MLP attribution patching for WinoQueer — sufficiency AND necessity, all 458k neurons.

Exact activation patching of every MLP neuron is infeasible (32 layers x 14,336 neurons x N pairs
= ~10^8 forwards). The standard scalable solution is ATTRIBUTION PATCHING (AtP; Nanda 2023,
Syed et al. AtP*): a first-order Taylor approximation of the patch effect that scores ALL neurons
from ~2 forward + 2 backward passes per pair. We then EXACT-patch the top-K neurons to verify the
approximation held (MLP neurons are usually faithful, unlike attention-softmax components).

The metric is the same continuation-logp bias score used throughout (avg logP of the shared
predicate continuation), and the SAME stringent single-identity-span token alignment from
run_winoqueer_resid_patching.py / run_winoqueer_head_ablation.py.

  M(run)  = continuation avg logP for that run            (queer high = biased, control low)
  denom   = M(queer) - M(control)                          (the per-pair bias gap)

SUFFICIENCY  (dest = control run; inject queer neuron -> does it CREATE bias?):
  dM_suff[L,n] ~= sum_{control pos c in patchset} (qpost[L][src(c),n] - cpost[L][c,n]) * dM_c/dcpost[L][c,n]
  suff_frac    = dM_suff / denom            ( >0 => injecting neuron n raises the stereotype )

NECESSITY   (dest = queer run; resample control neuron in -> does removing queer REDUCE bias?):
  dM_nec[L,n]  ~= sum_{queer pos q in patchset} (cpost[L][src(q),n] - qpost[L][q,n]) * dM_q/dqpost[L][q,n]
  nec_frac     = -dM_nec / denom            ( >0 => removing neuron n reduces the bias = necessary )

Both fracs are oriented so POSITIVE = the neuron contributes to the bias, directly comparable to
the head sufficiency/necessity work. core_score = z(suff_frac) + z(nec_frac).

Efficiency of AtP: freeze all weights and root the autograd graph at a leaf on
blocks.0.hook_resid_pre, then retain_grad() on every blocks.L.mlp.hook_post. One backward then
yields the exact total gradient wrt all neurons with no 32GB param-grad blowup. The two prompts'
cached hook_post values serve as each other's patch source.

Outputs:
  winoqueer_mlp_neuron_attribution.csv       every neuron: suff/nec frac, sign-consistency, core
  winoqueer_mlp_neuron_attribution_top.csv   trimmed top neurons (convenience)
  winoqueer_mlp_layer_profile.{csv,png}      per-layer attribution mass (which MLP layers matter)
  winoqueer_mlp_top_neurons.png              top neurons by sufficiency and by necessity
  winoqueer_mlp_suff_vs_nec.png              neuron-level sufficiency x necessity (core circuit)
  winoqueer_mlp_fingerprint.png              layer x neuron signed-attribution texture
  winoqueer_mlp_atp_vs_exact.{csv,png}       AtP vs EXACT patch for top-K (validation)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_winoqueer_head_ablation import prep_pair, ablate_positions, load_model  # noqa: E402
from run_winoqueer_resid_patching import continuation_logp  # noqa: E402
from winoqueer_identity_taxonomy import MIN_CELL, IDENTITY_AXIS  # noqa: E402


# ----------------------------------------------------------------------------- positions / helpers
def suff_positions(p, mode: str):
    """Control-run positions to inject the queer state into, + their queer source positions."""
    aln = p["aln"]
    if mode == "identity":
        cpos = list(aln["identity_control_positions"])
    elif mode == "readout":
        cpos = [p["c_cont_start"] - 1]
    else:
        cpos = list(range(p["len_c"]))
    spos = [aln["source_pos"][c] for c in cpos]
    return cpos, spos


def nec_positions(p, mode: str, device):
    """Queer-run positions to resample control into, + their control source positions."""
    qpos = ablate_positions(p, mode, device)            # queer positions (LongTensor)
    csrc = p["qsrc"][qpos]                               # control source for each queer position
    return qpos, csrc


def post_names(n_layers: int):
    from transformer_lens import utils as tl_utils
    return [tl_utils.get_act_name("post", L) for L in range(n_layers)]


# ----------------------------------------------------------------------------- attribution pass
def attribution_forward(model, ids, cont_start, cont_count, pnames, root_name):
    """One forward + backward. Returns (M_float, {name: post_value}, {name: post_grad}).

    Weights are frozen; the graph is rooted at a leaf on `root_name` (resid_pre[0]) so backward
    populates retained grads on every hook_post without touching the 8B param grads.
    """
    store = {}

    def root_hook(act, hook):
        return act.detach().requires_grad_(True)

    def save_hook(act, hook):
        act.retain_grad()
        store[hook.name] = act
        return act

    hooks = [(root_name, root_hook)] + [(n, save_hook) for n in pnames]
    logits = model.run_with_hooks(ids, fwd_hooks=hooks)
    M = continuation_logp(logits, ids[0], cont_start).sum() / cont_count
    M.backward()
    M_val = float(M.item())
    values = {n: store[n].detach() for n in pnames}
    grads = {n: (store[n].grad.detach() if store[n].grad is not None else torch.zeros_like(store[n])) for n in pnames}
    del store, logits, M
    return M_val, values, grads


def run_attribution(model, tokenizer, device, pairs, pnames, root_name, n_layers, d_mlp, ablate_mode,
                    group_map=None):
    # Pooled ratio estimator: accumulate per-neuron attribution NUMERATORS and the scalar
    # bias-gap denominator SEPARATELY, then divide once (Σnum / Σdenom). This is robust to the
    # heavy tail of a mean-of-per-pair-ratios, where a pair with a tiny |denom| (bias gap) would
    # otherwise blow up its ratio and dominate the average. (A true per-pair median over all
    # neurons would need ~20GB to hold every pair's [n_layers,d_mlp] map.) Sign-consistency
    # counters still use the per-pair ratio sign so their meaning is unchanged.
    suff_num_sum = torch.zeros(n_layers, d_mlp, dtype=torch.float32)
    nec_num_sum = torch.zeros(n_layers, d_mlp, dtype=torch.float32)
    suff_pos = torch.zeros(n_layers, d_mlp, dtype=torch.float32)   # sign-consistency counters
    nec_pos = torch.zeros(n_layers, d_mlp, dtype=torch.float32)
    denom_sum = 0.0
    n_used = 0
    skipped = 0
    # Per-group accumulators: gacc[key] = [suff_num_sum, nec_num_sum, suff_pos, nec_pos, denom_sum, n].
    # Each pair routes its (already-CPU) numerators into its axis/identity/axispred/idpred groups.
    gacc: dict[str, list] = {}

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="AtP (suff+nec)"):
        p = prep_pair(tokenizer, row, device)
        if p is None:
            skipped += 1
            continue
        cpos, spos = suff_positions(p, ablate_mode)
        qpos, csrc = nec_positions(p, ablate_mode, device)
        cpos_t = torch.tensor(cpos, device=device)
        spos_t = torch.tensor(spos, device=device)

        with torch.enable_grad():
            M_c, valc, gradc = attribution_forward(model, p["c_ids"], p["c_cont_start"], p["cont_count"], pnames, root_name)
            M_q, valq, gradq = attribution_forward(model, p["q_ids"], p["q_cont_start"], p["cont_count"], pnames, root_name)
        denom = M_q - M_c
        if abs(denom) < 1e-6:
            del valc, gradc, valq, gradq
            continue

        suff_num = torch.empty(n_layers, d_mlp, device=device)
        nec_num = torch.empty(n_layers, d_mlp, device=device)
        for L, name in enumerate(pnames):
            qv, cv = valq[name][0], valc[name][0]          # [Tq,d], [Tc,d]
            cg, qg = gradc[name][0], gradq[name][0]
            # sufficiency: inject queer at control positions, weighted by control-run gradient
            suff_num[L] = ((qv[spos_t] - cv[cpos_t]) * cg[cpos_t]).sum(0)
            # necessity: resample control into queer positions, weighted by queer-run gradient
            nec_num[L] = -((cv[csrc] - qv[qpos]) * qg[qpos]).sum(0)

        suff_num_c, nec_num_c = suff_num.float().cpu(), nec_num.float().cpu()
        # per-pair ratio sign only (for consistency counters); the pooled estimate accumulates
        # the numerators and divides by Σdenom once at the end.
        suff_pos_c = ((suff_num_c / denom) > 0).float()
        nec_pos_c = ((nec_num_c / denom) > 0).float()
        suff_num_sum += suff_num_c
        nec_num_sum += nec_num_c
        suff_pos += suff_pos_c
        nec_pos += nec_pos_c
        denom_sum += denom
        n_used += 1
        if group_map is not None:
            keys = group_map.get(row.get("row_id"))
            if keys:
                for key in keys:
                    ga = gacc.get(key)
                    if ga is None:
                        ga = [torch.zeros(n_layers, d_mlp), torch.zeros(n_layers, d_mlp),
                              torch.zeros(n_layers, d_mlp), torch.zeros(n_layers, d_mlp), 0.0, 0]
                        gacc[key] = ga
                    ga[0] += suff_num_c; ga[1] += nec_num_c; ga[2] += suff_pos_c; ga[3] += nec_pos_c
                    ga[4] += denom; ga[5] += 1
        del valc, gradc, valq, gradq, suff_num, nec_num, suff_num_c, nec_num_c, suff_pos_c, nec_pos_c
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"AtP: used {n_used} pairs, skipped {skipped}")
    n = max(n_used, 1)
    den = denom_sum if abs(denom_sum) > 1e-6 else float("nan")
    res = dict(
        suff_mean=(suff_num_sum / den), nec_mean=(nec_num_sum / den),  # pooled Σnum/Σdenom fracs
        suff_cons=(suff_pos / n), nec_cons=(nec_pos / n), n_used=n_used,
    )
    group_res = {}
    for key, (snum, nnum, spos, npos, gden, gn) in gacc.items():
        if gn == 0:
            continue
        gd = gden if abs(gden) > 1e-6 else float("nan")
        group_res[key] = dict(suff_mean=snum / gd, nec_mean=nnum / gd,
                              suff_cons=spos / gn, nec_cons=npos / gn, n_used=gn)
    return res, group_res


# ----------------------------------------------------------------------------- exact verification
@torch.no_grad()
def verify_topk(model, tokenizer, device, pairs, pnames, top_suff, top_nec, ablate_mode):
    """Exact-patch the given neurons (each in its own batch row) to validate AtP. Returns dict
    neuron->mean exact frac, for both directions. Neurons given as lists of (layer, neuron)."""
    from transformer_lens import utils as tl_utils

    def layer_groups(neurons):
        g = {}
        for i, (L, nu) in enumerate(neurons):
            g.setdefault(L, []).append((i, nu))
        return g

    suff_groups, nec_groups = layer_groups(top_suff), layer_groups(top_nec)
    # Pooled ratio (Σnum/Σdenom), matching run_attribution so the AtP-vs-exact scatter compares
    # like with like. suff and nec share the same per-pair denom (M_q - M_c).
    suff_num_acc = np.zeros(len(top_suff)); nec_num_acc = np.zeros(len(top_nec))
    den_acc = 0.0; cnt = 0

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Exact verify"):
        p = prep_pair(tokenizer, row, device)
        if p is None:
            continue
        _, cache = model.run_with_cache(p["c_ids"], names_filter=lambda nm: nm.endswith("mlp.hook_post"))
        valc = {n: cache[n] for n in pnames}
        _, cache_q = model.run_with_cache(p["q_ids"], names_filter=lambda nm: nm.endswith("mlp.hook_post"))
        valq = {n: cache_q[n] for n in pnames}
        M_c = float(continuation_logp(model(p["c_ids"]), p["c_ids"][0], p["c_cont_start"])[0].item()) / p["cont_count"]
        M_q = float(continuation_logp(model(p["q_ids"]), p["q_ids"][0], p["q_cont_start"])[0].item()) / p["cont_count"]
        denom = M_q - M_c
        if abs(denom) < 1e-6:
            del cache, cache_q
            continue

        cpos, spos = suff_positions(p, ablate_mode)
        cpos_t = torch.tensor(cpos, device=device); spos_t = torch.tensor(spos, device=device)
        qpos, csrc = nec_positions(p, ablate_mode, device)

        # sufficiency: inject queer neuron into the control run
        if top_suff:
            B = len(top_suff)
            fwd = []
            for L, rows in suff_groups.items():
                name = tl_utils.get_act_name("post", L)
                src = valq[name]

                def hk(act, hook, rows=rows, src=src, spos_t=spos_t, cpos_t=cpos_t):
                    act = act.clone()
                    for i, nu in rows:
                        act[i, cpos_t, nu] = src[0, spos_t, nu].to(act.dtype)
                    return act
                fwd.append((name, hk))
            logits = model.run_with_hooks(p["c_ids"].repeat(B, 1), fwd_hooks=fwd)
            sums = continuation_logp(logits, p["c_ids"][0], p["c_cont_start"]) / p["cont_count"]
            suff_num_acc += (sums.cpu().numpy() - M_c)

        # necessity: resample control neuron into the queer run
        if top_nec:
            B = len(top_nec)
            fwd = []
            for L, rows in nec_groups.items():
                name = tl_utils.get_act_name("post", L)
                src = valc[name]

                def hk(act, hook, rows=rows, src=src, csrc=csrc, qpos=qpos):
                    act = act.clone()
                    for i, nu in rows:
                        act[i, qpos, nu] = src[0, csrc, nu].to(act.dtype)
                    return act
                fwd.append((name, hk))
            logits = model.run_with_hooks(p["q_ids"].repeat(B, 1), fwd_hooks=fwd)
            sums = continuation_logp(logits, p["q_ids"][0], p["q_cont_start"]) / p["cont_count"]
            nec_num_acc += (M_q - sums.cpu().numpy())

        den_acc += denom  # shared denom, once per used pair
        cnt += 1
        del cache, cache_q
        if device == "cuda":
            torch.cuda.empty_cache()

    cnt = max(cnt, 1)
    den = den_acc if abs(den_acc) > 1e-6 else float("nan")
    return suff_num_acc / den, nec_num_acc / den, cnt


# ----------------------------------------------------------------------------- outputs / plots
def build_neuron_df(res, n_layers, d_mlp):
    layers = np.repeat(np.arange(n_layers), d_mlp)
    neurons = np.tile(np.arange(d_mlp), n_layers)
    suff = res["suff_mean"].numpy().ravel()
    nec = res["nec_mean"].numpy().ravel()
    df = pd.DataFrame({
        "layer": layers, "neuron": neurons,
        "suff_frac": suff, "nec_frac": nec,
        "suff_consistency": res["suff_cons"].numpy().ravel(),
        "nec_consistency": res["nec_cons"].numpy().ravel(),
    })
    # Rank-percentile core score: robust to AtP sufficiency outliers (a single huge-magnitude
    # false-positive would dominate a z-score). 0..2; top-right (high on BOTH) -> ~2.
    df["suff_pct"] = df["suff_frac"].rank(pct=True)
    df["nec_pct"] = df["nec_frac"].rank(pct=True)
    df["core_score"] = df["suff_pct"] + df["nec_pct"]
    df["n_pairs"] = res["n_used"]
    return df


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


def build_group_map(pairs, cohort_csv):
    """row_id -> list of '<level>::<value>' group keys, COLUMN-DRIVEN so it works for both schemas:
      - WinoQueer cohort: identity=Gender_ID_x, axis derived from IDENTITY_AXIS, block=identity
      - BBQ/CrowS cohort: identity=Group_x, axis/block read from the cohort's own columns
    Levels emitted: axis, identity, block, axispred (axis|predicate), idpred (identity|predicate).
    Returns None only if there is genuinely no identity/predicate/row_id to group on."""
    src = None
    if cohort_csv is not None and Path(cohort_csv).exists():
        src = pd.read_csv(cohort_csv)
    elif pairs is not None and "row_id" in pairs.columns:
        src = pairs
    if src is None or "row_id" not in src.columns or "predicate_label_provisional" not in src.columns:
        return None
    id_col = next((c for c in ("Group_x", "Gender_ID_x", "identity") if c in src.columns), None)
    if id_col is None:
        return None
    has_axis, has_block = "axis" in src.columns, "block" in src.columns
    gm = {}
    for _, r in src.iterrows():
        ident = str(r[id_col])
        axis = str(r["axis"]) if has_axis else str(IDENTITY_AXIS.get(ident, "unknown"))
        block = str(r["block"]) if has_block else ident
        pred = str(r["predicate_label_provisional"])
        gm[r["row_id"]] = [f"axis::{axis}", f"identity::{ident}", f"block::{block}",
                           f"axispred::{axis}|{pred}", f"idpred::{ident}|{pred}"]
    return gm


def layer_profile(res, n_layers) -> pd.DataFrame:
    """Per-layer attribution mass (summed positive / abs neuron attribution) for a result dict."""
    suff_m, nec_m = res["suff_mean"].numpy(), res["nec_mean"].numpy()
    return pd.DataFrame({
        "layer": np.arange(n_layers),
        "suff_pos_mass": np.clip(suff_m, 0, None).sum(1),
        "nec_pos_mass": np.clip(nec_m, 0, None).sum(1),
        "suff_abs_mass": np.abs(suff_m).sum(1),
        "nec_abs_mass": np.abs(nec_m).sum(1),
    })


def write_group_outputs(group_res, out_dir, n_layers, d_mlp, top_csv_rows):
    """Per-group neuron CSVs (trimmed to top rows by |core_score|) + a combined layer-profile CSV.
    Predicate-crossed levels (axispred/idpred) are gated at n>=MIN_CELL; axis/identity always emit."""
    written, prof_rows = [], []
    for key in sorted(group_res):
        gres = group_res[key]
        level = key.split("::", 1)[0]
        gated = level in ("axispred", "idpred")
        if gated and gres["n_used"] < MIN_CELL:
            continue
        gdf = build_neuron_df(gres, n_layers, d_mlp)
        gdf.insert(0, "group", key); gdf.insert(1, "level", level)
        path = out_dir / f"mlp_neuron_attribution__{_safe_name(key)}.csv"
        gdf.reindex(gdf["core_score"].abs().sort_values(ascending=False).index).head(top_csv_rows).to_csv(path, index=False)
        written.append(path)
        prof = layer_profile(gres, n_layers)
        prof["group"] = key; prof["level"] = level; prof["n_used"] = gres["n_used"]
        prof_rows.append(prof)
    if prof_rows:
        prof_path = out_dir / "mlp_layer_profile_by_group.csv"
        pd.concat(prof_rows, ignore_index=True).to_csv(prof_path, index=False)
        written.append(prof_path)
    return written


def make_plots(df, res, out_dir, n_layers, d_mlp, top_bar=25, top_scatter=40):
    paths = []

    # 1. per-layer attribution profile -------------------------------------------------
    suff_m, nec_m = res["suff_mean"].numpy(), res["nec_mean"].numpy()
    prof = pd.DataFrame({
        "layer": np.arange(n_layers),
        "suff_pos_mass": np.clip(suff_m, 0, None).sum(1),
        "nec_pos_mass": np.clip(nec_m, 0, None).sum(1),
        "suff_abs_mass": np.abs(suff_m).sum(1),
        "nec_abs_mass": np.abs(nec_m).sum(1),
    })
    prof_csv = out_dir / "mlp_layer_profile.csv"
    prof.to_csv(prof_csv, index=False); paths.append(prof_csv)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(prof["layer"], prof["suff_pos_mass"], "-o", color="#2c7fb8", lw=2, ms=4, label="sufficiency (Σ positive neuron attr)")
    ax.plot(prof["layer"], prof["nec_pos_mass"], "-s", color="#c0392b", lw=2, ms=4, label="necessity (Σ positive neuron attr)")
    ax.set_xlabel("layer"); ax.set_ylabel("summed positive neuron attribution (frac of bias gap)")
    ax.set_title("WinoQueer MLP — which layers' neurons write / are needed for the bias")
    ax.legend()
    p = out_dir / "mlp_layer_profile.png"
    fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 2. top-neuron bars ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, col, color, title in [
        (axes[0], "suff_frac", "#2c7fb8", "Top sufficiency neurons (injecting → creates bias)"),
        (axes[1], "nec_frac", "#c0392b", "Top necessity neurons (removing → reduces bias)"),
    ]:
        t = df.sort_values(col, ascending=False).head(top_bar).iloc[::-1]
        labels = [f"L{int(l)}N{int(nn)}" for l, nn in zip(t["layer"], t["neuron"])]
        ax.barh(range(len(t)), t[col], color=color, alpha=0.85)
        ax.set_yticks(range(len(t))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("attribution (frac of bias gap)"); ax.set_title(title, fontsize=11)
    p = out_dir / "mlp_top_neurons.png"
    fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 3. neuron sufficiency x necessity (density + labeled core) ------------------------
    # The signal lives in a thin tail (99.8% of neurons are ~0); set the axes to SPAN the core
    # overlay rather than clip to it — the near-zero bulk collapses to the origin blob.
    fig, ax = plt.subplots(figsize=(10, 8.5))
    hb = ax.hexbin(df["suff_frac"], df["nec_frac"], gridsize=120, bins="log", cmap="Greys", mincnt=1)
    fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02).set_label("log10(neuron count)")
    core = df.sort_values("core_score", ascending=False).head(top_scatter)
    sc = ax.scatter(core["suff_frac"], core["nec_frac"], c=core["layer"], cmap="viridis",
                    s=55, edgecolor="white", linewidth=0.5, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.10).set_label("layer (of labeled core neurons)")
    for _, r in core.head(15).iterrows():
        ax.annotate(f"L{int(r['layer'])}N{int(r['neuron'])}", (r["suff_frac"], r["nec_frac"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=8, fontweight="bold")
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(min(df["suff_frac"].quantile(0.01), -0.002), core["suff_frac"].max() * 1.08)
    ax.set_ylim(min(df["nec_frac"].quantile(0.01), -0.002), core["nec_frac"].max() * 1.10)
    ax.set_xlabel("sufficiency:  inject neuron → Δ bias  (frac of bias gap)")
    ax.set_ylabel("necessity:  remove neuron → Δ bias  (frac of bias gap)")
    ax.set_title("WinoQueer MLP neurons — sufficiency × necessity\n(grey = all 458k; colored = top core circuit, top-right)")
    p = out_dir / "mlp_suff_vs_nec.png"
    fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 4. layer x neuron fingerprint (signed sufficiency) -------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, mat, title in [(axes[0], suff_m, "sufficiency"), (axes[1], nec_m, "necessity")]:
        v = max(float(np.quantile(np.abs(mat), 0.999)), 1e-9)
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_ylabel("layer"); ax.set_title(f"MLP {title} attribution (layer × neuron)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    axes[1].set_xlabel(f"neuron index (0..{d_mlp-1})")
    p = out_dir / "mlp_fingerprint.png"
    fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths, prof


def make_verify_plot(df, top_suff, top_nec, suff_exact, nec_exact, out_dir):
    rows = []
    for (L, nu), ex in zip(top_suff, suff_exact):
        atp = float(df[(df.layer == L) & (df.neuron == nu)]["suff_frac"].iloc[0])
        rows.append({"direction": "sufficiency", "layer": L, "neuron": nu, "atp_frac": atp, "exact_frac": ex})
    for (L, nu), ex in zip(top_nec, nec_exact):
        atp = float(df[(df.layer == L) & (df.neuron == nu)]["nec_frac"].iloc[0])
        rows.append({"direction": "necessity", "layer": L, "neuron": nu, "atp_frac": atp, "exact_frac": ex})
    vdf = pd.DataFrame(rows)
    vcsv = out_dir / "mlp_atp_vs_exact.csv"
    vdf.to_csv(vcsv, index=False)

    # Authoritative neuron list: AtP screens, EXACT decides. Re-rank the verified neurons by the
    # exact effect (a positive AtP neuron that exact-patching flips to <=0 is an AtP false positive).
    ver = vdf.sort_values(["direction", "exact_frac"], ascending=[True, False]).copy()
    ver["atp_false_positive"] = (ver["atp_frac"] > 0) & (ver["exact_frac"] <= 0)
    vtop = out_dir / "mlp_verified_top.csv"
    ver.to_csv(vtop, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    stats = {}
    for direction, color in [("sufficiency", "#2c7fb8"), ("necessity", "#c0392b")]:
        s = vdf[vdf.direction == direction]
        if s.empty:
            continue
        ax.scatter(s["atp_frac"], s["exact_frac"], c=color, s=34, alpha=0.8, edgecolor="white", linewidth=0.3, label=direction)
        pear = s["atp_frac"].corr(s["exact_frac"])
        spear = s["atp_frac"].corr(s["exact_frac"], method="spearman")
        sign = float((np.sign(s["atp_frac"]) == np.sign(s["exact_frac"])).mean())
        stats[direction] = (pear, spear, sign)
    lim = np.nanmax(np.abs(np.concatenate([vdf["atp_frac"].values, vdf["exact_frac"].values]))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color="#999", lw=1, label="y = x")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("AtP (approx) attribution"); ax.set_ylabel("EXACT patch attribution")
    txt = "  |  ".join(f"{d}: r={v[0]:.2f}, ρ={v[1]:.2f}, sign={v[2]*100:.0f}%" for d, v in stats.items())
    ax.set_title("AtP vs exact MLP-neuron patching (top-K)\n" + txt, fontsize=10)
    ax.legend()
    p = out_dir / "mlp_atp_vs_exact.png"
    fig.tight_layout(); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return [vcsv, vtop, p], ver, stats


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="WinoQueer per-neuron MLP attribution patching (AtP) + exact verification.")
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tl_model_name", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="Hard cap on total pairs (applied AFTER --max_per_predicate). Default: no cap.")
    ap.add_argument("--max_per_predicate", type=int, default=None)
    ap.add_argument("--ablate_positions", choices=["all", "identity", "readout"], default="all")
    ap.add_argument("--verify_topk", type=int, default=96, help="# top neurons per direction to exact-patch (0=skip).")
    ap.add_argument("--verify_pairs", type=int, default=100, help="# pairs used for exact verification.")
    ap.add_argument("--top_csv_rows", type=int, default=4000)
    ap.add_argument("--no_resort", action="store_true",
                    help="Consume --pairs_csv in file order (no bias_score re-sort / per-predicate cap).")
    ap.add_argument("--cohort_csv", type=Path, default=None,
                    help="Cohort with row_id/Gender_ID_x/predicate_label_provisional for per-group "
                         "accumulators. If omitted, those columns are read from --pairs_csv when present.")
    args = ap.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv)
    if args.no_resort:
        pairs = pairs.reset_index(drop=True)
        print(f"Pairs: {len(pairs)} (no_resort) | ablate_positions={args.ablate_positions} | verify_topk={args.verify_topk}")
    else:
        pairs = pairs.sort_values("bias_score", ascending=False)
        if args.max_per_predicate is not None and "predicate" in pairs.columns:
            pairs = pairs.groupby("predicate", sort=False, group_keys=False).head(args.max_per_predicate).sort_values("bias_score", ascending=False)
        if args.max_pairs is not None:
            pairs = pairs.head(args.max_pairs)
        pairs = pairs.reset_index(drop=True)
        print(f"Pairs: {len(pairs)} | ablate_positions={args.ablate_positions} | verify_topk={args.verify_topk}")

    group_map = build_group_map(pairs, args.cohort_csv)
    print(f"Per-group accumulation: {'ON' if group_map is not None else 'OFF (no axis/identity columns)'}")

    model, tokenizer, device = load_model(args)
    model.requires_grad_(False)                       # freeze weights -> no param-grad blowup
    n_layers, d_mlp = int(model.cfg.n_layers), int(model.cfg.d_mlp)
    from transformer_lens import utils as tl_utils
    root_name = tl_utils.get_act_name("resid_pre", 0)
    pnames = post_names(n_layers)
    print(f"Model: {n_layers} layers x {d_mlp} neurons = {n_layers*d_mlp:,} neurons")

    res, group_res = run_attribution(model, tokenizer, device, pairs, pnames, root_name, n_layers, d_mlp,
                                     args.ablate_positions, group_map=group_map)

    df = build_neuron_df(res, n_layers, d_mlp)
    full_csv = args.out_dir / "mlp_neuron_attribution.csv"
    df.to_csv(full_csv, index=False)
    top_csv = args.out_dir / "mlp_neuron_attribution_top.csv"
    df.reindex(df["core_score"].abs().sort_values(ascending=False).index).head(args.top_csv_rows).to_csv(top_csv, index=False)

    group_paths = write_group_outputs(group_res, args.out_dir, n_layers, d_mlp, args.top_csv_rows)
    if group_paths:
        print(f"Per-group MLP outputs: {len(group_paths)} files "
              f"({len(group_res)} groups accumulated, gated levels at n>={MIN_CELL}).")

    plot_paths, prof = make_plots(df, res, args.out_dir, n_layers, d_mlp)

    # exact verification of the top-K neurons per direction
    verify_paths = []
    if args.verify_topk > 0:
        top_suff = [(int(r.layer), int(r.neuron)) for r in df.sort_values("suff_frac", ascending=False).head(args.verify_topk).itertuples()]
        top_nec = [(int(r.layer), int(r.neuron)) for r in df.sort_values("nec_frac", ascending=False).head(args.verify_topk).itertuples()]
        vpairs = pairs.head(min(args.verify_pairs, len(pairs)))
        print(f"\nExact-verifying top-{args.verify_topk} neurons/direction over {len(vpairs)} pairs...")
        suff_exact, nec_exact, vcnt = verify_topk(model, tokenizer, device, vpairs, pnames, top_suff, top_nec, args.ablate_positions)
        verify_paths, ver, vstats = make_verify_plot(df, top_suff, top_nec, suff_exact, nec_exact, args.out_dir)
        print(f"\nAtP screens, EXACT decides — verification over {vcnt} pairs:")
        for d, (pear, spear, sign) in vstats.items():
            fp = int(ver[(ver.direction == d) & ver.atp_false_positive].shape[0])
            print(f"  {d}: Pearson={pear:.3f}  Spearman={spear:.3f}  sign-agreement={sign*100:.0f}%  "
                  f"AtP false-positives={fp}/{args.verify_topk}")
        print("\nExact-validated top neurons (authoritative — sorted by EXACT patch effect):")
        for d in ["sufficiency", "necessity"]:
            s = ver[ver.direction == d].head(8)
            if not s.empty:
                print(f"  [{d}]")
                print(s[["layer", "neuron", "atp_frac", "exact_frac", "atp_false_positive"]].to_string(index=False))

    print("\nWrote:")
    for p in [full_csv, top_csv] + group_paths + plot_paths + verify_paths:
        print(f"  {p}")
    spear = df["suff_frac"].corr(df["nec_frac"], method="spearman")
    print(f"\nSpearman(suff_frac, nec_frac) over {n_layers*d_mlp:,} neurons: {spear:.3f}")
    print("\nTop 15 core (sufficient AND necessary) MLP neurons:")
    show = df.sort_values("core_score", ascending=False).head(15)[
        ["layer", "neuron", "suff_frac", "nec_frac", "suff_consistency", "nec_consistency", "core_score"]]
    print(show.to_string(index=False))
    print("\nTop sufficiency layers (Σ positive neuron attr):")
    print(prof.sort_values("suff_pos_mass", ascending=False).head(8)[["layer", "suff_pos_mass", "nec_pos_mass"]].to_string(index=False))
    print(f"\nruntime_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
