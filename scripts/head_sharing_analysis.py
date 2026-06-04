#!/usr/bin/env python3
"""Concrete, eyeball-able answer to: do sexual-orientation and gender-identity (and the individual
identities within them) activate the SAME attention heads -- within a dataset and across datasets --
and how does that compare to the other bias types?

Jaccard/selectivity collapse that into one number. This plots the actual per-head effects so the
structure is directly readable. Operates at IDENTITY granularity (so binary sex vs trans/nonbinary
within gender, and gay/lesbian/bi within orientation, stay separate), and aggregates up to axis.

Views:
  D. identity x identity circuit-similarity matrix  -- corr of each identity's 1024-head WRITE-effect
     vector vs every other's. Clustered. Blocks = identities using the same circuit. THE centerpiece:
     do gay/lesbian/bi cluster? does man/woman (binary sex) separate from trans/nonbinary? do
     orientation and gender overlap? does the same identity correlate across datasets?
  E. head x identity heatmap (focus cluster)        -- rows = top heads, cols = each orientation/gender
     identity (dataset-banded). Read a row: hot across many = shared head; isolated = identity-specific.
  A. cross-dataset axis-pooled head scatter         -- same heads do bias work in both datasets?
  B. head x (axis x dataset) heatmap                -- the broad cross-bias-type view.
  C. same-axis replication scatter                  -- per-head effect, an axis in dataset A vs B.

Each --run is NAME PATCHING_RAW COHORT. WRITE=normalized_restoration (hook_z sufficiency). READ=attn_readout.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# WRITE uses normalized_restoration (bias_effect / per-pair denom) so the per-head
# effect is on a comparable scale ACROSS datasets/identities with different baseline
# log-prob gaps. This matches every other Stage-7 analysis script; using raw bias_effect
# here would confound the cross-dataset Pearson scatters (A/C) and shared-vmax heatmaps
# (B/E) by per-dataset scale. (Panel D is a correlation matrix and is scale-invariant.)
WRITE, READ = "normalized_restoration", "attn_readout_to_identity"
ID_PREF = ["identity", "Group_x", "Gender_ID_x"]
DEFAULT_FOCUS = ["sexual_orientation", "gender_identity", "gender", "orientation"]


def hlabel(l, h):
    return f"L{int(l)}·H{int(h)}"


_CANON = {"transgender man": "trans man", "transgender woman": "trans woman",
          "transgender": "trans", "nb": "nonbinary", "non-binary": "nonbinary"}


def canon_identity(s: str) -> str:
    s = re.sub(r"^(a|an) ", "", str(s).strip().lower())
    return _CANON.get(s, s)


def read_cohort_identity(cohort_path: Path) -> pd.DataFrame:
    """-> cohort[row_id, axis, identity] with clean, canonicalized identity labels.

    Prefers `block` + filters to identity_mapped==True (clean labels incl. binary-sex vs trans for
    gender); else falls back to identity/Group_x/Gender_ID_x. Canonicalizes labels so the same
    identity lines up across datasets (e.g. 'a transgender woman' == WinoQueer 'Transgender'->trans).
    """
    chead = list(pd.read_csv(cohort_path, nrows=0).columns)
    use_block = "block" in chead and "identity_mapped" in chead
    idcol = "block" if use_block else next((c for c in ID_PREF if c in chead), None)
    if idcol is None:
        raise SystemExit(f"no identity column among {['block']+ID_PREF} in {chead}")
    want = ["row_id", "axis", idcol] + (["identity_mapped"] if use_block else [])
    coh = pd.read_csv(cohort_path, usecols=want).rename(columns={idcol: "identity"}).drop_duplicates("row_id")
    if use_block:
        coh = coh[coh["identity_mapped"].astype(str).str.lower().isin(["true", "1"])]
    coh["identity"] = coh["identity"].map(canon_identity)
    coh = coh[coh["identity"] != coh["axis"].astype(str).str.lower()]  # drop unmapped leftovers
    return coh[["row_id", "axis", "identity"]]


def load_run(raw_path: Path, cohort_path: Path) -> pd.DataFrame:
    """-> tidy (layer, head, axis, identity, write, read, n) for one dataset."""
    coh = read_cohort_identity(cohort_path)
    raw = pd.read_csv(raw_path, usecols=["row_id", "layer", "head", WRITE, READ])
    df = raw.merge(coh, on="row_id", how="inner")
    g = (df.groupby(["layer", "head", "axis", "identity"])
           .agg(write=(WRITE, "mean"), read=(READ, "mean"), n=(WRITE, "size"))
           .reset_index())
    return g


def load_pair_matrices(runs, focus=None, min_n=40):
    """Per-pair head-effect matrices for the permutation null.

    -> heads (list of (layer,head)), data {name: (X[pairs×H], labels[pairs])}, meta {label:(axis,name)}.
    Only heads present (NaN-free) in every dataset are kept so identity means and corr are well-defined.
    Labels are '<identity> · <name>'; identities with < min_n pairs are dropped.
    """
    per, head_masks, meta = {}, [], {}
    for name, raw, coh_path in runs:
        coh = read_cohort_identity(Path(coh_path))
        if focus:
            coh = coh[coh["axis"].astype(str).str.lower().apply(lambda a: any(f in a for f in focus))]
        id2ax = dict(zip(coh["identity"], coh["axis"].astype(str)))
        rawdf = pd.read_csv(Path(raw), usecols=["row_id", "layer", "head", WRITE])
        df = rawdf.merge(coh[["row_id"]], on="row_id", how="inner")  # restrict to kept pairs/axes
        df = df.merge(coh, on="row_id", how="left")
        df["hid"] = list(zip(df["layer"].astype(int), df["head"].astype(int)))
        piv = df.pivot_table(index="row_id", columns="hid", values=WRITE, aggfunc="mean")
        labels = (coh.set_index("row_id")["identity"].reindex(piv.index).astype(str) + " · " + name)
        vc = labels.value_counts()
        keep = labels.isin(vc[vc >= min_n].index).to_numpy()
        piv, labels = piv[keep], labels[keep]
        for lab in pd.unique(labels):
            meta[lab] = (id2ax.get(lab.rsplit(" · ", 1)[0], "?"), name)
        per[name] = (piv, labels.to_numpy())
        head_masks.append(set(piv.columns))
    heads = sorted(set.intersection(*head_masks)) if head_masks else []
    data = {}
    for name, (piv, labels) in per.items():
        X = piv.reindex(columns=heads).to_numpy(float)
        data[name] = (X, labels)
    # keep only heads NaN-free across every dataset (head patching is dense, so ~all survive)
    good = np.ones(len(heads), bool)
    for X, _ in data.values():
        good &= ~np.isnan(X).any(axis=0)
    heads = [h for h, k in zip(heads, good) if k]
    data = {n: (X[:, good], labs) for n, (X, labs) in data.items()}
    return heads, data, meta


def identity_corr_null(data, meta, n_perm=200, seed=0):
    """Observed identity×identity Pearson matrix + permutation-null one-sided p-values.

    Null: shuffle identity labels WITHIN each dataset across pairs and recompute every identity's
    per-head vector, so the baseline is 'two random pair-groups still correlate via the shared global
    bias signal'. p[i,j] = P(null r >= observed r). Diagonal -> NaN. Also returns the null off-diagonal
    mean distribution so the figure can show 'expected by chance'.
    """
    labels = sorted(meta)
    L = len(labels)

    def means(assign):  # assign: {name: label-array}
        V = np.empty((L, next(iter(data.values()))[0].shape[1]))
        for i, lab in enumerate(labels):
            name = meta[lab][1]
            X, _ = data[name]
            V[i] = X[assign[name] == lab].mean(axis=0)
        return V

    base = {n: labs for n, (X, labs) in data.items()}
    Cobs = np.corrcoef(means(base))
    rs = np.random.RandomState(seed)
    ge = np.zeros((L, L))
    null_offdiag = []
    offmask = ~np.eye(L, dtype=bool)
    for _ in range(n_perm):
        assign = {}
        for n, (X, labs) in data.items():
            p = labs.copy(); rs.shuffle(p); assign[n] = p
        Cp = np.corrcoef(means(assign))
        ge += (Cp >= Cobs)
        null_offdiag.append(float(np.nanmean(Cp[offmask])))
    P = (1.0 + ge) / (n_perm + 1.0)
    np.fill_diagonal(P, np.nan)
    Cdf = pd.DataFrame(Cobs, index=labels, columns=labels)
    Pdf = pd.DataFrame(P, index=labels, columns=labels)
    return Cdf, Pdf, np.array(null_offdiag)


def to_axis(tidy: pd.DataFrame, col="write") -> pd.DataFrame:
    """collapse identity -> axis (n-weighted mean)."""
    t = tidy.assign(_w=tidy[col] * tidy["n"])
    a = t.groupby(["layer", "head", "axis"]).agg(_w=("_w", "sum"), n=("n", "sum")).reset_index()
    a[col] = a["_w"] / a["n"]
    return a[["layer", "head", "axis", col, "n"]]


def pooled(axis_tidy: pd.DataFrame, col="write") -> pd.Series:
    w = axis_tidy.assign(wn=axis_tidy[col] * axis_tidy["n"]).groupby(["layer", "head"]).agg(
        wn=("wn", "sum"), n=("n", "sum"))
    return (w["wn"] / w["n"]).rename(col)


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), float("nan")
    pear = np.corrcoef(x, y)[0, 1]
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return pear, np.corrcoef(rx, ry)[0, 1]


def _cluster_order(C: pd.DataFrame):
    """average-linkage leaf order on a correlation matrix; fallback = input order."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        D = 1 - C.to_numpy()
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2
        order = leaves_list(linkage(squareform(D, checks=False), method="average"))
        return list(C.index[order])
    except Exception as e:
        print(f"  (clustering unavailable: {e}; keeping axis order)")
        return list(C.index)


# ---------- centerpiece: identity x identity circuit similarity ----------
def _stars(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def identity_similarity(tidies: dict, out: Path, focus=None, min_n=40, runs=None, n_perm=0):
    """Identity×identity circuit-similarity matrix. With runs+n_perm>0, adds a within-dataset
    label-permutation null: cells are starred only where observed r beats the shared-global-signal
    baseline (* p<.05, ** p<.01), and the title shows the null off-diagonal mean."""
    P = None; null_off = None
    if runs and n_perm > 0:
        heads, data, meta = load_pair_matrices(runs, focus=focus, min_n=min_n)
        if len(meta) < 3:
            print(f"identity_similarity: only {len(meta)} identities pass min_n={min_n}; skipping")
            return None
        C, P, null_off = identity_corr_null(data, meta, n_perm=n_perm)
        print(f"  null: {len(meta)} identities over {len(heads)} heads, {n_perm} permutations")
    else:
        rows, meta = [], {}
        for name, t in tidies.items():
            for (axis, ident), g in t.groupby(["axis", "identity"]):
                if focus and not any(f in str(axis).lower() for f in focus):
                    continue
                if int(g["n"].max()) < min_n:
                    continue
                lab = f"{ident} · {name}"
                rows.append(g.set_index(["layer", "head"])["write"].rename(lab))
                meta[lab] = (str(axis), name)
        if len(rows) < 3:
            print(f"identity_similarity: only {len(rows)} identities pass min_n={min_n}"
                  f"{' in focus '+str(focus) if focus else ''}; skipping")
            return None
        C = pd.concat(rows, axis=1).corr()

    order = _cluster_order(C)
    C = C.loc[order, order]
    C.to_csv(out.with_suffix(".csv"))
    if P is not None:
        P = P.loc[order, order]
        P.to_csv(out.with_name(out.stem + "_pval.csv"))

    axes_u = sorted({meta[l][0] for l in order})
    apal = {a: plt.cm.tab10(i % 10) for i, a in enumerate(axes_u)}
    ds_u = list(tidies)
    dpal = {d: plt.cm.Set2(i) for i, d in enumerate(ds_u)}

    n = len(order)
    fig, ax = plt.subplots(figsize=(0.5 * n + 4.5, 0.5 * n + 4))
    im = ax.imshow(C.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(order, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = C.iloc[i, j]
            star = "" if P is None or i == j else _stars(P.iloc[i, j])
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "#222")
    # axis + dataset color chips along the left edge
    for i, lab in enumerate(order):
        a, d = meta[lab]
        ax.add_patch(plt.Rectangle((-1.6, i - 0.5), 0.5, 1, color=apal[a], clip_on=False))
        ax.add_patch(plt.Rectangle((-1.0, i - 0.5), 0.5, 1, color=dpal[d], clip_on=False))
    h_axis = [Line2D([0], [0], marker="s", ls="", mfc=apal[a], mec="none", label=a) for a in axes_u]
    h_ds = [Line2D([0], [0], marker="s", ls="", mfc=dpal[d], mec="none", label=d) for d in ds_u]
    leg1 = ax.legend(handles=h_axis, title="axis", loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=h_ds, title="dataset", loc="upper left", bbox_to_anchor=(1.02, 0.45),
              fontsize=8, frameon=False)
    sub = "correlation of per-head WRITE-effect vectors  (1.0 = identical head usage)"
    if null_off is not None:
        offmask = ~np.eye(n, dtype=bool)
        obs_mean = float(np.nanmean(C.to_numpy()[offmask]))
        lo, hi = np.percentile(null_off, [2.5, 97.5])
        sub += (f"\nobserved mean off-diag r={obs_mean:.2f}  vs  null {null_off.mean():.2f} "
                f"[95% {lo:.2f},{hi:.2f}]   * p<.05  ** p<.01 (within-dataset label shuffle)")
    ax.set_title("Do these identities use the SAME circuit?\n" + sub, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.18).set_label("Pearson r of head-effect vectors")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return C


# ---------- E: head x identity heatmap (focus cluster) ----------
def head_identity_heatmap(tidies: dict, out: Path, focus=None, min_n=40, top_per=8):
    cols, top = [], set()
    for name, t in tidies.items():
        for (axis, ident), g in t.groupby(["axis", "identity"]):
            if focus and not any(f in str(axis).lower() for f in focus):
                continue
            if int(g["n"].max()) < min_n:
                continue
            cols.append((name, str(axis), str(ident)))
            for r in g.assign(m=g["write"].abs()).nlargest(top_per, "m")[["layer", "head"]].itertuples(index=False):
                top.add((r.layer, r.head))
    if len(cols) < 2:
        print("head_identity_heatmap: <2 focus identities; skipping")
        return
    heads = sorted(top)
    cols = sorted(cols, key=lambda c: (list(tidies).index(c[0]), c[1], c[2]))
    hidx = {h: i for i, h in enumerate(heads)}
    M = np.full((len(heads), len(cols)), np.nan)
    for jc, (name, axis, ident) in enumerate(cols):
        g = tidies[name]
        g = g[(g["axis"] == axis) & (g["identity"] == ident)].set_index(["layer", "head"])["write"]
        for (l, h), v in g.items():
            if (l, h) in hidx:
                M[hidx[(l, h)], jc] = v
    vmax = np.nanquantile(np.abs(M), 0.96) or 1e-9
    fig, ax = plt.subplots(figsize=(0.5 * len(cols) + 4, 0.30 * len(heads) + 3))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(heads))); ax.set_yticklabels([hlabel(*h) for h in heads], fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c[2]}" for c in cols], rotation=55, ha="right", fontsize=7)
    ds_names = list(tidies)
    pal = {n: plt.cm.Set2(i) for i, n in enumerate(ds_names)}
    for jc, (name, _, _) in enumerate(cols):
        ax.add_patch(plt.Rectangle((jc - 0.5, -1.3), 1, 0.7, color=pal[name], clip_on=False))
    for jc in range(1, len(cols)):
        if cols[jc][0] != cols[jc - 1][0]:
            ax.axvline(jc - 0.5, color="#111", lw=1.5)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", mfc=pal[n], mec="none", label=n) for n in ds_names],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, title="dataset")
    ax.set_title("WRITE effect per head (rows) x orientation/gender identity (cols)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.13).set_label("mean WRITE effect (normalized restoration)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------- A: cross-dataset axis-pooled head scatter ----------
def xdataset_scatter(pa, pb, na, nb, out, topk=12):
    j = pd.concat([pa.rename("a"), pb.rename("b")], axis=1).dropna()
    pear, spear = _corr(j["a"], j["b"])
    layers = np.array([idx[0] for idx in j.index])
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    sc = ax.scatter(j["a"], j["b"], c=layers, cmap="viridis", s=26, alpha=0.8, edgecolor="none")
    lim = [min(j["a"].min(), j["b"].min()), max(j["a"].max(), j["b"].max())]
    ax.plot(lim, lim, "--", color="#888", lw=1, zorder=0)
    ax.axhline(0, color="#ddd", lw=.8); ax.axvline(0, color="#ddd", lw=.8)
    for idx in (j["a"].abs() + j["b"].abs()).sort_values(ascending=False).head(topk).index:
        ax.annotate(hlabel(*idx), (j.loc[idx, "a"], j.loc[idx, "b"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"WRITE effect in {na} (per head)"); ax.set_ylabel(f"WRITE effect in {nb} (per head)")
    ax.set_title(f"Same heads do bias work in both datasets?\n"
                 f"each dot = one head    Pearson r={pear:.2f}   Spearman ρ={spear:.2f}")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label("layer")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return pear, spear


# ---------- B: head x (axis x dataset) heatmap ----------
def head_axis_heatmap(axis_tidies: dict, out: Path, top_per=10):
    top, cols = set(), []
    for name, t in axis_tidies.items():
        for axis, g in t.groupby("axis"):
            cols.append((name, axis))
            for r in g.assign(m=g["write"].abs()).nlargest(top_per, "m")[["layer", "head"]].itertuples(index=False):
                top.add((r.layer, r.head))
    heads = sorted(top)
    cols = sorted(cols, key=lambda c: (list(axis_tidies).index(c[0]), c[1]))
    hidx = {h: i for i, h in enumerate(heads)}
    M = np.full((len(heads), len(cols)), np.nan)
    for jc, (name, axis) in enumerate(cols):
        g = axis_tidies[name]; g = g[g["axis"] == axis].set_index(["layer", "head"])["write"]
        for (l, h), v in g.items():
            if (l, h) in hidx:
                M[hidx[(l, h)], jc] = v
    vmax = np.nanquantile(np.abs(M), 0.96) or 1e-9
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 4, 0.30 * len(heads) + 3))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(heads))); ax.set_yticklabels([hlabel(*h) for h in heads], fontsize=7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=55, ha="right", fontsize=7)
    pal = {n: plt.cm.Set2(i) for i, n in enumerate(axis_tidies)}
    for jc, (name, _) in enumerate(cols):
        ax.add_patch(plt.Rectangle((jc - 0.5, -1.2), 1, 0.7, color=pal[name], clip_on=False))
    for jc in range(1, len(cols)):
        if cols[jc][0] != cols[jc - 1][0]:
            ax.axvline(jc - 0.5, color="#111", lw=1.5)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", mfc=pal[n], mec="none", label=n) for n in axis_tidies],
              loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=8, frameon=False, title="dataset")
    ax.set_title("WRITE effect per head (rows) across every axis x dataset (cols)\n"
                 "hot band across a row = head shared across axes/datasets; isolated cell = axis-specific", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.13).set_label("mean WRITE effect (normalized restoration)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------- C: same-axis replication scatter ----------
def replication_scatter(axis_tidies: dict, out: Path, topk=10):
    names = list(axis_tidies)
    if len(names) < 2:
        return
    a, b = names[0], names[1]
    shared = sorted(set(axis_tidies[a]["axis"]) & set(axis_tidies[b]["axis"]))
    if not shared:
        print("replication: no shared axis; skipping")
        return
    ncol = min(3, len(shared)); nrow = (len(shared) + ncol - 1) // ncol
    fig, axarr = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow), squeeze=False)
    for ax, axis in zip(axarr.ravel(), shared):
        ga = axis_tidies[a][axis_tidies[a].axis == axis].set_index(["layer", "head"])["write"]
        gb = axis_tidies[b][axis_tidies[b].axis == axis].set_index(["layer", "head"])["write"]
        j = pd.concat([ga.rename("a"), gb.rename("b")], axis=1).dropna()
        pear, spear = _corr(j["a"], j["b"])
        ax.scatter(j["a"], j["b"], s=16, alpha=0.6, color="#3b6", edgecolor="none")
        lim = [min(j["a"].min(), j["b"].min()), max(j["a"].max(), j["b"].max())]
        ax.plot(lim, lim, "--", color="#888", lw=1)
        for idx in (j["a"].abs() + j["b"].abs()).nlargest(topk).index:
            ax.annotate(hlabel(*idx), (j.loc[idx, "a"], j.loc[idx, "b"]), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_title(f"{axis}\nr={pear:.2f}  ρ={spear:.2f}", fontsize=9)
        ax.set_xlabel(a, fontsize=8); ax.set_ylabel(b, fontsize=8)
    for ax in axarr.ravel()[len(shared):]:
        ax.axis("off")
    fig.suptitle(f"Same-axis replication: per-head WRITE effect, {a} vs {b}", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", metavar=("NAME", "RAW", "COHORT"), required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--focus_axes", nargs="*", default=DEFAULT_FOCUS,
                    help="substrings; identity-level views restrict to axes matching these")
    ap.add_argument("--min_n", type=int, default=40, help="min pairs for an identity to be shown")
    ap.add_argument("--top_per", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=200,
                    help="within-dataset label permutations for the significance null (0 = skip)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    focus = [f.lower() for f in args.focus_axes] if args.focus_axes else None

    tidies, axis_tidies = {}, {}
    for name, raw, coh in args.run:
        print(f"loading {name} ...")
        t = load_run(Path(raw), Path(coh))
        tidies[name] = t
        axis_tidies[name] = to_axis(t)
        t.to_csv(args.out_dir / f"head_effect_by_identity__{name}.csv", index=False)
        ids = sorted(t["identity"].unique())
        print(f"  {name}: axes={sorted(t['axis'].unique())}\n         identities ({len(ids)}): {ids}")

    print("\n--- D: identity x identity circuit similarity (focus cluster) ---")
    identity_similarity(tidies, args.out_dir / "D_identity_similarity_focus.png", focus=focus,
                        min_n=args.min_n, runs=args.run, n_perm=args.n_perm)
    print("--- D-all: identity x identity circuit similarity (ALL axes) ---")
    identity_similarity(tidies, args.out_dir / "D_identity_similarity_all.png", focus=None,
                        min_n=args.min_n, runs=args.run, n_perm=args.n_perm)

    print("--- E: head x identity heatmap (focus cluster) ---")
    head_identity_heatmap(tidies, args.out_dir / "E_head_x_identity_focus.png", focus=focus, min_n=args.min_n)

    names = list(axis_tidies)
    if len(names) >= 2:
        pe, sp = xdataset_scatter(pooled(axis_tidies[names[0]]), pooled(axis_tidies[names[1]]),
                                  names[0], names[1], args.out_dir / "A_xdataset_head_scatter.png")
        print(f"--- A: cross-dataset head correlation {names[0]} vs {names[1]}: r={pe:.3f} ρ={sp:.3f}")
    head_axis_heatmap(axis_tidies, args.out_dir / "B_head_x_axis_dataset_heatmap.png", top_per=args.top_per)
    replication_scatter(axis_tidies, args.out_dir / "C_shared_axis_replication_scatter.png")
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
