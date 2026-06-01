#!/usr/bin/env python3
"""Single source of truth for the identity-segmented WinoQueer circuit analysis.

Taxonomy (grounded in the data; the control word confirms the axis):
  sexual_orientation : Asexual, Bisexual, Gay, Lesbian, Pansexual   (controls Straight/Heterosexual)
  gender_identity    : Transgender, NB                              (controls Cis/Cisgender)
  umbrella           : Queer, LGBTQ      (pair with BOTH controls; kept SEPARATE, excluded from the
                                          7-identity specific analyses, reported only as contrast)

Also provides the segmentation grouping keys (axis / identity / axis x predicate / identity x
predicate) and the reusable statistics: empirical-Bayes shrinkage, entropy selectivity, bootstrap
CIs, Benjamini-Hochberg FDR, Jaccard, and rank-biased overlap.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- taxonomy
IDENTITY_AXIS: dict[str, str] = {
    "Asexual": "sexual_orientation",
    "Bisexual": "sexual_orientation",
    "Gay": "sexual_orientation",
    "Lesbian": "sexual_orientation",
    "Pansexual": "sexual_orientation",
    "Transgender": "gender_identity",
    "NB": "gender_identity",
    "Queer": "umbrella",
    "LGBTQ": "umbrella",
}
CONTROL_AXIS: dict[str, str] = {
    "Straight": "sexual_orientation",
    "Heterosexual": "sexual_orientation",
    "Cis": "gender_identity",
    "Cisgender": "gender_identity",
}
UMBRELLA: set[str] = {"Queer", "LGBTQ"}
SPECIFIC_IDENTITIES: list[str] = [k for k, v in IDENTITY_AXIS.items() if v != "umbrella"]
AXES: list[str] = ["sexual_orientation", "gender_identity", "umbrella"]
PREDICATE_CATEGORIES: list[str] = [
    "ABNORMALITY_DEVIANCE", "EXCLUSION_NONBELONGING", "GENERAL_NEGATIVE_STEREOTYPE",
    "IDENTITY_ASSOCIATION", "INAUTHENTICITY_ILLEGITIMACY", "PATHOLOGIZING_MEDICALIZING",
    "SEXUAL_MORAL_THREAT", "SOCIAL_PROFESSIONAL_INCOMPETENCE", "UNCLEAR_OTHER", "VIOLENCE_HARM",
]
MIN_CELL = 30                     # min pairs for an identity x predicate cell to be reported
LEVELS = ["axis", "identity", "axispred", "idpred"]
CELL_COL = "predicate_label_provisional"
GROUP_SEP = "::"                  # group key = "<level>::<value>"


def annotate(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Add `axis`, `identity` (= Gender_ID_x), `is_umbrella`. With strict=True, raise on any
    Gender_ID_x not in the taxonomy or any specific identity paired with the wrong control axis."""
    out = df.copy()
    out["identity"] = out["Gender_ID_x"].astype(str)
    out["axis"] = out["identity"].map(IDENTITY_AXIS)
    out["is_umbrella"] = out["identity"].isin(UMBRELLA)
    if out["axis"].isna().any():
        unknown = sorted(out.loc[out["axis"].isna(), "identity"].unique())
        msg = f"Unknown Gender_ID_x values not in taxonomy: {unknown}"
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    if strict:
        m = axis_mismatch_count(df)
        if m:
            raise ValueError(f"{m} specific-identity pairs have a control whose axis disagrees.")
    return out


def axis_mismatch_count(df: pd.DataFrame) -> int:
    """# specific (non-umbrella) pairs whose control word's axis disagrees with the identity's axis."""
    ax = df["Gender_ID_x"].map(IDENTITY_AXIS)
    cy = df["Gender_ID_y"].map(CONTROL_AXIS)
    mask = ax.isin(["sexual_orientation", "gender_identity"]) & cy.notna() & (ax != cy)
    return int(mask.sum())


def group_label(level: str, value: str) -> str:
    return f"{level}{GROUP_SEP}{value}"


def split_group(key: str) -> tuple[str, str]:
    level, _, value = str(key).partition(GROUP_SEP)
    return level, value


def group_keys_for_row(row) -> list[str]:
    """All segmentation group keys a single annotated pair belongs to (axis, identity, axispred, idpred).
    Umbrella identities contribute to axis='umbrella' and their own identity group, but NOT to the
    sexual_orientation/gender_identity axis aggregates."""
    axis = IDENTITY_AXIS.get(str(row["Gender_ID_x"]))
    identity = str(row["Gender_ID_x"])
    pred = str(row[CELL_COL])
    keys = [group_label("axis", axis), group_label("identity", identity),
            group_label("axispred", f"{axis}|{pred}"), group_label("idpred", f"{identity}|{pred}")]
    return keys


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the 4 group-key columns (axis_key/identity_key/axispred_key/idpred_key) to an annotated df."""
    out = df.copy()
    axis = out["Gender_ID_x"].map(IDENTITY_AXIS)
    ident = out["Gender_ID_x"].astype(str)
    pred = out[CELL_COL].astype(str)
    out["axis_key"] = (level_prefix := "axis" + GROUP_SEP) + axis.astype(str)
    out["identity_key"] = "identity" + GROUP_SEP + ident
    out["axispred_key"] = "axispred" + GROUP_SEP + axis.astype(str) + "|" + pred
    out["idpred_key"] = "idpred" + GROUP_SEP + ident + "|" + pred
    return out


# ----------------------------------------------------------------------------- statistics
def eb_shrink(means: np.ndarray, ns: np.ndarray, grand_mean: float, k: float | None = None) -> np.ndarray:
    """Empirical-Bayes shrinkage of per-group means toward the grand mean.

    shrunk_g = (n_g * mean_g + k * grand_mean) / (n_g + k)

    Small-n groups are pulled hardest toward `grand_mean`, so they cannot fake specificity.
    `k` is the prior strength; defaults to the median group n.
    """
    means = np.asarray(means, dtype=float)
    ns = np.asarray(ns, dtype=float)
    if k is None:
        pos = ns[ns > 0]
        k = float(np.median(pos)) if pos.size else 1.0
    denom = ns + k
    return np.where(denom > 0, (ns * means + k * grand_mean) / denom, grand_mean)


def selectivity(frac_by_group: Sequence[float], n_by_group: Sequence[float],
                grand_mean: float | None = None, k: float | None = None, min_n: int = 1) -> float:
    """Entropy-based selectivity in [0,1] over G groups, on EB-shrunk group means.

    p_g = max(ē_g, 0) / Σ_g max(ē_g, 0);  selectivity = 1 − H(p)/log G.
    1 => one group carries all the (positive) effect; 0 => uniform across groups.

    Structural-zero / sub-threshold groups (n < min_n) are EXCLUDED from the denominator (they
    would otherwise be shrunk to the grand mean and fake a flat, low-selectivity distribution).
    Returns NaN when fewer than 2 groups qualify or no qualifying group has a positive shrunk mean.
    """
    means = np.asarray(frac_by_group, dtype=float)
    ns = np.asarray(n_by_group, dtype=float)
    incl = ns >= min_n
    means, ns = means[incl], ns[incl]
    G = means.size
    if G < 2:
        return float("nan")
    if grand_mean is None:
        w = ns.sum()
        grand_mean = float((means * ns).sum() / w) if w > 0 else float(np.nanmean(means))
    shrunk = eb_shrink(means, ns, grand_mean, k=k)
    pos = np.clip(shrunk, 0, None)
    total = pos.sum()
    if total <= 0:
        return float("nan")
    p = pos / total
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    return float(1.0 - H / np.log(G))


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, pct: tuple[float, float] = (2.5, 97.5),
                 seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap percentile CI on the mean. Returns (mean, lo, hi). NaNs dropped."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, pct)
    return (float(v.mean()), float(lo), float(hi))


def bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                      pct: tuple[float, float] = (2.5, 97.5), seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap percentile CI on mean(a) − mean(b) (independent resamples). Returns (diff, lo, hi)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ai = rng.integers(0, a.size, size=(n_boot, a.size))
    bi = rng.integers(0, b.size, size=(n_boot, b.size))
    boot = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    lo, hi = np.percentile(boot, pct)
    return (float(a.mean() - b.mean()), float(lo), float(hi))


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.10) -> np.ndarray:
    """Return a boolean mask of hypotheses passing BH-FDR at level q. NaN p-values fail."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.zeros(p.shape, dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    sub = p[idx]
    order = np.argsort(sub)
    m = sub.size
    thresh = (np.arange(1, m + 1) / m) * q
    passed = sub[order] <= thresh
    if passed.any():
        kmax = int(np.max(np.where(passed)[0]))
        out[idx[order[: kmax + 1]]] = True
    return out


def sign_test_p(values: Sequence[float]) -> float:
    """Two-sided sign test that the median effect != 0 (binomial on the sign). NaN if no nonzero."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    v = v[v != 0]
    n = v.size
    if n == 0:
        return float("nan")
    pos = int((v > 0).sum())
    # two-sided binomial tail via normal approx for large n, exact-ish for small
    from math import comb
    if n <= 60:
        tail = sum(comb(n, i) for i in range(0, min(pos, n - pos) + 1)) / (2 ** n)
        return float(min(1.0, 2 * tail))
    z = (pos - n / 2) / (0.5 * np.sqrt(n))
    from math import erfc
    return float(erfc(abs(z) / np.sqrt(2)))


# ----------------------------------------------------------------------------- set overlap
def jaccard(a: Iterable, b: Iterable) -> float:
    """Jaccard index |A∩B| / |A∪B|. Two empty sets -> NaN."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return float("nan")
    return len(sa & sb) / len(union)


def rbo(list_a: Sequence, list_b: Sequence, p: float = 0.9) -> float:
    """Rank-biased overlap (non-extrapolated) of two ranked lists, weighting top ranks more.

    RBO = (1−p) Σ_{d=1..k} p^{d−1} · |A_d ∩ B_d| / d,  where A_d/B_d are the top-d prefixes.
    p in (0,1): smaller p concentrates weight on the very top. Identical lists -> ~1.
    """
    k = min(len(list_a), len(list_b))
    if k == 0:
        return float("nan")
    sa: set = set()
    sb: set = set()
    total = 0.0
    for d in range(k):
        sa.add(list_a[d])
        sb.add(list_b[d])
        overlap = len(sa & sb) / (d + 1)
        total += (p ** d) * overlap
    return float((1 - p) * total)


def random_jaccard_null(pool_size: int, k: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Mean/SD Jaccard of two random k-subsets drawn from a pool of `pool_size` items."""
    if pool_size <= 0 or k <= 0 or k > pool_size:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    js = np.empty(n_boot)
    for i in range(n_boot):
        a = set(rng.choice(pool_size, size=k, replace=False).tolist())
        b = set(rng.choice(pool_size, size=k, replace=False).tolist())
        js[i] = len(a & b) / len(a | b)
    return (float(js.mean()), float(js.std()))
