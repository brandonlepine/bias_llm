#!/usr/bin/env python3
"""Residualize the identity direction out of the bias direction: v_stereotype ~ v_bias - v_identity.

Inputs are two .pt files in the shared steering-vector schema (vectors[n_layers, d_model], norms):
  --bias      v_bias .pt      (run_winoqueer_steering_sweep.py --save_vectors)
  --identity  v_identity .pt  (build_identity_vectors.py)

Modes:
  subtract (default) — v_stereotype[L] = v_bias[L] - v_identity[L]   (the plan's rough residual)
  project            — remove the identity COMPONENT from v_bias per layer:
                       v_stereotype[L] = v_bias[L] - (v_bias[L]·û_id) û_id ,  û_id = v_identity/‖·‖
                       (cleaner: orthogonalizes v_bias against the identity axis; magnitude-invariant
                        to v_identity's norm, unlike plain subtraction)

POSITION CAVEAT: a clean residual needs both vectors read at the same residual-stream position.
v_bias defaults to the 'readout' (decision-token) convention; v_identity's PRIMARY is 'identity'
(identity-token span). If the two files' vector_position disagree this script WARNS and, when the
v_identity file carries matching 'variants', can pull a position-matched variant via --identity_variant.

Output: v_stereotype .pt (same schema, loadable by run_bbq_steering_transfer.py) reporting, per layer,
the cosine(v_bias, v_identity) and the fraction of v_bias norm removed — i.e. how much of the bias
direction is just identity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser(description="v_stereotype = v_bias - v_identity (subtract or project).")
    ap.add_argument("--bias", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["subtract", "project"], default="subtract")
    ap.add_argument("--identity_variant", choices=["identity", "last", "mean", "auto"], default="auto",
                    help="which v_identity readout to use; 'auto' position-matches the bias file when possible")
    args = ap.parse_args()

    b = torch.load(args.bias, map_location="cpu")
    idn = torch.load(args.identity, map_location="cpu")
    vb = b["vectors"].float()
    if vb.shape != idn["vectors"].shape:
        raise SystemExit(f"shape mismatch: bias {tuple(vb.shape)} vs identity {tuple(idn['vectors'].shape)}")

    b_pos = b.get("vector_position", "readout")
    variants = idn.get("variants", {})
    # Pick the v_identity readout variant.
    if args.identity_variant == "auto":
        want = "last" if b_pos == "readout" else b_pos      # readout(decision token) ~ last token
        chosen = want if want in variants else idn.get("vector_position", "identity")
    else:
        chosen = args.identity_variant
    vi = (variants[chosen] if chosen in variants else idn["vectors"]).float()
    i_pos = chosen

    if b_pos != i_pos:
        print(f"WARNING: position mismatch — bias='{b_pos}' vs identity='{i_pos}'. The residual mixes "
              f"positions; pass --identity_variant to align (available: {sorted(variants) or ['<none>']}).")

    cos = F.cosine_similarity(vb, vi, dim=1)               # per layer
    if args.mode == "subtract":
        vs = vb - vi
    else:
        u = F.normalize(vi, dim=1)
        vs = vb - (vb * u).sum(1, keepdim=True) * u
    removed = 1.0 - vs.norm(dim=1) / vb.norm(dim=1).clamp_min(1e-8)

    print(f"bias_pos={b_pos} identity_variant={i_pos} mode={args.mode}")
    print(f"{'L':>3} {'cos(vb,vid)':>12} {'‖vb‖':>8} {'‖vstereo‖':>10} {'frac_removed':>13}")
    for L in range(vb.shape[0]):
        print(f"{L:>3} {cos[L]:>12.3f} {vb[L].norm():>8.2f} {vs[L].norm():>10.2f} {removed[L]:>13.3f}")
    print(f"mean cos={cos.mean():.3f}  mean frac_removed={removed.mean():.3f}")

    out = {
        "vectors": vs,
        "norms": vs.norm(dim=1),
        "vector_position": b_pos,
        "tl_model_name": b.get("tl_model_name"),
        "n_layers": int(vb.shape[0]),
        "source": "v_stereotype",
        "residual_mode": args.mode,
        "bias_src": str(args.bias),
        "identity_src": str(args.identity),
        "identity_variant": i_pos,
        "cos_per_layer": cos,
        "frac_removed_per_layer": removed,
        "axis": idn.get("axis") or b.get("axis"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nSaved v_stereotype -> {args.out}")


if __name__ == "__main__":
    main()
