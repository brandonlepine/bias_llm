#!/usr/bin/env python3
"""Verify that every (queer, control) pair tokenizes to the SAME length and
report exactly which token positions differ.

This is the gate for the causal-inference / mech-interp design: if a pair has
unequal token counts, downstream positions are misaligned and per-position
activation patching is invalid. Exits non-zero if any pair fails equal-length.

Example:
  python -m pilot.audit_tokens --model ../models/Llama-3.1-8B
  python -m pilot.audit_tokens --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from .modelref import default_model
DEFAULT_MODEL = default_model()


def load_pairs(manifest_path):
    by_pair = {}
    with open(manifest_path) as f:
        for line in f:
            row = json.loads(line)
            by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
    return by_pair


def diff_positions(a_ids, b_ids):
    """Indices where token ids differ, up to the shorter length."""
    n = min(len(a_ids), len(b_ids))
    return [i for i in range(n) if a_ids[i] != b_ids[i]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer path or HF id (default: local Llama-3.1-8B)")
    ap.add_argument("--gen-dir", default=os.path.join(ROOT, "generated"))
    ap.add_argument("--add-special-tokens", action="store_true",
                    help="include BOS/EOS in the count (default: off, content-only)")
    ap.add_argument("--show", type=int, default=3, help="how many pairs to detail")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    manifest = os.path.join(args.gen_dir, "manifest.jsonl")
    by_pair = load_pairs(manifest)

    n_pairs = len(by_pair)
    n_fail = 0
    n_diff_tokens = []
    lengths = []
    detailed = 0

    for pair_id in sorted(by_pair):
        rec = by_pair[pair_id]
        with open(os.path.join(args.gen_dir, rec["queer"]["file"])) as f:
            q_txt = f.read()
        with open(os.path.join(args.gen_dir, rec["control"]["file"])) as f:
            c_txt = f.read()
        q_ids = tok.encode(q_txt, add_special_tokens=args.add_special_tokens)
        c_ids = tok.encode(c_txt, add_special_tokens=args.add_special_tokens)
        lengths.append(len(q_ids))

        if len(q_ids) != len(c_ids):
            n_fail += 1
            print(f"[FAIL] pair {pair_id:04d}: queer={len(q_ids)} != control={len(c_ids)} tokens")
            continue

        dpos = diff_positions(q_ids, c_ids)
        n_diff_tokens.append(len(dpos))

        if detailed < args.show:
            detailed += 1
            print(f"[ok]   pair {pair_id:04d}: {len(q_ids)} tokens, "
                  f"{len(dpos)} differing position(s) at {dpos}")
            for i in dpos:
                print(f"          pos {i}: queer={tok.decode([q_ids[i]])!r} "
                      f"control={tok.decode([c_ids[i]])!r}")

    print("\n=== AUDIT SUMMARY ===")
    print(f"pairs:                 {n_pairs}")
    print(f"equal-length pairs:    {n_pairs - n_fail}")
    print(f"FAILED (unequal len):  {n_fail}")
    if lengths:
        print(f"resume length tokens:  min={min(lengths)} max={max(lengths)}")
    if n_diff_tokens:
        import statistics
        print(f"differing tokens/pair: min={min(n_diff_tokens)} "
              f"max={max(n_diff_tokens)} mean={statistics.mean(n_diff_tokens):.2f}")
        if max(n_diff_tokens) == 1:
            print("  -> single-token causal locus on every pair (ideal for patching).")
    if n_fail:
        print("\nRESULT: FAIL -- fix signal spans so token counts match.")
        sys.exit(1)
    print("\nRESULT: PASS -- all pairs token-length matched.")


if __name__ == "__main__":
    main()
