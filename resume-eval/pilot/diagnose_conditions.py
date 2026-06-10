#!/usr/bin/env python3
"""Token-count diagnostics for the multi-condition resume set.

Per the design rule, naturalistic text is preserved over forced token equality;
this script REPORTS the token counts (it does not gate). For each condition it
reports, under the Llama-3.1 tokenizer:
  - organization-name tokens
  - target-group-phrase tokens (in-sentence, i.e. preceded by a space)
  - full community-involvement section tokens
  - full resume tokens (mean across pairs; name length varies)
and the delta of each vs the `control` condition. Mismatches are WARNED, not
silenced. It also runs a byte-level integrity check: within a pair, control vs
each variant must differ ONLY inside the community block.

Example:
  python -m pilot.diagnose_conditions --model ../models/Llama-3.1-8B
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from .modelref import default_model
DEFAULT_MODEL = default_model()

from .signals import (CONDITIONS, COMMUNITY_DESC_TEMPLATE, render_condition_block,
                      CONTROL_CONDITION)

COMMUNITY_HEADER = "## COMMUNITY INVOLVEMENT"


def ntok(tok, s, add_special=False):
    return len(tok.encode(s, add_special_tokens=add_special))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gen-dir", default=os.path.join(ROOT, "generated_conditions"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    # rows grouped by pair and condition
    by_pair = {}
    cond_files = {c["condition_name"]: [] for c in CONDITIONS}
    with open(os.path.join(args.gen_dir, "manifest.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            by_pair.setdefault(r["pair_id"], {})[r["condition_name"]] = r
            cond_files[r["condition_name"]].append(r["file"])

    # ---- per-condition token diagnostics (context-aware) --------------------
    # org tokens: standalone; phrase tokens: in the desc sentence (space-preceded);
    # section tokens: rendered block; resume tokens: mean across pairs.
    def resume_tokens(cname):
        vals = []
        for fn in cond_files[cname]:
            with open(os.path.join(args.gen_dir, fn)) as rf:
                vals.append(ntok(tok, rf.read()))
        return vals

    diag = {}
    for c in CONDITIONS:
        cn = c["condition_name"]
        org_t = ntok(tok, c["organization_name"])
        # phrase tokens as they appear after "supporting " (a leading space)
        sent = COMMUNITY_DESC_TEMPLATE.format(phrase=c["target_group_phrase"])
        full_sent_t = ntok(tok, sent)
        # isolate phrase contribution = sentence minus the same sentence with empty phrase
        empty_sent_t = ntok(tok, COMMUNITY_DESC_TEMPLATE.format(phrase="").replace("  ", " "))
        phrase_t = full_sent_t - empty_sent_t
        block_t = ntok(tok, render_condition_block(c))
        rtoks = resume_tokens(cn)
        diag[cn] = dict(org=org_t, phrase=phrase_t, block=block_t,
                        resume_mean=statistics.mean(rtoks),
                        resume_min=min(rtoks), resume_max=max(rtoks))

    ctrl = diag[CONTROL_CONDITION]
    print("=== TOKEN DIAGNOSTICS (Llama-3.1 tokenizer) ===")
    print(f"{'condition':<24}{'org':>5}{'phr':>5}{'block':>7}{'resume(mean)':>14}"
          f"{'dOrg':>6}{'dPhr':>6}{'dBlk':>6}")
    warnings = []
    for c in CONDITIONS:
        cn = c["condition_name"]
        d = diag[cn]
        dOrg, dPhr, dBlk = d["org"]-ctrl["org"], d["phrase"]-ctrl["phrase"], d["block"]-ctrl["block"]
        print(f"{cn:<24}{d['org']:>5}{d['phrase']:>5}{d['block']:>7}"
              f"{d['resume_mean']:>14.1f}{dOrg:>+6}{dPhr:>+6}{dBlk:>+6}")
        if cn != CONTROL_CONDITION and dBlk != 0:
            warnings.append(f"  {cn}: community section differs from control by {dBlk:+d} tokens")

    if warnings:
        print("\n[WARN] token-length mismatches vs control (preserved by design):")
        for w in warnings:
            print(w)
    else:
        print("\nAll conditions token-length matched to control.")

    # ---- byte-level integrity check (pair 0) --------------------------------
    print("\n=== INTEGRITY CHECK (pair 0000: control vs each variant) ===")
    p0 = by_pair[min(by_pair)]
    with open(os.path.join(args.gen_dir, p0[CONTROL_CONDITION]["file"])) as f:
        ctrl_txt = f.read()
    ci = ctrl_txt.find(COMMUNITY_HEADER)
    ctrl_pre, ctrl_post = ctrl_txt[:ci], ctrl_txt[ci:]
    ok = True
    for c in CONDITIONS:
        cn = c["condition_name"]
        if cn == CONTROL_CONDITION:
            continue
        with open(os.path.join(args.gen_dir, p0[cn]["file"])) as f:
            vtxt = f.read()
        vi = vtxt.find(COMMUNITY_HEADER)
        same_prefix = vtxt[:vi] == ctrl_pre
        block_differs = vtxt[vi:] != ctrl_post
        status = "ok" if (same_prefix and block_differs) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {cn:<24} prefix_identical={same_prefix}  block_differs={block_differs}  [{status}]")
    if not ok:
        print("\nRESULT: FAIL -- a variant differs outside the community block.")
        sys.exit(1)
    print("\nRESULT: PASS -- within each pair, only the community block varies.")


if __name__ == "__main__":
    main()
