#!/usr/bin/env python3
"""Numerical-equivalence gate for speedups, on the CURRENT device/dtype.

The pilot uses exact batch=1 forwards because the effects are ~1e-3 in p(advance)
and shortcut artifacts at that scale are invisible. Two speedups were REJECTED on
Apple MPS (fp16) for drifting ~2e-2 vs batch=1: shared-prefix KV cache, and
left-pad batching. Their drift is hardware/dtype dependent, so on a CUDA pod they
MIGHT be safe. This script measures the drift so you can decide before trusting
batched/cached numbers there.

It compares, for several real prompts, last-token logits/probabilities from:
  (1) batch=1            -- the reference (what the pipeline uses)
  (2) left-pad batch     -- all prompts in one padded forward
  (3) shared-prefix KV   -- encode the prefix once, reuse for each suffix
and reports max |prob diff| overall and on the decision Yes/No tokens.

PASS if max prob diff << effect size (~1e-3). Example (run on the pod):
  python -m pilot.check_numerics --model ../models/Llama-3.1-8B
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from .modelref import default_model, dtype_kwargs
DEFAULT_MODEL = default_model()

from . import prompts
from .run_eval_prompts import pick_device_dtype, single_token_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--resume", default=os.path.join(
        ROOT, "generated_conditions", "pair_0000_control.txt"))
    ap.add_argument("--job", default=os.path.join(
        ROOT, "job-descriptions", "mech-e-job-description.txt"))
    ap.add_argument("--threshold", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    device, dtype = pick_device_dtype(args.device, args.dtype)
    print(f"Device={device} dtype={dtype}  threshold={args.threshold:g}\n")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwargs(dtype)).to(device).eval()

    job = open(args.job).read()
    resume = open(args.resume).read()
    prefix = prompts._prefix(job, resume)
    suffixes = [
        "\n" + prompts.DECISION_QUESTION + "\nAnswer:",
        "\n" + prompts._SCORE_INSTRUCTION + "\nTechnical qualifications (0-9): ",
        "\n" + prompts.SINGLE_QUESTIONS["single_technical"] + prompts.NUMBER_ENDING,
    ]
    full = [prefix + s for s in suffixes]
    yes_ids = single_token_ids(tok, prompts.YES_FORMS)
    no_ids = single_token_ids(tok, prompts.NO_FORMS)
    dec_ids = torch.tensor(yes_ids + no_ids, device=device)

    def b1(text):
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return model(ids).logits[0, -1, :].float()
    ref = [b1(t) for t in full]

    # (2) left-pad batch
    tok.padding_side = "left"
    enc = tok(full, return_tensors="pt", padding=True)
    ids, am = enc.input_ids.to(device), enc.attention_mask.to(device)
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        bat = model(input_ids=ids, attention_mask=am, position_ids=pos).logits[:, -1, :].float()

    # (3) shared-prefix KV cache
    pre_ids = tok(prefix, return_tensors="pt").input_ids.to(device)
    plen = pre_ids.shape[1]
    kv = None
    try:
        cache = DynamicCache()
        with torch.no_grad():
            model(pre_ids, past_key_values=cache, use_cache=True)
        kv = []
        for s in suffixes:
            sfx = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            cp = torch.arange(plen, plen + sfx.shape[1], device=device)
            with torch.no_grad():
                out = model(sfx, past_key_values=cache, use_cache=True, cache_position=cp)
            kv.append(out.logits[0, -1, :].float())
            cache.crop(plen)  # needs transformers with DynamicCache.crop
    except Exception as e:
        print(f"  [shared-prefix KV skipped: {type(e).__name__}: {e}]")

    def report(name, got):
        worst = worst_dec = 0.0
        for i in range(len(full)):
            pr = torch.softmax(ref[i], -1); pg = torch.softmax(got[i], -1)
            worst = max(worst, (pr - pg).abs().max().item())
            worst_dec = max(worst_dec, (pr[dec_ids] - pg[dec_ids]).abs().max().item())
        verdict = "PASS" if max(worst, worst_dec) < args.threshold else "FAIL"
        print(f"  {name:<18} max|prob diff|={worst:.2e}  on Yes/No={worst_dec:.2e}  -> {verdict}")

    print("vs batch=1 reference:")
    report("left-pad batch", bat)
    if kv is not None:
        report("shared-prefix KV", kv)
    print("\nIf PASS on this device, you can safely enable that speedup here.")


if __name__ == "__main__":
    main()
