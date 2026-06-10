#!/usr/bin/env python3
"""Score every generated resume with a HuggingFace causal LM (default: the local
base meta-llama/Llama-3.1-8B, same weights as the mech-interp stack).

Readouts (select with --readout), all single-forward-pass logit readouts:
  decision   : logit(Yes) - logit(No) and p(Yes) at the advance/reject position.
  dimensions : expected score (0-9) per rubric dimension from the digit
               distribution, plus the digit probability mass (a QC number).
  both       : decision + dimensions (default).
  rubric     : OPTIONAL free-gen JSON (instruction-tuned / API models only).

Each resume is scored independently. Output: results/scores.jsonl (one row per
resume), joinable to generated/manifest.jsonl on (pair_id, condition).

Examples:
  python -m pilot.run_eval                                   # base model, both
  python -m pilot.run_eval --readout decision --limit 4      # quick sanity
  python -m pilot.run_eval --model meta-llama/Llama-3.1-8B-Instruct --readout rubric
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from .modelref import default_model, dtype_kwargs
DEFAULT_MODEL = default_model()

from . import prompts


def pick_device_dtype(device_arg, dtype_arg):
    import torch
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_arg
    if dtype_arg == "auto":
        dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        dtype = getattr(torch, dtype_arg)
    return device, dtype


def single_token_ids(tok, forms):
    ids = []
    for s in forms:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF id or local path (default: local base Llama-3.1-8B)")
    ap.add_argument("--job", default=os.path.join(
        ROOT, "job-descriptions", "mech-e-job-description.txt"))
    ap.add_argument("--gen-dir", default=os.path.join(ROOT, "generated"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "scores.jsonl"))
    ap.add_argument("--readout", default="both",
                    choices=["both", "decision", "dimensions", "rubric"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=120,
                    help="only for --readout rubric")
    ap.add_argument("--limit", type=int, default=0, help="score only first N resumes")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.job) as f:
        job_text = f.read()
    rows = []
    with open(os.path.join(args.gen_dir, "manifest.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]

    device, dtype = pick_device_dtype(args.device, args.dtype)
    print(f"Loading {args.model} on {device} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwargs(dtype)).to(device).eval()
    is_chat = tok.chat_template is not None

    do_decision = args.readout in ("both", "decision")
    do_dims = args.readout in ("both", "dimensions")
    do_rubric = args.readout == "rubric"

    yes_ids = single_token_ids(tok, prompts.YES_FORMS)
    no_ids = single_token_ids(tok, prompts.NO_FORMS)
    digit_ids = {}
    for d in range(prompts.SCORE_MAX + 1):
        enc = tok.encode(str(d), add_special_tokens=False)
        if len(enc) == 1:
            digit_ids[d] = enc[0]
    if do_decision and (not yes_ids or not no_ids):
        sys.exit("Could not resolve single-token Yes/No ids for this tokenizer.")
    if do_dims and len(digit_ids) < prompts.SCORE_MAX + 1:
        sys.exit("Some digit tokens are multi-token for this tokenizer; adjust SCORE_MAX.")
    if do_rubric and not is_chat:
        print("WARNING: --readout rubric on a base (non-chat) model is unreliable.")

    def last_logits(ids):
        with torch.no_grad():
            return model(ids.to(device)).logits[0, -1, :].float()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_done = 0
    with open(args.out, "w") as outf:
        for row in rows:
            with open(os.path.join(args.gen_dir, row["file"])) as rf:
                resume_text = rf.read()
            # carry ALL manifest metadata through except the file path
            rec = {k: v for k, v in row.items() if k != "file"}

            if do_decision:
                logits = last_logits(prompts.encode_readout(
                    tok, job_text, resume_text, *prompts.decision_spec()))
                y = torch.logsumexp(logits[yes_ids], dim=0).item()
                n = torch.logsumexp(logits[no_ids], dim=0).item()
                m = max(y, n)
                p_yes = math.exp(y - m) / (math.exp(y - m) + math.exp(n - m))
                rec.update(yes_logit=y, no_logit=n, logit_diff=y - n, p_yes=p_yes)

            if do_dims:
                for k in prompts.DIMENSION_KEYS:
                    logits = last_logits(prompts.encode_readout(
                        tok, job_text, resume_text, *prompts.dimension_spec(k)))
                    probs = torch.softmax(logits, dim=-1)
                    dmass = sum(probs[i].item() for i in digit_ids.values())
                    ev = sum(d * probs[i].item() for d, i in digit_ids.items()) / max(dmass, 1e-9)
                    rec[k] = ev
                    rec[k + "_digitmass"] = dmass

            if do_rubric:
                msgs = prompts.build_rubric_messages(job_text, resume_text)
                ids = (tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               return_tensors="pt")
                       if is_chat else
                       tok(prompts.SYSTEM_PROMPT + "\n\n" +
                           prompts._job_and_resume(job_text, resume_text),
                           return_tensors="pt").input_ids).to(device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                rec["rubric_raw"] = gen
                rec["rubric"] = prompts.parse_rubric_json(gen)

            outf.write(json.dumps(rec) + "\n")
            outf.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(rows):
                print(f"  scored {n_done}/{len(rows)}", flush=True)

    print(f"Wrote {n_done} rows to {args.out}")
    print("Next: python -m pilot.analyze --by-gender")


if __name__ == "__main__":
    main()
