#!/usr/bin/env python3
"""Evaluate every resume under the NEW prompt conditions (single-dimension 0-100
prompts + binary yes/no). The multi_dim_rubric condition is NOT recomputed here:
its results are reused from the prior run (scores_conditions.jsonl +
scores_neutral.jsonl), which used the identical batch=1 code path.

All readouts are deterministic single-forward last-token logit reads (no sampling,
no decode loop). For 0-100 prompts every integer is a single token, so one forward
gives EXPECTED VALUE (DV), ARGMAX (the spec's literal number), and NUMBER-MASS
(parse_success / QC). batch=1 on purpose: KV-cache and left-pad batching both drift
~1e-2 in fp16, which is the size of the effects we are measuring.

Output: results/scores_prompts.jsonl, one row per (resume, prompt_condition).

Example:
  python -m pilot.run_eval_prompts \
    --gen-dirs generated_conditions generated_neutral --readout-check
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from .modelref import default_model
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
    dtype = (torch.float32 if device == "cpu" else torch.float16) \
        if dtype_arg == "auto" else getattr(torch, dtype_arg)
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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--job", default=os.path.join(
        ROOT, "job-descriptions", "mech-e-job-description.txt"))
    ap.add_argument("--gen-dirs", nargs="+",
                    default=[os.path.join(ROOT, "generated_conditions"),
                             os.path.join(ROOT, "generated_neutral")])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "scores_prompts.jsonl"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.job) as f:
        job_text = f.read()

    # gather all resumes (manifest rows) across the gen dirs
    rows = []
    for gd in args.gen_dirs:
        mpath = os.path.join(gd, "manifest.jsonl")
        with open(mpath) as f:
            for line in f:
                r = json.loads(line)
                r["_gen_dir"] = gd
                rows.append(r)
    if args.limit:
        rows = rows[:args.limit]

    device, dtype = pick_device_dtype(args.device, args.dtype)
    print(f"Loading {args.model} on {device} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    yes_ids = single_token_ids(tok, prompts.YES_FORMS)
    no_ids = single_token_ids(tok, prompts.NO_FORMS)
    # bare number tokens 0..100 (all single-token, no leading space)
    number_ids = {}
    for v in range(prompts.NUMBER_SCALE_MAX + 1):
        enc = tok.encode(str(v), add_special_tokens=False)
        if len(enc) == 1:
            number_ids[v] = enc[0]
    assert len(number_ids) == prompts.NUMBER_SCALE_MAX + 1, "0-100 not all single-token"

    def last_logits(prompt_text):
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return model(ids).logits[0, -1, :].float(), ids.shape[1]

    def number_readout(prompt_text):
        logits, n_prompt = last_logits(prompt_text)
        probs = torch.softmax(logits, dim=-1)
        mass = sum(probs[i].item() for i in number_ids.values())
        ev = sum(v * probs[i].item() for v, i in number_ids.items()) / max(mass, 1e-9)
        argmax_v = max(number_ids, key=lambda v: probs[number_ids[v]].item())
        return dict(score=ev, argmax=argmax_v, number_mass=mass,
                    parse_success=bool(mass > 0.5), raw_output=str(argmax_v),
                    token_count_prompt=n_prompt)

    def binary_readout(prompt_text):
        logits, n_prompt = last_logits(prompt_text)
        y = torch.logsumexp(logits[torch.tensor(yes_ids)], dim=0).item()
        n = torch.logsumexp(logits[torch.tensor(no_ids)], dim=0).item()
        m = max(y, n)
        p_yes = math.exp(y - m) / (math.exp(y - m) + math.exp(n - m))
        return dict(score=p_yes, p_yes=p_yes, logit_diff=y - n,
                    parse_success=True, raw_output="Yes" if y > n else "No",
                    token_count_prompt=n_prompt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_done = 0
    n_total = len(rows) * len(prompts.PROMPT_CONDITIONS)
    with open(args.out, "w") as outf:
        for row in rows:
            with open(os.path.join(row["_gen_dir"], row["file"])) as rf:
                resume_text = rf.read()
            tcr = len(tok.encode(resume_text))
            meta = {
                "resume_condition": row["condition_name"],
                "pair_id": row["pair_id"],
                "base_resume_id": row.get("base_resume_id"),
                "name": row.get("name"),
                "gender": row.get("gender"),
                "token_count_resume": tcr,
            }
            for pc in prompts.PROMPT_CONDITIONS:
                if pc["kind"] == "number":
                    out = prompts.build_single_prompt(job_text, resume_text, pc["name"])
                    res = number_readout(out)
                else:
                    res = binary_readout(prompts.build_binary_prompt(job_text, resume_text))
                rec = dict(meta)
                rec["prompt_condition"] = pc["name"]
                rec[pc["dv_key"]] = res["score"]  # canonical DV key
                rec.update({k: v for k, v in res.items() if k != "score"})
                outf.write(json.dumps(rec) + "\n")
                outf.flush()
                n_done += 1
            if n_done % 60 == 0 or n_done == n_total:
                print(f"  scored {n_done}/{n_total}", flush=True)

    print(f"Wrote {n_done} rows to {args.out}")
    print("Next: python -m pilot.analyze_prompts")


if __name__ == "__main__":
    main()
