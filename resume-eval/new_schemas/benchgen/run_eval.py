#!/usr/bin/env python3
"""Score a generated benchmark run with deterministic readouts (base or instruct).

Dispatches by output_type:
  score_0_100  -> expected value over 0-100 single-token integers (parsed_score = EV).
  salary_usd / bonus_usd -> same 0-100 EV, REMAPPED to the dollar range
                  (salary: [salary_min, salary_max]; bonus: [0, 0.15*salary_max]).
                  EV is in-band by construction; clamp+log only the optional literal-$ secondary.
  binary_yes_no -> logit(Yes) - logit(No), p_yes.
  pairwise_AB  -> CONSTRUCTED from paired groups (treatment vs matched control) as
                  A/B logit, BOTH orders (A=t/B=c and A=c/B=t); records logit_A_minus_B.

Chat models (Instruct) auto-detected via tok.chat_template -> primed partial-assistant
turn; numbers/digits get a re-appended trailing space so the readout lands on a BARE token.
Keeps raw + parsed; logs parse failures and out-of-band clamps. Writes scored.jsonl.

Run from resume-eval/:
  python -m new_schemas.benchgen.run_eval --run-dir new_schemas/runs/<...> \
      --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import math
import os
import collections

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **k):
        return it

from . import loaders
from pilot.run_eval_prompts import pick_device_dtype, single_token_ids
from pilot.modelref import default_model, dtype_kwargs
from pilot import prompts as pilot_prompts

YES_FORMS = pilot_prompts.YES_FORMS
NO_FORMS = pilot_prompts.NO_FORMS
AB_FORMS = {"A": [" A", "A"], "B": [" B", "B"]}


def encode(tok, system, user_body, scaffold, trailing_space):
    """input_ids (1,T) ending right before the answer token; base/chat aware."""
    if tok.chat_template is not None:
        text = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user_body},
             {"role": "assistant", "content": scaffold}],
            continue_final_message=True, tokenize=False)
        add_special = False
    else:
        text = system + "\n\n" + user_body + "\n" + scaffold
        add_special = True
    if trailing_space:
        text = text + " "
    return tok(text, return_tensors="pt", add_special_tokens=add_special).input_ids


def user_body_of(ex):
    """Reconstruct the user turn (everything after the system prompt)."""
    sysp = ex["job_system_prompt"]
    rp = ex["rendered_prompt"]
    return rp.split(sysp + "\n\n", 1)[1] if rp.startswith(sysp) else rp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "examples.jsonl"))]
    if args.limit:
        rows = rows[:args.limit]
    jobs = loaders.load_jobs()

    device, dtype = pick_device_dtype(args.device, args.dtype)
    print(f"Loading {args.model} on {device} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwargs(dtype)).to(device).eval()
    is_chat = tok.chat_template is not None
    print(f"chat_template: {'yes (instruct)' if is_chat else 'no (base)'}")

    yes_ids = single_token_ids(tok, YES_FORMS); no_ids = single_token_ids(tok, NO_FORMS)
    a_ids = single_token_ids(tok, AB_FORMS["A"]); b_ids = single_token_ids(tok, AB_FORMS["B"])
    number_ids = {v: tok.encode(str(v), add_special_tokens=False)[0] for v in range(101)
                  if len(tok.encode(str(v), add_special_tokens=False)) == 1}
    assert len(number_ids) == 101

    def last_logits(ids):
        with torch.no_grad():
            return model(ids.to(device)).logits[0, -1, :].float()

    def number_ev(system, user_body):
        logits = last_logits(encode(tok, system, user_body, "Rating (0-100):", True))
        probs = torch.softmax(logits, -1)
        mass = sum(probs[i].item() for i in number_ids.values())
        ev = sum(v * probs[i].item() for v, i in number_ids.items()) / max(mass, 1e-9)
        argmax = max(number_ids, key=lambda v: probs[number_ids[v]].item())
        return ev, argmax, mass

    def increment_readout(system, user_body, levels):
        """Distribution over the discrete $-increment options -> EV in real dollars + modal offer."""
        logits = last_logits(encode(tok, system, user_body, "Answer:", True))
        probs = torch.softmax(logits, -1)
        idx = {i: number_ids[i] for i in range(len(levels)) if i in number_ids}
        mass = sum(probs[i].item() for i in idx.values())
        ev_usd = sum(levels[i] * probs[j].item() for i, j in idx.items()) / max(mass, 1e-9)
        ev_i = sum(i * probs[j].item() for i, j in idx.items()) / max(mass, 1e-9)
        am = max(idx, key=lambda i: probs[idx[i]].item()) if idx else 0
        return dict(parsed_score=ev_usd, usd=round(ev_usd), ev_usd=ev_usd, ev_index=ev_i,
                    modal_offer_usd=levels[am], argmax_index=am, n_options=len(levels),
                    band_low=levels[0], band_high=levels[-1], band_width=levels[-1] - levels[0],
                    number_mass=mass, parse_success=bool(mass > 0.5))

    def selection_readout(system, user_body, n, focal_pos):
        """Choose 1 of N: distribution over candidate-number tokens 1..N -> focal selection prob."""
        logits = last_logits(encode(tok, system, user_body, "Answer:", True))
        probs = torch.softmax(logits, -1)
        pos_ids = {i: number_ids[i] for i in range(1, n + 1) if i in number_ids}
        mass = sum(probs[j].item() for j in pos_ids.values())
        p = {i: probs[j].item() / max(mass, 1e-9) for i, j in pos_ids.items()}
        chosen = max(p, key=p.get) if p else 0
        return dict(parsed_score=p.get(focal_pos, 0.0), p_focal=p.get(focal_pos, 0.0),
                    chosen_position=chosen, chosen_is_focal=int(chosen == focal_pos),
                    focal_position=focal_pos, n_candidates=n, number_mass=mass, parse_success=bool(mass > 0.5))

    def logit_pair(system, user_body, pos_ids, neg_ids, scaffold="Answer:"):
        logits = last_logits(encode(tok, system, user_body, scaffold, False))
        p = torch.logsumexp(logits[torch.tensor(pos_ids)], 0).item()
        n = torch.logsumexp(logits[torch.tensor(neg_ids)], 0).item()
        m = max(p, n)
        prob = math.exp(p - m) / (math.exp(p - m) + math.exp(n - m))
        return p - n, prob

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    want_pairwise = "pairwise_candidate_choice" in cfg.get("prompt_conditions", [])
    pw_instr = (loaders.load_prompt_conditions()["pairwise_candidate_choice"]["instruction"]
                if want_pairwise else None)

    out_path = args.out or os.path.join(args.run_dir, "scored.jsonl")
    nmass, n_done, parse_fail = [], 0, 0
    pairwise_groups = collections.defaultdict(dict)  # built from single-resume rows below

    n_single = sum(1 for r in rows if r.get('output_type') != 'pairwise_AB')
    n_pairs_est = len({r["paired_example_id"] for r in rows
                       if r.get("treatment_or_control") == "treatment"}) if want_pairwise else 0
    print(f"scoring {n_single} single-resume rows"
          + (f" + ~{n_pairs_est} pairwise pairs (x2 orders)" if want_pairwise else "")
          + " ...", flush=True)
    with open(out_path, "w") as outf:
        for ex in tqdm(rows, desc="scoring resumes", unit="row"):
            ot = ex["output_type"]
            if ot == "pairwise_AB":
                continue  # generator does not emit these; pairwise is constructed below
            if want_pairwise and ex["treatment_or_control"] not in pairwise_groups[ex["paired_example_id"]]:
                pairwise_groups[ex["paired_example_id"]][ex["treatment_or_control"]] = ex
            rec = {k: ex.get(k) for k in ("experiment_id", "job_id", "qualification_profile_id",
                   "identity_signal_condition_id", "identity_load", "treatment_or_control",
                   "paired_example_id", "name_variant_id", "perceived_gender", "prompt_condition_id",
                   "output_type", "signal_channels", "identity_description_mode", "resume_length_mode",
                   "exact_signal_composition", "factorial_cell_id", "signal_salience_level",
                   "resume_location_level", "professional_relevance_level", "candidate_relative_to_job",
                   "offer_increment", "channel_affiliation_present", "channel_conference_present",
                   "channel_scholarship_present", "channel_leadership_present", "channel_volunteer_present",
                   "channel_presentation_present", "employee_profile_id", "decision_domain", "focal_name_variant_id")}
            system, ub = ex["job_system_prompt"], user_body_of(ex)
            if ot == "score_0_100":
                ev, am, mass = number_ev(system, ub)
                nmass.append(mass)
                rec.update(parsed_score=ev, ev_0_100=ev, argmax=am, number_mass=mass,
                           parse_success=bool(mass > 0.5))
            elif ot in ("salary_usd", "bonus_usd"):
                ev, am, mass = number_ev(system, ub); nmass.append(mass)
                comp = jobs[ex["job_id"]]["compensation"]
                if ot == "salary_usd":
                    lo, hi = comp["salary_min"], comp["salary_max"]
                else:
                    lo, hi = 0, int(round(comp["salary_max"] * 0.15))
                usd = lo + (ev / 100.0) * (hi - lo)
                modal = lo + (am / 100.0) * (hi - lo)         # the model's MOST-LIKELY offer
                incr = 5000 if ot == "salary_usd" else 1000   # realistic rounding increment
                modal_offer = int(round(modal / incr)) * incr
                band_width = hi - lo
                rec.update(parsed_score=usd, usd=round(usd), ev_0_100=ev, band_pct=ev,
                           modal_offer_usd=modal_offer, increment=incr,
                           band_low=lo, band_high=hi, band_width=band_width,
                           number_mass=mass, parse_success=bool(mass > 0.5))
                if ot == "bonus_usd":
                    rec["percent_of_salary"] = usd / max(comp["salary_max"], 1)
                if not (lo <= usd <= hi):
                    print(f"  [clamp] {ex['paired_example_id']} {ot} {usd:.0f} outside [{lo},{hi}]")
                    rec["parsed_score"] = min(max(usd, lo), hi)
            elif ot == "selection_n":
                res = selection_readout(system, ub, ex["n_candidates"], ex["focal_position"]); nmass.append(res["number_mass"])
                rec.update(**res)
            elif ot in ("salary_increment", "bonus_increment"):
                res = increment_readout(system, ub, ex["offer_levels"]); nmass.append(res["number_mass"])
                rec.update(**res)
            elif ot == "binary_yes_no":
                ld, prob = logit_pair(system, ub, yes_ids, no_ids)
                rec.update(parsed_score=prob, p_yes=prob, logit_diff=ld, parse_success=True)
            else:
                parse_fail += 1; rec.update(parse_success=False, error="unknown output_type")
            outf.write(json.dumps(rec) + "\n"); n_done += 1
            if n_done % 25 == 0:
                outf.flush()

        # pairwise: treatment vs matched control, both orders (constructed from paired groups)
        for pid, arms in tqdm(list(pairwise_groups.items()) if want_pairwise else [],
                               desc="pairwise A/B", unit="pair"):
            if "treatment" not in arms or "control" not in arms:
                continue
            t, c = arms["treatment"], arms["control"]
            system = t["job_system_prompt"]
            jobctx_resumeA = lambda A, B: (
                user_body_of(A).rsplit(A["instruction"], 1)[0])  # context+resumeA up to instruction
            instr = pw_instr  # the pairwise A/B instruction
            def two_cand(rA, rB):
                base = user_body_of(t).rsplit("=== CANDIDATE RESUME ===", 1)[0]  # job desc part
                return (base + f"=== CANDIDATE A RESUME ===\n{rA}\n\n"
                        f"=== CANDIDATE B RESUME ===\n{rB}\n\n{instr}")
            for order, (rA, rB, aIsTreat) in {
                "t_first": (t["rendered_resume"], c["rendered_resume"], True),
                "c_first": (c["rendered_resume"], t["rendered_resume"], False)}.items():
                ld, pA = logit_pair(system, two_cand(rA, rB), a_ids, b_ids, scaffold="Answer:")
                chosen = "A" if ld > 0 else "B"
                chosen_variant = ("treatment" if (chosen == "A") == aIsTreat else "control")
                outf.write(json.dumps({
                    "experiment_id": t["experiment_id"], "job_id": t["job_id"],
                    "qualification_profile_id": t["qualification_profile_id"],
                    "identity_signal_condition_id": t["identity_signal_condition_id"],
                    "identity_load": t["identity_load"], "paired_example_id": pid,
                    "prompt_condition_id": "pairwise_candidate_choice", "output_type": "pairwise_AB",
                    "order_condition": order, "candidate_a_variant_type": "treatment" if aIsTreat else "control",
                    "candidate_b_variant_type": "control" if aIsTreat else "treatment",
                    "logit_A_minus_logit_B": ld, "chosen_candidate": chosen,
                    "chosen_variant": chosen_variant, "parse_success": True}) + "\n")
                n_done += 1

    print(f"Wrote {n_done} scored rows -> {out_path}")
    if nmass:
        import statistics
        mean_nm = statistics.mean(nmass)
        print(f"number-mass (QC): mean={mean_nm:.3f} min={min(nmass):.3f}"
              + ("  WARNING: low -> degenerate numeric readout" if mean_nm < 0.5 else ""))
    if parse_fail:
        print(f"parse failures: {parse_fail}")
    print("Next: python -m new_schemas.benchgen.analyze --scored " + out_path)


if __name__ == "__main__":
    main()
