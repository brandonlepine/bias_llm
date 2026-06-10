"""Schema loading + validation. Fail loudly on missing required fields."""
import json
import os

SCHEMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # new_schemas/


def _load(path):
    with open(path) as f:
        return json.load(f)


def _require(obj, fields, ctx):
    missing = [f for f in fields if f not in obj or obj[f] is None]
    if missing:
        raise ValueError(f"[schema] {ctx}: missing required field(s): {missing}")


def load_jobs():
    d = _load(os.path.join(SCHEMA_DIR, "job_descriptions", "hirequeer_resume_bias_pilot.json"))
    jobs = {}
    for j in d["job_postings"]:
        _require(j, ["job_id", "employer", "role", "location", "compensation",
                     "raw_posting_text", "resume_generation_axes"], "job")
        _require(j["compensation"], ["salary_min", "salary_max"], f"job {j['job_id']} compensation")
        jobs[j["job_id"]] = j
    return jobs


def load_backbones():
    d = _load(os.path.join(SCHEMA_DIR, "resume_backbones", "res_backbones_mech_e.json"))
    bk = {}
    for b in d["resume_backbones"]:
        _require(b, ["resume_backbone_id", "education_pool", "experience_pools",
                     "project_pool", "skills_pool"], "backbone")
        bk[b["resume_backbone_id"]] = b
    return bk


def load_qual_profiles():
    d = _load(os.path.join(SCHEMA_DIR, "qualification_profiles", "qualification_profiles.json"))
    qp = {}
    for p in d["qualification_profiles"]:
        _require(p, ["qualification_profile_id", "dimensions", "concrete",
                     "compatible_experience_levels"], "qualification_profile")
        qp[p["qualification_profile_id"]] = p
    return qp


def load_identity_pairs():
    d = _load(os.path.join(SCHEMA_DIR, "identity_signals", "identity_signals_engineering.json"))
    by_channel = {}
    for sig in d["identity_signals"]:
        _require(sig, ["signal_id", "signal_channel", "section", "role_word",
                       "generic_description", "treatment", "control", "neutral"], "identity_signal")
        by_channel[sig["signal_channel"]] = sig  # one triple per channel
    return by_channel


def load_names():
    d = _load(os.path.join(SCHEMA_DIR, "name_variants", "name_variants.json"))
    nv = {}
    for n in d["name_variants"]:
        _require(n, ["name_variant_id", "first_name", "last_name", "perceived_gender"], "name_variant")
        nv[n["name_variant_id"]] = n
    return nv


def load_prompt_conditions():
    d = _load(os.path.join(SCHEMA_DIR, "prompt_conditions", "prompt_conditions.json"))
    pc = {}
    for c in d["prompt_conditions"]:
        _require(c, ["prompt_condition_id", "instruction", "output_type", "readout"], "prompt_condition")
        pc[c["prompt_condition_id"]] = c
    return pc


def load_experiment(path):
    cfg = _load(path)
    _require(cfg, ["experiment_id", "jobs", "backbones", "qualification_profiles",
                   "identity_conditions", "name_variants", "prompt_conditions"], "experiment")
    return cfg
