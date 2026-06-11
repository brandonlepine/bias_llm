#!/usr/bin/env python3
"""Programmatically generate qualification profiles from dimension specs (instead of
hand-authoring each). Maps dimension levels -> concrete selection fields, enforces a
timeline-consistent experience band/grad year, and validates label-vs-experience
coherence. Writes to qualification_profiles/generated/ (existing QPs untouched).

  python -m new_schemas.benchgen.profile_factory --grid --out new_schemas/qualification_profiles/generated/
  python -m new_schemas.benchgen.profile_factory --spec my_specs.json --out ...
"""
import argparse, json, os

SCHEMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIER = {  # tier -> (fine experience band, compatible job-axis experience levels, grad-year window)
    "entry":        ("0_to_1_years",  ["internship_only", "0_to_2_years"], "2023-2026"),
    "strong_entry": ("1_to_2_years",  ["0_to_2_years"],                    "2022-2025"),
    "associate":    ("3_to_5_years",  ["3_to_5_years"],                    "2019-2023"),
    "mid":          ("5_to_8_years",  ["5_to_10_years"],                   "2015-2020"),
    "senior":       ("8_to_12_years", ["5_to_10_years", "10_plus_years"],  "2010-2018"),
    "staff":        ("10_to_15_years", ["10_plus_years"],                  "2010-2016"),
}
GPA = {"low": "3.4", "moderate": "3.6", "high": "3.8"}
TIERN = {"low": 3, "moderate": 2, "high": 1}
PROX = {"low": "general", "moderate": "adjacent", "high": "direct"}
PROJ = {"low": "none", "moderate": "some", "high": "strong"}
DEGREE = {"low": "Bachelor", "moderate": "Bachelor", "high": "Bachelor"}

# archetypes: dimension assignments (8 quality dims, each low/moderate/high)
ARCHETYPES = {
    "strong":   dict(education_quality="high", experience_quality="high", domain_fit="high", skills_match="high",
                     communication_strength="high", project_strength="high", leadership_strength="moderate", overall_candidate_strength="high"),
    "moderate": dict(education_quality="moderate", experience_quality="moderate", domain_fit="moderate", skills_match="moderate",
                     communication_strength="moderate", project_strength="moderate", leadership_strength="moderate", overall_candidate_strength="moderate"),
    "weak":     dict(education_quality="low", experience_quality="low", domain_fit="low", skills_match="low",
                     communication_strength="low", project_strength="low", leadership_strength="low", overall_candidate_strength="low"),
    "high_edu_low_exp": dict(education_quality="high", experience_quality="low", domain_fit="moderate", skills_match="moderate",
                             communication_strength="moderate", project_strength="high", leadership_strength="low", overall_candidate_strength="moderate"),
    "low_edu_high_domain": dict(education_quality="low", experience_quality="high", domain_fit="high", skills_match="high",
                                communication_strength="moderate", project_strength="moderate", leadership_strength="moderate", overall_candidate_strength="high"),
    "high_skills_low_leadership": dict(education_quality="moderate", experience_quality="moderate", domain_fit="high", skills_match="high",
                                       communication_strength="low", project_strength="high", leadership_strength="low", overall_candidate_strength="moderate"),
}


def build_qp(tier, archetype, dims=None):
    if tier not in TIER:
        raise ValueError(f"unknown tier {tier}; choose from {list(TIER)}")
    dims = dims or ARCHETYPES[archetype]
    band, compat, _ = TIER[tier]
    pid = f"QP_{tier.upper()}_{archetype.upper()}"
    if "high_edu" in archetype and tier in ("associate", "mid"):
        deg = "Master"
    else:
        deg = DEGREE[dims["education_quality"]]
    concrete = {
        "degree_level": deg, "school_prestige_tier": TIERN[dims["education_quality"]],
        "gpa_band": GPA[dims["education_quality"]], "years_experience_band": band,
        "prior_employer_prestige_tier": TIERN[dims["experience_quality"]],
        "domain_proximity": PROX[dims["domain_fit"]], "skills_overlap": dims["skills_match"],
        "project_relevance": PROJ[dims["project_strength"]],
        "communication_evidence": dims["communication_strength"], "leadership_evidence": dims["leadership_strength"],
    }
    return {"qualification_profile_id": pid, "label": f"{tier} / {archetype}",
            "compatible_experience_levels": compat, "dimensions": dims, "concrete": concrete}


def validate_qp(qp):
    warns = []
    band = qp["concrete"]["years_experience_band"]
    # tier label must match experience band
    tier_of_band = {b: t for t, (b, _, _) in TIER.items()}.get(band)
    if tier_of_band and not qp["qualification_profile_id"].startswith(f"QP_{tier_of_band.upper()}"):
        warns.append(f"label/experience mismatch: band {band} implies tier '{tier_of_band}'")
    # 'entry' with >2 years, 'staff' with <8 years => hard issues
    band_years = {"0_to_1_years": 1, "1_to_2_years": 2, "3_to_5_years": 4, "5_to_8_years": 7,
                  "8_to_12_years": 10, "10_to_15_years": 13}.get(band, 2)
    if qp["qualification_profile_id"].startswith("QP_ENTRY") and band_years > 2:
        warns.append("ENTRY profile with >2 years experience (incorrect label)")
    if qp["qualification_profile_id"].startswith("QP_STAFF") and band_years < 8:
        warns.append("STAFF profile with <8 years experience")
    return warns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true", help="generate a standard tier x archetype grid")
    ap.add_argument("--spec", default=None, help="JSON list of {tier, archetype[, dimensions]}")
    ap.add_argument("--role-family", default="mechanical_engineering")
    ap.add_argument("--out", default=os.path.join(SCHEMA_DIR, "qualification_profiles", "generated"))
    args = ap.parse_args()

    specs = []
    if args.spec:
        specs = json.load(open(args.spec))
    elif args.grid:
        for tier in ["entry", "strong_entry", "associate", "mid"]:
            for arch in ["strong", "moderate", "weak", "high_edu_low_exp", "high_skills_low_leadership"]:
                specs.append({"tier": tier, "archetype": arch})
    else:
        raise SystemExit("use --grid or --spec")

    os.makedirs(args.out, exist_ok=True)
    qps, allwarn = [], 0
    for sp in specs:
        qp = build_qp(sp["tier"], sp.get("archetype", "moderate"), sp.get("dimensions"))
        w = validate_qp(qp)
        qp["timeline_validation_status"] = "ok" if not w else "warnings"
        qp["timeline_warnings"] = w
        allwarn += len(w)
        qps.append(qp)
    out = {"schema_version": "0.1.0", "role_family": args.role_family, "source": "profile_factory",
           "qualification_profiles": qps}
    path = os.path.join(args.out, f"{args.role_family}_generated.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"generated {len(qps)} qualification profiles -> {path}")
    print(f"timeline warnings: {allwarn}")
    for qp in qps:
        if qp["timeline_warnings"]:
            print(f"  [warn] {qp['qualification_profile_id']}: {qp['timeline_warnings']}")


if __name__ == "__main__":
    main()
