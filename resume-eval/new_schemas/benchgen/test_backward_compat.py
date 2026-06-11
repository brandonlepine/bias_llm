#!/usr/bin/env python3
"""Backward-compatibility checks: existing hand-authored JSONs (no field_provenance)
still load and generate; auto + explicit backbone selection both work.

  python -m new_schemas.benchgen.test_backward_compat
"""
import random
from . import loaders, render, generate


def main():
    jobs = loaders.load_jobs(); bks = loaders.load_backbones()
    quals = loaders.load_qual_profiles(); names = loaders.load_names()
    pcs = loaders.load_prompt_conditions(); chans = loaders.load_identity_pairs()
    assert jobs and bks and quals and names and pcs and chans, "loaders returned empty"

    # existing job has NO field_provenance and still loads with required fields
    j = jobs["JOB_LOCKHEED_MECH_ENG_ENTRY_SUNNYVALE_2026"]
    assert "field_provenance" not in j, "existing job unexpectedly has provenance"
    assert j["compensation"]["salary_min"] and j["role"]["job_family"]

    # explicit + auto backbone selection
    fam_map = loaders.backbone_family_map(bks)
    assert generate.select_backbone(j, fam_map, bks) == "MECH_ENG_BACKBONE_001", "role-family selection failed"

    # fit scores + a real render of one resume (existing schema)
    qp = quals["QP_ASSOCIATE_STRONG_ALIGNED"]
    fit = generate.fit_scores(qp, j)
    assert 0 <= fit["overall_constructed_fit_score"] <= 1
    grad = render.grad_year(qp)
    secs, diag = render.identity_sections(j, ["affiliation"], "treatment", chans, grad)
    base, sig, full = render.render_base_and_signal(
        j, bks["MECH_ENG_BACKBONE_001"], qp, secs, names["NV_M_WHITE_01"], "Sunnyvale, California",
        random.Random("x"))
    assert "PROFESSIONAL EXPERIENCE" in full and sig, "render produced malformed resume"
    assert generate.candidate_relative_to_job(qp, j) in ("underqualified", "near_match", "strong_match", "overqualified")

    print("BACKWARD-COMPAT: PASS")
    print(f"  backbones={list(bks)}  jobs={len(jobs)}  quals={len(quals)}")
    print(f"  Lockheed -> {generate.select_backbone(j, fam_map, bks)}  fit={fit['overall_constructed_fit_score']}")


if __name__ == "__main__":
    main()
