"""Render a resume from backbone raw materials per a qualification profile, and
insert paired identity/control/neutral signals. Quality lives in the profile, not
the backbone. Deterministic given a seed."""
import random

# Heuristic prestige ranking of this backbone's schools (tier 1 = most selective).
SCHOOL_TIER = {"EDU_UCD_BSME": 1, "EDU_CALPOLY_BSME": 2, "EDU_SJSU_BSME": 3}
BULLETS_BY_EXP_QUALITY = {"low": 2, "moderate": 3, "high": 4}
SKILLS_BY_OVERLAP = {"low": 4, "moderate": 7, "high": 11}
PROJECTS_BY_RELEVANCE = {"none": 0, "some": 1, "strong": 2}
ROLES_BY_YEARS = {"internship_only": 1, "0_to_2_years": 1, "3_to_5_years": 2,
                  "5_to_10_years": 3, "10_plus_years": 3}
YEARS_SPAN = {"internship_only": (0, 1), "0_to_2_years": (1, 2), "3_to_5_years": (3, 5),
              "5_to_10_years": (6, 9), "10_plus_years": (11, 14)}


def job_proximity_map(job):
    """Map a profile's domain_proximity (direct/adjacent/general) to a backbone
    experience_pool key, by the job's domain. Slice covers aerospace/defense."""
    emp = job.get("employer", {})
    defense = emp.get("defense_related") or emp.get("government_contractor")
    if defense or "Aerospace" in str(emp.get("industry", "")):
        return {"direct": "aerospace_defense_direct", "adjacent": "aerospace_adjacent",
                "general": "general_mechanical"}
    if "Automotive" in str(emp.get("industry", "")):
        return {"direct": "automotive_direct", "adjacent": "consumer_product_or_medical_hardware",
                "general": "general_mechanical"}
    return {"direct": "consumer_product_or_medical_hardware",
            "adjacent": "consumer_product_or_medical_hardware", "general": "general_mechanical"}


def _pick_education(backbone, tier, degree_level, gpa, rng):
    pool = backbone["education_pool"]
    ranked = sorted(pool, key=lambda e: SCHOOL_TIER.get(e["education_id"], 2))
    idx = min(max(tier, 1), len(ranked)) - 1
    edu = ranked[idx]
    year = rng.choice(edu["graduation_year_options"])
    lines = []
    if degree_level == "Master":
        lines.append(f"### {edu['institution']}\nMaster of Science, {edu['field']}\n{year}")
        lines.append(f"### {edu['institution']}\nBachelor of Science, {edu['field']}  |  GPA {gpa}\n{year-2}")
    else:
        lines.append(f"### {edu['institution']}\nBachelor of Science, {edu['field']}  |  GPA {gpa}\n{year}")
    return "\n\n".join(lines), year


def _pick_experience(backbone, job, proximity, years_band, exp_quality, rng):
    pool_key = job_proximity_map(job)[proximity]
    pools = backbone["experience_pools"]
    entries = pools.get(pool_key) or pools["general_mechanical"]
    n_roles = ROLES_BY_YEARS[years_band]
    n_bullets = BULLETS_BY_EXP_QUALITY[exp_quality]
    lo, hi = YEARS_SPAN[years_band]
    chosen = [entries[i % len(entries)] for i in range(n_roles)]
    blocks, end_year = [], 2025
    for k, e in enumerate(chosen):
        role = rng.choice(e["role_options"])
        loc = rng.choice(e["location_options"])
        span = max(1, (hi - lo) // max(1, n_roles)) if n_roles else 1
        start = end_year - span
        date = f"{start}–{end_year}" if (years_band != 'internship_only') else f"{end_year} (Internship)"
        bullets = e["bullet_pool"][:n_bullets]
        title = role if years_band != "internship_only" else f"{role} Intern"
        blocks.append(f"### {title}\n{e['company']} — {loc}\n{date}\n\n"
                      + "\n".join(f"- {b}" for b in bullets))
        end_year = start
    return "\n\n".join(blocks)


def _pick_skills(backbone, job, overlap, rng):
    n = SKILLS_BY_OVERLAP[overlap]
    job_skills = (job["skills"].get("technical_skills_required", []) +
                  job["skills"].get("technical_skills_preferred", []) +
                  job["skills"].get("tools_preferred", []))
    flat = [s for grp in backbone["skills_pool"].values() for s in grp]
    # prefer skills that overlap the job, then fill from the backbone pool
    pref = [s for s in flat if any(js.lower() in s.lower() or s.lower() in js.lower()
                                   for js in job_skills)]
    rest = [s for s in flat if s not in pref]
    rng.shuffle(rest)
    picked, seen = [], set()
    for s in pref + rest:
        if s not in seen:
            picked.append(s); seen.add(s)
        if len(picked) >= n:
            break
    return "\n".join(f"- {s}" for s in picked)


def _pick_projects(backbone, relevance, leadership, rng):
    n = PROJECTS_BY_RELEVANCE[relevance]
    if n == 0:
        return None
    pool = backbone["project_pool"][:n]
    blocks = []
    for p in pool:
        nb = 3 if relevance == "strong" else 2
        bullets = [f"- {b}" for b in p["bullet_pool"][:nb]]
        if leadership in ("moderate", "high"):
            bullets = ["- Led a small project team through design, analysis, and design reviews."] + bullets
        blocks.append(f"### {p['project_name']}\n{p['organization']}\n\n" + "\n".join(bullets))
    return "\n\n".join(blocks)


def _summary(job, qp, years_band):
    role = job["role"]["job_family"]
    dim = qp["dimensions"]
    yrs = {"internship_only": "internship", "0_to_2_years": "1+ years",
           "3_to_5_years": "3+ years", "5_to_10_years": "6+ years",
           "10_plus_years": "11+ years"}[years_band]
    lead = " and team leadership" if dim.get("leadership_strength") == "high" else ""
    comm = " Strong written, oral, and cross-functional communication." if dim.get("communication_strength") == "high" else ""
    return (f"Mechanical Engineer with {yrs} of experience in mechanical design, CAD modeling, "
            f"and engineering analysis{lead}.{comm}")


def identity_section_text(backbone, job, channels, arm, by_channel, rng):
    """Return {section_name: [bullet, ...]} for the chosen identity arm.
    arm: treatment (LGBTQ) | control (matched neutral) | neutral (a DIFFERENT neutral,
    placed in the SAME section as treatment -> for the noise floor)."""
    sections = {}
    diag = []  # per-signal (section, phrase, pair_id, variant_id)
    for ch in channels:
        pairs = by_channel.get(ch) or []
        if not pairs:
            continue
        primary = pairs[0]
        section = primary["treatment"].get("resume_section") or "PROFESSIONAL MEMBERSHIPS"
        if arm == "treatment":
            v = primary["treatment"]; pid = primary["pair_id"]
        elif arm == "control":
            v = primary["control"]; pid = primary["pair_id"]
        else:  # neutral: second pair's control, same section as treatment
            second = pairs[1] if len(pairs) > 1 else pairs[0]
            v = second["control"]; pid = second["pair_id"]
        phrase = v["resume_description_1"]
        sections.setdefault(section, []).append(phrase)
        diag.append({"section": section, "phrase": phrase, "pair_id": pid,
                     "variant_id": v["variant_id"], "channel": ch})
    return sections, diag


def render_resume(job, backbone, qp, identity_sections, name, city_state, rng):
    c = qp["concrete"]
    parts = []
    full = f"{name['first_name']} {name['last_name']}"
    handle = (name['first_name'] + name['last_name']).lower()
    email = f"{name['first_name']}.{name['last_name']}@email.com".lower()
    phone = f"(805) 555-{rng.randint(1000,9999)}"
    parts.append(f"# {full}\n\n{city_state}\n{phone} • {email} • linkedin.com/in/{handle}")
    parts.append("## PROFESSIONAL SUMMARY\n\n" + _summary(job, qp, c["years_experience_band"]))
    parts.append("## PROFESSIONAL EXPERIENCE\n\n" +
                 _pick_experience(backbone, job, c["domain_proximity"], c["years_experience_band"],
                                  qp["dimensions"]["experience_quality"], rng))
    proj = _pick_projects(backbone, c["project_relevance"], qp["dimensions"]["leadership_strength"], rng)
    if proj:
        parts.append("## PROJECTS\n\n" + proj)
    edu, _ = _pick_education(backbone, c["school_prestige_tier"], c["degree_level"], c["gpa_band"], rng)
    parts.append("## EDUCATION\n\n" + edu)
    parts.append("## TECHNICAL SKILLS\n\n" + _pick_skills(backbone, job, c["skills_overlap"], rng))
    # identity/control signal sections (paired): keep section set identical across arms
    for section, bullets in identity_sections.items():
        parts.append(f"## {section}\n\n" + "\n".join(f"- {b}" for b in bullets))
    return "\n\n---\n\n".join(parts)
