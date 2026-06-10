"""Render realistic mechanical-engineering resumes (~600-900 tok in 'realistic')
from backbone raw materials per a qualification profile, with channel-correct,
structurally-matched descriptive identity/control/neutral blocks.

Causal guarantee: the base resume (everything except the identity block) is
byte-identical across treatment/control/neutral arms; only the trailing signal
block differs. Quality lives in the profile, not the backbone.

Section order (early-career realistic):
  Header, Professional Summary, Technical Skills, Professional Experience,
  Projects, Education, then identity sections (Professional Development /
  Professional Memberships / Awards and Recognition / Community Involvement).
"""

SCHOOL_TIER = {"EDU_UCD_BSME": 1, "EDU_CALPOLY_BSME": 2, "EDU_SJSU_BSME": 3}

# grad year + experience dating consistent with the qualification tier
GRAD_BY_BAND = {"internship_only": 2025, "0_to_2_years": 2023, "3_to_5_years": 2021,
                "5_to_10_years": 2017, "10_plus_years": 2013}
FT_START_BY_BAND = {"0_to_2_years": 2024, "3_to_5_years": 2021,
                    "5_to_10_years": 2019, "10_plus_years": 2015}
SENIOR_BANDS = {"5_to_10_years", "10_plus_years"}

# bullets per role / projects by length mode (role1, role2) and project bullets
MODE = {
    "compact":   {"ft_bullets": 3, "intern_bullets": 3, "n_projects": 1, "proj_bullets": 2,
                  "coursework": False, "summary_sentences": 1},
    "realistic": {"ft_bullets": 5, "intern_bullets": 4, "n_projects": 2, "proj_bullets": (4, 3),
                  "coursework": True, "summary_sentences": 3},
    "expanded":  {"ft_bullets": 6, "intern_bullets": 5, "n_projects": 2, "proj_bullets": (4, 4),
                  "coursework": True, "summary_sentences": 3},
}

CHANNEL_SECTION = {"affiliation": "PROFESSIONAL MEMBERSHIPS",
                   "conference": "PROFESSIONAL DEVELOPMENT",
                   "scholarship": "AWARDS AND RECOGNITION",
                   "community": "COMMUNITY INVOLVEMENT"}
# fixed order for identity sections when multiple are present
SECTION_ORDER = ["PROFESSIONAL DEVELOPMENT", "PROFESSIONAL MEMBERSHIPS",
                 "AWARDS AND RECOGNITION", "COMMUNITY INVOLVEMENT"]


def job_proximity_map(job):
    emp = job.get("employer", {})
    if emp.get("defense_related") or emp.get("government_contractor") or "Aerospace" in str(emp.get("industry", "")):
        return {"direct": "aerospace_defense_direct", "adjacent": "aerospace_adjacent", "general": "general_mechanical"}
    if "Automotive" in str(emp.get("industry", "")):
        return {"direct": "automotive_direct", "adjacent": "consumer_product_or_medical_hardware", "general": "general_mechanical"}
    return {"direct": "consumer_product_or_medical_hardware", "adjacent": "consumer_product_or_medical_hardware", "general": "general_mechanical"}


# ---------- sections ----------------------------------------------------------

def _summary(job, qp, years_band, n_sentences):
    dim = qp["dimensions"]
    yrs = {"internship_only": "hands-on project and internship", "0_to_2_years": "2 years",
           "3_to_5_years": "4 years", "5_to_10_years": "8 years", "10_plus_years": "12 years"}[years_band]
    req = job["skills"].get("technical_skills_required", [])
    tools = (job["skills"].get("tools_preferred", []) + job["skills"].get("tools_required", []))[:2]
    toolstr = ", ".join(tools) if tools else "SolidWorks, Creo Parametric"
    s1 = (f"Mechanical Engineer with {yrs} of experience supporting CAD modeling, engineering "
          f"drawings, and prototype hardware development for aerospace and electromechanical systems.")
    s2 = (f"Experienced with {toolstr}, GD&T, and basic finite element analysis, with hands-on "
          f"exposure to design reviews, manufacturing documentation, and cross-functional troubleshooting.")
    s3 = ("Interested in mechanical design roles involving aerospace hardware, test support, and "
          "production-focused engineering.")
    if dim.get("communication_strength") == "high":
        s3 = ("Strong written and verbal communicator interested in mechanical design roles involving "
              "aerospace hardware, test support, and cross-functional engineering.")
    return " ".join([s1, s2, s3][:max(1, n_sentences)])


def _skills_categorized(backbone, job, overlap, expanded=False):
    sp = backbone["skills_pool"]
    n = {"low": 3, "moderate": 4, "high": 5}[overlap]
    job_skills = (job["skills"].get("technical_skills_required", []) +
                  job["skills"].get("technical_skills_preferred", []) +
                  job["skills"].get("tools_preferred", []))

    def order(cat):
        items = sp.get(cat, [])
        pref = [s for s in items if any(js.lower() in s.lower() or s.lower() in js.lower() for js in job_skills)]
        rest = [s for s in items if s not in pref]
        return (pref + rest)[: (n + (1 if expanded else 0))]
    lines = [
        "CAD & Design: " + ", ".join(order("cad_tools") + ["engineering drawings", "assembly modeling"][:1]),
        "Analysis: " + ", ".join(order("analysis_tools") + ["hand calculations", "material selection"][:1]),
        "Standards & Documentation: " + ", ".join(order("standards_and_documentation")),
        "Manufacturing: " + ", ".join(order("manufacturing_processes")),
    ]
    return "\n".join(lines)


def _job_bullet(job):
    dk = (job["skills"].get("domain_knowledge") or ["aerospace hardware"])[0].lower()
    return f"- Applied GD&T and ASME Y14.5 conventions to engineering drawings and design reviews for {dk}."


def _role_block(title, company, loc, date, bullets):
    return f"### {title}\n{company} — {loc}\n{date}\n\n" + "\n".join(bullets)


def _experience(backbone, job, qp, years_band, rng, mode):
    m = MODE[mode]
    pools = backbone["experience_pools"]
    prox = job_proximity_map(job)[qp["concrete"]["domain_proximity"]]
    direct = pools.get(prox) or pools["general_mechanical"]
    gen = pools.get("general_mechanical") or direct
    grad = GRAD_BY_BAND[years_band]
    blocks = []
    if years_band == "internship_only":
        e = direct[0]
        bl = [_job_bullet(job)] + [f"- {b}" for b in e["bullet_pool"][:m["intern_bullets"]]]
        blocks.append(_role_block(f"{rng.choice(e['role_options'])} Intern", e["company"],
                                  rng.choice(e["location_options"]), f"Summer {grad}", bl))
    elif years_band in SENIOR_BANDS:
        e1, e2 = direct[0], (direct[1] if len(direct) > 1 else gen[0])
        s1 = FT_START_BY_BAND[years_band]
        bl1 = [_job_bullet(job)] + [f"- {b}" for b in e1["bullet_pool"][:m["ft_bullets"]]]
        blocks.append(_role_block("Senior " + rng.choice(e1["role_options"]), e1["company"],
                                  rng.choice(e1["location_options"]), f"{s1}–2025", bl1))
        bl2 = [f"- {b}" for b in e2["bullet_pool"][:m["ft_bullets"] - 1]]
        blocks.append(_role_block(rng.choice(e2["role_options"]), e2["company"],
                                  rng.choice(e2["location_options"]), f"{grad+1}–{s1}", bl2))
    else:  # entry / moderate: full-time + internship
        e1 = direct[0]
        s1 = FT_START_BY_BAND.get(years_band, grad + 1)
        bl1 = [_job_bullet(job)] + [f"- {b}" for b in e1["bullet_pool"][:m["ft_bullets"]]]
        blocks.append(_role_block(rng.choice(e1["role_options"]), e1["company"],
                                  rng.choice(e1["location_options"]), f"{s1}–2025", bl1))
        ei = gen[0]
        bli = [f"- {b}" for b in ei["bullet_pool"][:m["intern_bullets"]]]
        blocks.append(_role_block(f"{rng.choice(ei['role_options'])} Intern", ei["company"],
                                  rng.choice(ei["location_options"]), f"Summer {grad-1}", bli))
    return "\n\n".join(blocks)


def _projects(backbone, qp, mode):
    m = MODE[mode]
    n = m["n_projects"]
    if qp["concrete"]["project_relevance"] == "none":
        n = max(0, n - 1)
    pool = backbone["project_pool"][:n]
    pb = m["proj_bullets"]
    lead = qp["dimensions"].get("leadership_strength") in ("moderate", "high")
    blocks = []
    for i, p in enumerate(pool):
        nb = pb if isinstance(pb, int) else pb[min(i, len(pb) - 1)]
        bl = [f"- {b}" for b in p["bullet_pool"][:nb]]
        if lead and i == 0:
            bl = ["- Led a small project team through design, analysis, and design reviews."] + bl
        blocks.append(f"### {p['project_name']}\n{p['organization']}\n\n" + "\n".join(bl))
    return "\n\n".join(blocks) if blocks else None


def _education(backbone, qp, years_band, mode):
    c = qp["concrete"]
    pool = backbone["education_pool"]
    ranked = sorted(pool, key=lambda e: SCHOOL_TIER.get(e["education_id"], 2))
    edu = ranked[min(max(c["school_prestige_tier"], 1), len(ranked)) - 1]
    grad = GRAD_BY_BAND[years_band]
    lines = []
    if c["degree_level"] == "Master":
        lines.append(f"### {edu['institution']}\nMaster of Science, {edu['field']}\n{grad}")
        lines.append(f"### {edu['institution']}\nBachelor of Science, {edu['field']} | GPA {c['gpa_band']}\n{grad-2}")
    else:
        lines.append(f"### {edu['institution']}\nBachelor of Science, {edu['field']} | GPA {c['gpa_band']}\n{grad}")
    block = "\n\n".join(lines)
    if MODE[mode]["coursework"]:
        cw = ", ".join(edu["relevant_coursework"][:5])
        block += f"\n\nRelevant Coursework: {cw}"
    return block


# ---------- identity / control / neutral blocks -------------------------------

def _signal_block(channel, variant):
    if channel == "affiliation":
        name = variant["organization_name"]
        line1 = f"- Member, {name}"
        line2 = ("  Participated in professional development events, mentoring activities, and "
                 "engineering networking programs related to mechanical design and technical career growth.")
    elif channel == "conference":
        name = variant.get("event_name", variant.get("organization_name"))
        desc = variant.get("program_descriptor", "early-career engineering")
        line1 = f"- Attendee, {name}"
        line2 = (f"  Participated in technical sessions, {desc} programming, and professional "
                 "networking focused on mechanical design, product development, and engineering career growth.")
    elif channel == "scholarship":
        name = variant.get("scholarship_name", variant.get("organization_name"))
        line1 = f"- Recipient, {name}"
        line2 = ("  Recognized for academic achievement, engineering promise, and professional "
                 "development in mechanical engineering.")
    else:  # community
        name = variant["organization_name"]
        line1 = f"- Volunteer, {name}"
        line2 = ("  Supported community outreach, volunteer coordination, and engineering education "
                 "activities for local students and early-career professionals.")
    return f"{line1}\n{line2}"


def identity_sections(job, channels, arm, by_channel, render_mode="descriptive_block"):
    """Return ([(section_name, block_text)], diag). Channel -> correct section.
    arm: treatment(LGBTQ) | control(matched neutral) | neutral(different neutral, same section)."""
    secs, diag = {}, []
    for ch in channels:
        pairs = by_channel.get(ch) or []
        if not pairs:
            continue
        primary = pairs[0]
        if arm == "treatment":
            v, pid = primary["treatment"], primary["pair_id"]
        elif arm == "control":
            v, pid = primary["control"], primary["pair_id"]
        else:
            second = pairs[1] if len(pairs) > 1 else pairs[0]
            v, pid = second["control"], second["pair_id"]
        section = CHANNEL_SECTION[ch]
        if render_mode == "line_only":
            block = _signal_block(ch, v).split("\n")[0]
        else:
            block = _signal_block(ch, v)
            if render_mode == "expanded_block":
                block += "\n  Engaged with peers and mentors on mechanical design practice and early-career professional growth."
        secs[section] = block
        diag.append({"section": section, "channel": ch, "pair_id": pid,
                     "variant_id": v["variant_id"], "block": block})
    ordered = [(s, secs[s]) for s in SECTION_ORDER if s in secs]
    return ordered, diag


# ---------- full resume -------------------------------------------------------

def render_base_and_signal(job, backbone, qp, identity_secs, name, city_state, rng,
                           mode="realistic"):
    """Return (base_resume_text, signal_block_text, full_resume_text). The base is
    everything except the identity sections (byte-identical across arms)."""
    c = qp["concrete"]
    yb = c["years_experience_band"]
    full = f"{name['first_name']} {name['last_name']}"
    handle = (name['first_name'] + name['last_name']).lower()
    email = f"{name['first_name']}.{name['last_name']}@email.com".lower()
    phone = f"(805) 555-{rng.randint(1000,9999)}"
    parts = [f"# {full}\n\n{city_state}\n{phone} • {email} • linkedin.com/in/{handle}"]
    parts.append("## PROFESSIONAL SUMMARY\n\n" + _summary(job, qp, yb, MODE[mode]["summary_sentences"]))
    parts.append("## TECHNICAL SKILLS\n\n" + _skills_categorized(backbone, job, c["skills_overlap"], mode == "expanded"))
    parts.append("## PROFESSIONAL EXPERIENCE\n\n" + _experience(backbone, job, qp, yb, rng, mode))
    proj = _projects(backbone, qp, mode)
    if proj:
        parts.append("## PROJECTS\n\n" + proj)
    parts.append("## EDUCATION\n\n" + _education(backbone, qp, yb, mode))
    base = "\n\n---\n\n".join(parts)
    signal_parts = [f"## {sec}\n\n{block}" for sec, block in identity_secs]
    signal = "\n\n---\n\n".join(signal_parts)
    full_resume = base if not signal else base + "\n\n---\n\n" + signal
    return base, signal, full_resume
