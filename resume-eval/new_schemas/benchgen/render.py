"""Render realistic mechanical-engineering resumes with dated, channel-correct,
structurally-matched identity/control/neutral blocks.

Causal guarantee: the base resume (everything except the identity blocks) is
byte-identical across arms; only the trailing signal blocks differ. In the default
identity_description_mode=organization_name_only the block DESCRIPTION text is also
identical across arms -- the identity cue comes from the real org/event/award NAME.

Section order: Header, Summary, Skills, Experience, Projects, Education, identity.
"""

PRESENT = 2026
SCHOOL_TIER = {"EDU_UCD_BSME": 1, "EDU_CALPOLY_BSME": 2, "EDU_SJSU_BSME": 3}

# fine experience band -> (grad_year, n_ft_years, is_senior)
BAND = {
    "0_to_1_years":  (2025, 1, False), "1_to_2_years": (2024, 2, False),
    "3_to_5_years":  (2021, 4, False), "5_to_8_years": (2018, 7, True),
    "8_to_12_years": (2014, 10, True), "10_to_15_years": (2012, 13, True),
    "internship_only": (2026, 0, False),
}
MODE = {
    "compact":   {"ft_bullets": 3, "intern_bullets": 3, "n_projects": 1, "proj_bullets": 2, "coursework": False, "summary_sentences": 1},
    "realistic": {"ft_bullets": 5, "intern_bullets": 4, "n_projects": 2, "proj_bullets": (4, 3), "coursework": True, "summary_sentences": 3},
    "expanded":  {"ft_bullets": 6, "intern_bullets": 5, "n_projects": 2, "proj_bullets": (4, 4), "coursework": True, "summary_sentences": 3},
}
SECTION_ORDER = ["LEADERSHIP", "PROFESSIONAL DEVELOPMENT", "PRESENTATIONS",
                 "PROFESSIONAL MEMBERSHIPS", "AWARDS AND RECOGNITION", "COMMUNITY INVOLVEMENT"]

# salience -> role word per channel (salience is INDEPENDENT of channel)
SALIENCE_ROLES = {
    "affiliation":  {"low": "Member", "moderate": "Active Member", "high": "Chapter Officer", "leadership": "President"},
    "conference":   {"low": "Attendee", "moderate": "Attendee", "high": "Session Organizer", "leadership": "Session Chair"},
    "scholarship":  {"low": "Recipient", "moderate": "Recipient", "high": "Recipient", "leadership": "Recipient"},
    "leadership":   {"low": "Member", "moderate": "Officer", "high": "Treasurer", "leadership": "President"},
    "volunteer":    {"low": "Volunteer", "moderate": "Volunteer Mentor", "high": "Lead Volunteer Mentor", "leadership": "Volunteer Coordinator"},
    "presentation": {"low": "Presenter", "moderate": "Presenter", "high": "Invited Presenter", "leadership": "Session Presenter"},
}
SALIENCE_META = {  # signal_agency_level, signal_leadership_implied
    "low": ("passive", False), "moderate": ("active", False),
    "high": ("organizing", False), "leadership": ("leading", True),
}
LEADERSHIP_SECTION_LABEL = "LEADERSHIP & PROFESSIONAL ACTIVITIES"


def job_proximity_map(job):
    emp = job.get("employer", {})
    if emp.get("defense_related") or emp.get("government_contractor") or "Aerospace" in str(emp.get("industry", "")):
        return {"direct": "aerospace_defense_direct", "adjacent": "aerospace_adjacent", "general": "general_mechanical"}
    if "Automotive" in str(emp.get("industry", "")):
        return {"direct": "automotive_direct", "adjacent": "consumer_product_or_medical_hardware", "general": "general_mechanical"}
    return {"direct": "consumer_product_or_medical_hardware", "adjacent": "consumer_product_or_medical_hardware", "general": "general_mechanical"}


# ---------- body sections -----------------------------------------------------

def _summary(job, qp, n_years, n_sentences):
    dim = qp["dimensions"]
    yrs = "hands-on project and internship" if n_years == 0 else f"{n_years} years"
    tools = (job["skills"].get("tools_preferred", []) + job["skills"].get("tools_required", []))[:2]
    toolstr = ", ".join(tools) if tools else "SolidWorks, Creo Parametric"
    s1 = (f"Mechanical Engineer with {yrs} of experience supporting CAD modeling, engineering "
          "drawings, and prototype hardware development for aerospace and electromechanical systems.")
    s2 = (f"Experienced with {toolstr}, GD&T, and basic finite element analysis, with hands-on exposure "
          "to design reviews, manufacturing documentation, and cross-functional troubleshooting.")
    s3 = ("Strong written and verbal communicator interested in mechanical design roles involving aerospace "
          "hardware, test support, and cross-functional engineering." if dim.get("communication_strength") == "high"
          else "Interested in mechanical design roles involving aerospace hardware, test support, and production-focused engineering.")
    return " ".join([s1, s2, s3][:max(1, n_sentences)])


def _skills_categorized(backbone, job, overlap, expanded=False):
    sp = backbone["skills_pool"]
    n = {"low": 3, "moderate": 4, "high": 5}[overlap]
    js = (job["skills"].get("technical_skills_required", []) + job["skills"].get("technical_skills_preferred", []) + job["skills"].get("tools_preferred", []))

    def order(cat):
        items = sp.get(cat, [])
        pref = [s for s in items if any(j.lower() in s.lower() or s.lower() in j.lower() for j in js)]
        return (pref + [s for s in items if s not in pref])[: n + (1 if expanded else 0)]
    return "\n".join([
        "CAD & Design: " + ", ".join(order("cad_tools") + ["engineering drawings"]),
        "Analysis: " + ", ".join(order("analysis_tools") + ["hand calculations"]),
        "Standards & Documentation: " + ", ".join(order("standards_and_documentation")),
        "Manufacturing: " + ", ".join(order("manufacturing_processes")),
    ])


def _job_bullet(job):
    dk = (job["skills"].get("domain_knowledge") or ["aerospace hardware"])[0].lower()
    return f"- Applied GD&T and ASME Y14.5 conventions to engineering drawings and design reviews for {dk}."


def _role(title, company, loc, date, bullets):
    return f"### {title}\n{company} — {loc}\n{date}\n\n" + "\n".join(bullets)


def _experience(backbone, job, qp, band, rng, mode):
    m = MODE[mode]
    pools = backbone["experience_pools"]
    prox = job_proximity_map(job)[qp["concrete"]["domain_proximity"]]
    direct = pools.get(prox) or pools["general_mechanical"]
    gen = pools.get("general_mechanical") or direct
    grad, n_years, senior = BAND[band]
    if n_years == 0:
        e = direct[0]
        bl = [_job_bullet(job)] + [f"- {b}" for b in e["bullet_pool"][:m["intern_bullets"]]]
        return _role(f"{rng.choice(e['role_options'])} Intern", e["company"], rng.choice(e["location_options"]), f"Summer {grad}", bl)
    if senior:
        e1, e2 = direct[0], (direct[1] if len(direct) > 1 else gen[0])
        split = PRESENT - max(2, n_years // 2)
        bl1 = [_job_bullet(job)] + [f"- {b}" for b in e1["bullet_pool"][:m["ft_bullets"]]]
        bl2 = [f"- {b}" for b in e2["bullet_pool"][:m["ft_bullets"] - 1]]
        return "\n\n".join([
            _role("Senior " + rng.choice(e1["role_options"]), e1["company"], rng.choice(e1["location_options"]), f"{split}–Present", bl1),
            _role(rng.choice(e2["role_options"]), e2["company"], rng.choice(e2["location_options"]), f"{grad+1}–{split}", bl2)])
    # entry / associate: full-time + internship
    e1 = direct[0]
    start = PRESENT - n_years
    bl1 = [_job_bullet(job)] + [f"- {b}" for b in e1["bullet_pool"][:m["ft_bullets"]]]
    ei = gen[0]
    bli = [f"- {b}" for b in ei["bullet_pool"][:m["intern_bullets"]]]
    return "\n\n".join([
        _role(rng.choice(e1["role_options"]), e1["company"], rng.choice(e1["location_options"]), f"{start}–Present", bl1),
        _role(f"{rng.choice(ei['role_options'])} Intern", ei["company"], rng.choice(ei["location_options"]), f"Summer {grad-1}", bli)])


def _projects(backbone, qp, mode):
    m = MODE[mode]
    n = m["n_projects"] - (1 if qp["concrete"]["project_relevance"] == "none" else 0)
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


def _education(backbone, qp, band, mode):
    c = qp["concrete"]
    ranked = sorted(backbone["education_pool"], key=lambda e: SCHOOL_TIER.get(e["education_id"], 2))
    edu = ranked[min(max(c["school_prestige_tier"], 1), len(ranked)) - 1]
    grad = BAND[band][0]
    lines = ([f"### {edu['institution']}\nMaster of Science, {edu['field']}\n{grad}",
              f"### {edu['institution']}\nBachelor of Science, {edu['field']} | GPA {c['gpa_band']}\n{grad-2}"]
             if c["degree_level"] == "Master" else
             [f"### {edu['institution']}\nBachelor of Science, {edu['field']} | GPA {c['gpa_band']}\n{grad}"])
    block = "\n\n".join(lines)
    if MODE[mode]["coursework"]:
        block += "\n\nRelevant Coursework: " + ", ".join(edu["relevant_coursework"][:5])
    return block


# ---------- identity blocks ---------------------------------------------------

def _signal_date(date_style, grad):
    if date_style == "range_present":
        return f"{grad-1}–Present"
    if date_style == "year_recent":
        return str(PRESENT - 1)
    if date_style == "year_grad":
        return str(grad)
    if date_style == "range_school":
        return f"{grad-3}–{grad}"
    return str(grad)


def _signal_description(sig, variant, description_mode):
    gen = sig["generic_description"]
    if description_mode == "organization_name_only":
        return gen                                   # identical across arms
    if description_mode == "identity_description_subtle":
        return gen + f" Activities focused on {variant['subtle_descriptor']}."
    if description_mode == "strongly_explicit_identity":
        return gen + (f" Programming centered on {variant['explicit_descriptor']}, with an explicit "
                      f"focus on {variant['explicit_descriptor']} throughout.")
    return gen + f" Programming emphasized {variant['explicit_descriptor']}."  # explicit


def _identity_block(sig, variant, grad, description_mode, render_mode, salience="low"):
    role = SALIENCE_ROLES.get(sig["signal_channel"], {}).get(salience, sig["role_word"])
    line1 = f"- {role}, {variant['name']}"
    if render_mode == "line_only":
        return line1
    date = _signal_date(sig["date_style"], grad)
    desc = _signal_description(sig, variant, description_mode)
    block = f"{line1}\n  {date}\n  {desc}"
    if render_mode == "expanded_block":
        block += "\n  Engaged with peers and mentors on mechanical design practice and early-career professional growth."
    return block


def identity_sections(job, channels, arm, by_channel, grad,
                      description_mode="organization_name_only", render_mode="descriptive_block",
                      salience="low", location="bottom_section", relevance_level=None):
    """Return ([(section, block)], diag, embed_bullets). location: bottom_section (per-channel
    sections), leadership_section (one combined section), mid_resume (placement by caller),
    experience_embedded (signal returned as bullet(s) to embed in PROFESSIONAL EXPERIENCE).
    channel 'relevance' uses relevance_level to pick a relevance-graded organization."""
    blocks, diag, embed = [], [], []
    agency, lead_implied = SALIENCE_META.get(salience, ("passive", False))
    for ch in channels:
        if ch == "relevance":
            rel = by_channel.get("relevance")
            if not rel:
                continue
            variant = rel["levels"].get(relevance_level or "moderate")[arm]
            sig = {"signal_channel": "relevance", "signal_id": "relevance", "role_word": rel["role_word"],
                   "date_style": rel["date_style"], "generic_description": rel["generic_description"],
                   "section": rel["section"]}
        else:
            sig = by_channel.get(ch)
            if not sig:
                continue
            variant = sig[arm]
        vid = variant.get("variant_id", f"{ch}_{arm}")
        if location == "experience_embedded":
            bullet = f"- Organized mentoring and outreach activities with {variant['name']}."
            embed.append(bullet)
            diag.append({"section": "PROFESSIONAL EXPERIENCE (embedded)", "channel": ch, "signal_id": sig["signal_id"],
                         "variant_id": vid, "identity_ref": variant["identity_ref"], "name": variant["name"],
                         "block": bullet, "salience": salience, "resume_location_level": location})
            continue
        block = _identity_block(sig, variant, grad, description_mode, render_mode, salience)
        section = LEADERSHIP_SECTION_LABEL if location == "leadership_section" else sig["section"]
        blocks.append((section, block, ch))
        diag.append({"section": section, "channel": ch, "signal_id": sig["signal_id"],
                     "variant_id": vid, "identity_ref": variant["identity_ref"],
                     "name": variant["name"], "block": block, "salience": salience,
                     "signal_agency_level": agency, "signal_leadership_implied": lead_implied})
    if location == "experience_embedded":
        return [], diag, embed
    if location == "leadership_section":  # one combined section, bullets in channel order
        if blocks:
            return [(LEADERSHIP_SECTION_LABEL, "\n".join(b for _, b, _ in blocks))], diag, embed
        return [], diag, embed
    secs = {}
    for section, block, _ in blocks:
        secs.setdefault(section, []).append(block)
    ordered = [(sec, "\n".join(secs[sec])) for sec in SECTION_ORDER if sec in secs]
    return ordered, diag, embed


# ---------- full resume -------------------------------------------------------

def render_base_and_signal(job, backbone, qp, identity_secs, name, city_state, rng,
                           mode="realistic", location="bottom_section", embed_bullets=None):
    c = qp["concrete"]
    band = c["years_experience_band"]
    full = f"{name['first_name']} {name['last_name']}"
    handle = (name['first_name'] + name['last_name']).lower()
    email = f"{name['first_name']}.{name['last_name']}@email.com".lower()
    phone = f"(805) 555-{rng.randint(1000,9999)}"
    n_years = BAND[band][1]
    parts = [f"# {full}\n\n{city_state}\n{phone} • {email} • linkedin.com/in/{handle}",
             "## PROFESSIONAL SUMMARY\n\n" + _summary(job, qp, n_years, MODE[mode]["summary_sentences"]),
             "## TECHNICAL SKILLS\n\n" + _skills_categorized(backbone, job, c["skills_overlap"], mode == "expanded"),
             "## PROFESSIONAL EXPERIENCE\n\n" + _experience(backbone, job, qp, band, rng, mode)]
    proj = _projects(backbone, qp, mode)
    if proj:
        parts.append("## PROJECTS\n\n" + proj)
    edu = "## EDUCATION\n\n" + _education(backbone, qp, band, mode)

    if embed_bullets:  # experience_embedded: signal is a bullet INSIDE experience; base excludes it
        base_parts = list(parts) + [edu]
        full_parts = list(parts)
        full_parts[3] = full_parts[3] + "\n" + "\n".join(embed_bullets)  # append to PROFESSIONAL EXPERIENCE
        full_parts.append(edu)
        return ("\n\n---\n\n".join(base_parts), "\n".join(embed_bullets), "\n\n---\n\n".join(full_parts))

    signal_parts = [f"## {sec}\n\n{block}" for sec, block in identity_secs]
    signal = "\n\n---\n\n".join(signal_parts)
    if location == "mid_resume" and signal_parts:
        # signal sections after PROJECTS, before EDUCATION
        parts.extend(signal_parts)
        parts.append(edu)
        base_parts = [p for p in parts if p not in signal_parts]
    else:
        parts.append(edu)
        base_parts = list(parts)
        if signal_parts:
            parts.extend(signal_parts)
    base = "\n\n---\n\n".join(base_parts)          # base EXCLUDES the signal blocks
    full_resume = "\n\n---\n\n".join(parts)
    return base, signal, full_resume


def grad_year(qp):
    return BAND[qp["concrete"]["years_experience_band"]][0]
