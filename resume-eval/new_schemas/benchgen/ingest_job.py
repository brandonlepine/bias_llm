#!/usr/bin/env python3
"""Rule-based job-posting ingestion -> draft structured job JSON with provenance.

Extracts observable facts (regex/headings), rule-INFERS analytical metadata
(bias-relevant + latent dimensions + role family), records field_provenance
(extracted|inferred|manual + confidence + evidence), lists missing_fields, and
writes a human-review report. Never hallucinates exact values; inferred categoricals
are allowed but flagged. Writes to job_descriptions/drafts/ -- existing hand-authored
jobs are untouched. review_status='draft_unreviewed' until a human approves.

  python -m new_schemas.benchgen.ingest_job --raw-text path/to/job.txt \
      --company "Acme" --url "https://..." --extractor rule_based \
      --out new_schemas/job_descriptions/drafts/
"""
import argparse, json, os, re, datetime

SCHEMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKILL_LEXICON = [
    "SolidWorks", "Creo", "Creo Parametric", "AutoCAD", "Fusion 360", "CATIA", "NX",
    "GD&T", "ASME Y14.5", "Finite Element Analysis", "FEA", "ANSYS", "MATLAB",
    "tolerance analysis", "stress analysis", "3D modeling", "engineering drawings",
    "CAD", "PDM", "Windchill", "design for manufacturability", "sheet metal",
    "injection molding", "die casting", "machining", "mechanical design",
    "Python", "C++", "Java", "SQL", "machine learning", "data analysis",
]
ROLE_FAMILY_KEYWORDS = {
    "aerospace_engineering": ["aerospace", "spacecraft", "aircraft", "flight", "avionics"],
    "mechanical_engineering": ["mechanical engineer", "mechanical design", "gd&t", "cad", "solidworks", "creo"],
    "software_engineering": ["software engineer", "developer", "backend", "frontend", "full stack"],
    "data_science": ["data scientist", "machine learning", "ml engineer", "analytics"],
    "healthcare": ["nurse", "clinical", "patient", "medical device", "healthcare"],
    "product_management": ["product manager", "product management", "roadmap"],
}


def prov(source, conf, evidence):
    return {"source": source, "confidence": conf, "evidence": evidence}


def _search(rx, text, flags=re.I):
    m = re.search(rx, text, flags)
    return m


def extract(raw, company=None, url=None, role_family_override=None, date_accessed=None):
    text = raw
    P, missing = {}, []                       # field_provenance, missing_fields
    low = text.lower()

    # --- observable facts (extracted) ---
    salaries = [int(s.replace(",", "")) for s in re.findall(r"\$\s?([\d]{2,3}(?:,\d{3})+|\d{4,6})", text)]
    salaries = [s for s in salaries if 20000 <= s <= 600000]
    salary_min = min(salaries) if salaries else None
    salary_max = max(salaries) if salaries else None
    if salaries:
        P["compensation.salary_min"] = prov("extracted", 1.0, f"${salary_min:,}")
        P["compensation.salary_max"] = prov("extracted", 1.0, f"${salary_max:,}")
    else:
        missing += ["compensation.salary_min", "compensation.salary_max"]

    _STATES = "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
    loc = _search(r"\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+){0,2}),\s*(" + _STATES.replace(" ", "|") +
                  r"|California|Washington|Texas|New York|Florida)\b", text, re.I and 0)
    city = loc.group(1) if loc else None
    state = loc.group(2) if loc else None
    if loc: P["location.city"] = prov("extracted", 0.7, loc.group(0))
    else: missing.append("location.city")

    title = None
    tm = _search(r"(?:position|role|title)\s*[:\-]\s*(.+)", text) or _search(r"^([A-Z][\w \-/]{4,60}Engineer[\w \-/]*)$", text, re.I | re.M)
    if tm: title = tm.group(1).strip()[:80]
    if title: P["role.job_title"] = prov("extracted", 0.6, title)
    else: missing.append("role.job_title")

    degree = "Bachelor's degree" if re.search(r"bachelor", low) else ("Master's degree" if re.search(r"master", low) else None)
    if degree: P["requirements.minimum_education_level"] = prov("extracted", 0.9, degree)
    else: missing.append("requirements.minimum_education_level")

    ym = _search(r"(\d+)\+?\s*(?:or more\s*)?years", text)
    min_years = int(ym.group(1)) if ym else (0 if re.search(r"entry[- ]level|new grad|early career", low) else None)
    if ym or min_years == 0: P["requirements.minimum_years_experience"] = prov("extracted", 0.8, ym.group(0) if ym else "entry-level")
    else: missing.append("requirements.minimum_years_experience")

    clearance = bool(re.search(r"security clearance|clearance required|secret clearance", low))
    citizenship = bool(re.search(r"u\.?s\.? citizen|citizenship required|us person", low))
    arrangement = ("On-site" if re.search(r"on-?site", low) else "Remote" if re.search(r"\bremote\b", low)
                   else "Hybrid" if re.search(r"hybrid", low) else None)
    if clearance: P["requirements.clearance_required"] = prov("extracted", 0.95, "security clearance referenced")
    if citizenship: P["requirements.citizenship_required"] = prov("extracted", 0.9, "US citizenship referenced")
    if arrangement: P["location.work_arrangement"] = prov("extracted", 0.7, arrangement.lower())

    skills = sorted({s for s in SKILL_LEXICON if re.search(r"\b" + re.escape(s.lower()) + r"\b", low)})
    if skills: P["skills.technical_skills_required"] = prov("extracted", 0.6, f"{len(skills)} lexicon matches")
    else: missing.append("skills.technical_skills_required")

    # section split (best effort)
    def section(*keys):
        for k in keys:
            m = re.search(rf"{k}\s*[:\n](.{{0,1200}}?)(?:\n\s*\n|\Z)", text, re.I | re.S)
            if m: return m.group(1).strip()
        return None
    raw_sections = {
        "position_overview": section("overview", "about the role", "summary") or text[:600].strip(),
        "key_responsibilities": section("responsibilities", "what you.ll do", "duties"),
        "required_qualifications": section("required qualifications", "requirements", "qualifications", "must have"),
        "preferred_qualifications": section("preferred", "nice to have", "bonus"),
        "benefits": section("benefits", "compensation", "what we offer"),
    }

    # --- role family (inferred or override) ---
    if role_family_override:
        role_family = role_family_override
        P["role.job_family"] = prov("manual", 1.0, "override")
    else:
        scores = {f: sum(low.count(k) for k in kws) for f, kws in ROLE_FAMILY_KEYWORDS.items()}
        role_family = max(scores, key=scores.get) if any(scores.values()) else "generic_professional"
        P["role.job_family"] = prov("inferred", min(0.9, 0.5 + 0.1 * scores.get(role_family, 0)),
                                    f"keyword match -> {role_family}")
    seniority = ("Entry Level" if (min_years or 0) <= 1 else "Mid-Level" if (min_years or 0) <= 5 else "Senior")
    P["role.seniority_label"] = prov("inferred", 0.7, f"{min_years} yrs -> {seniority}")

    defense = bool(re.search(r"defense|national security|dod|department of defense|missile|weapon", low)) or clearance
    if defense: P["employer.defense_related"] = prov("inferred", 0.85, "defense/clearance/national-security references")

    # --- inferred analytical metadata ---
    bias = infer_bias(role_family, defense, clearance, low)
    latent = infer_latent(role_family, seniority, low, salary_max)
    for k in bias: P[f"bias_relevant_dimensions.{k}"] = prov("inferred", 0.6, "rule-based heuristic")
    for k in latent: P[f"latent_dimensions.{k}"] = prov("inferred", 0.6, "rule-based heuristic")

    job_id = "JOB_" + re.sub(r"[^A-Z0-9]+", "_", (company or "UNKNOWN").upper())[:24] + "_DRAFT"
    job = {
        "job_id": job_id, "job_status": "draft", "review_status": "draft_unreviewed",
        "source": {"source_type": "ingested", "posting_url": url,
                   "date_accessed": date_accessed or str(datetime.date.today()), "raw_text_available": True},
        "employer": {"company_name": company, "defense_related": defense,
                     "government_contractor": defense, "industry": None},
        "role": {"job_title": title, "job_family": role_family, "seniority_label": seniority,
                 "career_stage": "Early Career" if seniority == "Entry Level" else "Experienced",
                 "occupational_domain": "Engineering" if "engineer" in role_family else None},
        "location": {"city": city, "state": state, "country": "United States",
                     "work_arrangement": arrangement},
        "compensation": {"salary_min": salary_min, "salary_max": salary_max, "currency": "USD"},
        "requirements": {"minimum_education_level": degree, "minimum_years_experience": min_years,
                         "clearance_required": clearance, "citizenship_required": citizenship},
        "skills": {"technical_skills_required": skills, "technical_skills_preferred": [],
                   "tools_preferred": [], "domain_knowledge": [], "soft_skills": []},
        "responsibilities": {"core_responsibilities": [], "collaboration_partners": []},
        "job_attributes": {}, "latent_dimensions": latent, "bias_relevant_dimensions": bias,
        "resume_generation_axes": {
            "experience_level": ["internship_only", "0_to_2_years"] if seniority == "Entry Level"
                                else ["3_to_5_years", "5_to_10_years"],
            "domain_match_level": ["general_mechanical", "aerospace_adjacent", "aerospace_defense_direct"],
            "fit_level": ["low", "moderate", "high"]},
        "raw_posting_text": raw_sections,
        "field_provenance": P,
        "missing_fields": sorted(set(missing)),
    }
    return job


def infer_bias(role_family, defense, clearance, low):
    eng_hardware = role_family in ("mechanical_engineering", "aerospace_engineering")
    b = {}
    if eng_hardware:
        b.update(occupation_gender_type="masculine_stereotyped", masculinity_stereotype_strength="moderate_to_high",
                 femininity_stereotype_strength="low")
    if "health" in role_family or re.search(r"patient|care|clinical", low):
        b.update(caregiving_intensity="high", interpersonal_intensity="high")
    if defense or clearance:
        b.update(military_association_strength="high", conservatism_stereotype_strength="moderate_to_high",
                 security_sensitivity="high", public_sector_or_defense_proximity="high",
                 heteronormativity_stereotype_strength="high")
    else:
        b.setdefault("military_association_strength", "low")
        b.setdefault("public_sector_or_defense_proximity", "low")
    b.setdefault("lgbtq_stereotype_violation_potential", "high" if (eng_hardware and (defense or clearance)) else "moderate")
    b.setdefault("identity_signal_sensitivity_expected", "moderate_to_high" if (defense or clearance) else "moderate")
    return b


def infer_latent(role_family, seniority, low, salary_max):
    eng = "engineer" in role_family
    l = {"technical_intensity": "high" if eng else "moderate",
         "math_intensity": "high" if eng else "moderate",
         "physical_science_intensity": "high" if role_family in ("mechanical_engineering", "aerospace_engineering") else "low",
         "hands_on_hardware_intensity": "high" if role_family in ("mechanical_engineering", "aerospace_engineering") else "low",
         "teamwork_intensity": "high" if re.search(r"cross-functional|collaborat|team", low) else "moderate",
         "leadership_intensity": "high" if seniority == "Senior" else "low",
         "documentation_intensity": "high" if re.search(r"document|specification|drawing", low) else "moderate",
         "creativity_intensity": "moderate",
         "prestige_level": "high" if (salary_max or 0) >= 120000 else "moderate",
         "status_level": "high" if (salary_max or 0) >= 120000 else "moderate"}
    return l


def review_report(job):
    P = job["field_provenance"]
    lines = [f"# Inferred-metadata review — {job['job_id']}", "",
             f"review_status: **{job['review_status']}**  |  role_family: **{job['role']['job_family']}**", ""]
    lines.append("## Extracted (observed in posting)")
    for k, v in sorted(P.items()):
        if v["source"] == "extracted":
            lines.append(f"- `{k}` — conf {v['confidence']} — {v['evidence']}")
    lines.append("\n## Inferred (analytical — REVIEW THESE)")
    for k, v in sorted(P.items()):
        if v["source"] == "inferred":
            lines.append(f"- `{k}` = {_dig(job, k)} — conf {v['confidence']} — {v['evidence']}")
    lines.append("\n## Missing / needs manual entry")
    for k in job["missing_fields"]:
        lines.append(f"- `{k}`")
    lines.append("\nSet `review_status: approved_for_experiment` after correcting fields to use in final runs.")
    return "\n".join(lines)


def _dig(d, dotted):
    cur = d
    for p in dotted.split("."):
        cur = cur.get(p) if isinstance(cur, dict) else None
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-text", required=True)
    ap.add_argument("--company", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--role-family", default=None)
    ap.add_argument("--date-accessed", default=None)
    ap.add_argument("--extractor", default="rule_based", choices=["rule_based", "llm_assisted"])
    ap.add_argument("--out", default=os.path.join(SCHEMA_DIR, "job_descriptions", "drafts"))
    args = ap.parse_args()
    if args.extractor == "llm_assisted":
        raise SystemExit("llm_assisted extractor not implemented yet; use --extractor rule_based.")

    raw = open(args.raw_text).read()
    job = extract(raw, args.company, args.url, args.role_family, args.date_accessed)
    os.makedirs(args.out, exist_ok=True)
    stem = job["job_id"]
    json.dump(job, open(os.path.join(args.out, stem + ".json"), "w"), indent=2)
    json.dump({"extracted": [k for k, v in job["field_provenance"].items() if v["source"] == "extracted"],
               "inferred": [k for k, v in job["field_provenance"].items() if v["source"] == "inferred"]},
              open(os.path.join(args.out, stem + ".extraction_report.json"), "w"), indent=2)
    json.dump(job["missing_fields"], open(os.path.join(args.out, stem + ".missing_fields.json"), "w"), indent=2)
    open(os.path.join(args.out, stem + ".inferred_metadata_review.md"), "w").write(review_report(job))
    print(f"draft job -> {os.path.join(args.out, stem + '.json')}")
    print(f"  extracted: {sum(1 for v in job['field_provenance'].values() if v['source']=='extracted')} | "
          f"inferred: {sum(1 for v in job['field_provenance'].values() if v['source']=='inferred')} | "
          f"missing: {len(job['missing_fields'])}")
    print(f"  review report: {os.path.join(args.out, stem + '.inferred_metadata_review.md')}")
    print("  review_status=draft_unreviewed (set approved_for_experiment after human review)")


if __name__ == "__main__":
    main()
