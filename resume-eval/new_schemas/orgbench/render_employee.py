"""Render an internal EMPLOYEE profile to text (vs a resume = external candidate).
Reuses resume_backbone pools for contributions and benchgen identity_sections for the
identity factorial. Identity surface defaults to 'employee_bio'."""
RATING = {"low": "Meets some expectations", "moderate": "Meets expectations", "high": "Exceeds expectations"}
NEXT_LEVEL = {"Associate": "Senior Mechanical Engineer", "Senior": "Staff Mechanical Engineer",
              "Mid": "Senior Mechanical Engineer", "Staff": "Principal Mechanical Engineer"}


def next_level_for(emp):
    lvl = emp.get("current_level", "")
    for k, v in NEXT_LEVEL.items():
        if k.lower() in lvl.lower() or k.lower() in emp.get("current_title", "").lower():
            return v
    return "the next level"


def render_employee(emp, backbone, identity_secs, name, rng):
    dims = emp["performance_dimensions"]
    parts = [f"# {name['first_name']} {name['last_name']}\n\n"
             f"{emp['current_title']}  |  {emp['current_level']}  |  {emp['tenure_years']} years tenure"]
    parts.append("## PERFORMANCE SUMMARY\n\n" +
                 "\n".join(f"- {d.replace('_', ' ').title()}: {RATING[dims[d]]}" for d in emp.get("performance_dimensions", {})))
    proj = backbone["project_pool"][:2]
    contrib = "\n\n".join(f"### {p['project_name']}\n" + "\n".join(f"- {b}" for b in p["bullet_pool"][:3]) for p in proj)
    parts.append("## KEY CONTRIBUTIONS\n\n" + contrib)
    parts.append("## MANAGER FEEDBACK\n\n" + emp["manager_feedback"])
    parts.append("## PEER FEEDBACK\n\n" + emp["peer_feedback"])
    base = "\n\n---\n\n".join(parts)
    signal = "\n\n---\n\n".join(f"## {sec}\n\n{block}" for sec, block in identity_secs)
    full = base if not signal else base + "\n\n---\n\n" + signal
    return base, signal, full
