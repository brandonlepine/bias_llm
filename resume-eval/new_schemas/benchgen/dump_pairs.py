#!/usr/bin/env python3
"""Write matched treatment/control/neutral resumes as readable markdown for human
inspection. One file per paired group + an index.

Run from resume-eval/:
  python -m new_schemas.benchgen.dump_pairs --run-dir new_schemas/runs/<...>
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "examples.jsonl"))]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["paired_example_id"]][r["treatment_or_control"]] = r

    insp = os.path.join(args.run_dir, "inspect")
    os.makedirs(insp, exist_ok=True)
    index = ["# Matched-pair inspection index\n",
             f"run: `{os.path.basename(args.run_dir)}`  |  {len(by)} paired groups\n"]

    arm_order = ["baseline", "treatment", "control", "neutral"]
    for i, (pid, arms) in enumerate(sorted(by.items())):
        ref = arms.get("treatment") or next(iter(arms.values()))
        slug = f"pair_{i:03d}_{ref['qualification_profile_id']}_{ref['identity_signal_condition_id']}_{ref['name_variant_id']}"
        lines = [f"# {slug}\n",
                 f"- job: **{ref['job_id']}**",
                 f"- qualification_profile: **{ref['qualification_profile_id']}**",
                 f"- identity_condition: **{ref['identity_signal_condition_id']}** (load {ref['identity_load']})",
                 f"- name: **{ref['name_variant_id']}** ({ref['perceived_gender']})",
                 f"- arms present: {sorted(arms.keys())}\n"]
        # matched-signal comparison
        lines.append("## Matched identity signals (treatment vs control vs neutral)\n")
        for arm in ["treatment", "control", "neutral"]:
            if arm in arms:
                sigs = arms[arm]["identity_signals"]
                tc = arms[arm]["token_counts"]
                lines.append(f"**{arm}** (resume {tc['resume_tokens']} tok, identity-section {tc['identity_section_tokens']} tok):")
                for s in sigs:
                    lines.append(f"  - [{s['section']}] {s['phrase']}")
                if not sigs:
                    lines.append("  - (no identity signal)")
                lines.append("")
        # full resumes
        for arm in arm_order:
            if arm in arms:
                lines.append(f"\n---\n\n## Full resume — {arm}\n")
                lines.append("```\n" + arms[arm]["rendered_resume"] + "\n```")
        with open(os.path.join(insp, slug + ".md"), "w") as f:
            f.write("\n".join(lines))
        index.append(f"- [{slug}]({slug}.md) — {ref['identity_signal_condition_id']} / "
                     f"{ref['qualification_profile_id']} / {ref['name_variant_id']}")

    with open(os.path.join(insp, "index.md"), "w") as f:
        f.write("\n".join(index) + "\n")
    print(f"wrote {len(by)} matched-pair files -> {insp}")
    print(f"index: {os.path.join(insp, 'index.md')}")


if __name__ == "__main__":
    main()
