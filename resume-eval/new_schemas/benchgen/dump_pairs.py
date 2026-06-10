#!/usr/bin/env python3
"""Write matched treatment/control/neutral resumes as readable markdown for human
inspection, with signal-block token diagnostics (NOT whole-resume token equality).

Run from resume-eval/:
  python -m new_schemas.benchgen.dump_pairs --run-dir new_schemas/runs/<...>
"""
import argparse
import collections
import json
import os


def fmt_tc(tc):
    def g(k): return tc.get(k)
    return (f"base_excl_signal={g('base_resume_excluding_signal_tokens')} | "
            f"signal_block={g('signal_block_tokens')} | full_resume={g('full_resume_tokens')} | "
            f"full_prompt={g('full_prompt_tokens')} | "
            f"signal_delta_vs_control={g('signal_token_delta_vs_control')} | "
            f"exact_signal_match={g('exact_signal_match')} | within_tolerance={g('within_tolerance')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "examples.jsonl"))]
    # dedupe to one row per (paired group, arm) — prompt_condition doesn't change the resume
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["paired_example_id"]].setdefault(r["treatment_or_control"], r)

    insp = os.path.join(args.run_dir, "inspect")
    os.makedirs(insp, exist_ok=True)
    index = ["# Matched-pair inspection index\n",
             f"run: `{os.path.basename(args.run_dir)}`  |  {len(by)} paired groups\n",
             "Each group is **base-identical with token-diagnosed signal blocks** "
             "(whole-resume token equality is NOT enforced).\n"]
    arm_order = ["baseline", "treatment", "control", "neutral"]

    for i, (pid, arms) in enumerate(sorted(by.items())):
        ref = arms.get("treatment") or next(iter(arms.values()))
        slug = f"pair_{i:03d}_{ref['qualification_profile_id']}_{ref['identity_signal_condition_id']}_{ref['name_variant_id']}"
        lines = [f"# {slug}\n",
                 f"- job: **{ref['job_id']}**",
                 f"- qualification_profile: **{ref['qualification_profile_id']}**",
                 f"- modes: length={ref.get('resume_length_mode')} | identity_description={ref.get('identity_description_mode')} | render={ref.get('identity_signal_render_mode')} | token_match={ref.get('token_match_mode')}",
                 f"- signal_channels: {ref.get('signal_channels')}",
                 f"- identity_condition: **{ref['identity_signal_condition_id']}** (load {ref['identity_load']})",
                 f"- name: **{ref['name_variant_id']}** ({ref['perceived_gender']})",
                 f"- arms present: {sorted(arms.keys())}\n",
                 "## Signal blocks + token diagnostics\n"]
        for arm in arm_order:
            if arm in arms:
                a = arms[arm]
                lines.append(f"**{arm}** — {fmt_tc(a['token_counts'])}")
                for d in a["identity_signals"]:
                    lines.append(f"  - [{d['section']}]\n```\n{d['block']}\n```")
                if not a["identity_signals"]:
                    lines.append("  - (no identity signal — baseline)")
                lines.append("")
        # base-identity check across arms
        bases = set()
        for arm in arms:
            a = arms[arm]
            base = a["rendered_resume"]
            for d in a["identity_signals"]:
                base = base.replace(f"## {d['section']}", "##__SIG__")  # crude marker; full check below
        # rigorous: compare rendered_resume with identity sections stripped
        def strip_sig(a):
            secs = a["rendered_resume"].split("\n\n---\n\n")
            id_secs = {d["section"] for d in a["identity_signals"]}
            return "\n\n---\n\n".join(s for s in secs if not any(s.startswith(f"## {n}") for n in id_secs))
        stripped = {strip_sig(arms[arm]) for arm in arms}
        lines.append(f"**base byte-identical across arms:** {len(stripped) == 1}\n")
        for arm in arm_order:
            if arm in arms:
                lines.append(f"\n---\n\n## Full resume — {arm}\n\n```\n{arms[arm]['rendered_resume']}\n```")
        open(os.path.join(insp, slug + ".md"), "w").write("\n".join(lines))
        index.append(f"- [{slug}]({slug}.md) — {ref['identity_signal_condition_id']} / "
                     f"{ref['qualification_profile_id']} / {ref['name_variant_id']} "
                     f"(resume {ref['token_counts']['full_resume_tokens']} tok)")

    open(os.path.join(insp, "index.md"), "w").write("\n".join(index) + "\n")
    print(f"wrote {len(by)} matched-pair files -> {insp}")
    print(f"index: {os.path.join(insp, 'index.md')}")


if __name__ == "__main__":
    main()
