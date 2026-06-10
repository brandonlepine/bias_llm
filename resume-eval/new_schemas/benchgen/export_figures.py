#!/usr/bin/env python3
"""Copy a run's figures + summaries into a git-TRACKED directory so they can be
committed and pulled to a local checkout (runs/ itself is gitignored). Synchronous;
no async, no network.

On the pod:
  python -m new_schemas.benchgen.export_figures --run-dir new_schemas/runs/<...>
  git add new_schemas/figures && git commit -m "pilot figures" && git push
Locally:
  git pull   # figures land in new_schemas/figures/<run_tag>/
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dest", default=os.path.join(SCHEMA_DIR, "figures"))
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    tag = os.path.basename(run_dir.rstrip("/"))
    out = os.path.join(args.dest, tag)
    os.makedirs(out, exist_ok=True)

    copied = []
    fig_dir = os.path.join(run_dir, "figures")
    if os.path.isdir(fig_dir):
        for f in sorted(os.listdir(fig_dir)):
            if f.lower().endswith((".png", ".pdf", ".svg")):
                shutil.copy2(os.path.join(fig_dir, f), os.path.join(out, f))
                copied.append(f)
    for extra in ("analysis_summary.json", "config.json", "signal_diagnostics.jsonl"):
        src = os.path.join(run_dir, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, extra))
            copied.append(extra)

    print(f"copied {len(copied)} files -> {out}")
    for f in copied:
        print(f"  {f}")
    rel = os.path.relpath(out, os.path.dirname(SCHEMA_DIR))
    print(f"\nTo bring these to a local checkout:\n"
          f"  git add {rel} && git commit -m 'figures: {tag}' && git push\n"
          f"  # then locally: git pull")


if __name__ == "__main__":
    main()
