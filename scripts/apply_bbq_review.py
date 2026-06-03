#!/usr/bin/env python3
"""Propagate the Prodigy review decisions back onto every BBQ pair row.

Reads the per-card annotations exported from Prodigy (`prodigy db-out bbq_review > ...jsonl`) and
joins them to the full pairs CSV on (category, Group_x, Group_y, predicate_label_provisional), so each
decision covers all of that stereotype's name/frame variants. For 'edit' cards with a correction, the
predicate (and the sentences/continuation derived from it) are rewritten.

Outputs:
  data/bbq/stereotypes/bbq_pairs_reviewed.csv   all rows + review_action / review_edit
  data/bbq/stereotypes/bbq_pairs_approved.csv   rows whose action is approve or (applied) edit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bbq_winoqueer_pairs import pluralize  # noqa: E402

KEY = ["category", "Group_x", "Group_y", "predicate_label_provisional"]
VALID = {"approve", "reject", "review", "edit"}


def action_of(eg: dict) -> str | None:
    if eg.get("answer") == "ignore":
        return None
    chosen = (eg.get("accept") or [None])[0]
    if chosen in VALID:
        return chosen
    return {"accept": "approve", "reject": "reject"}.get(eg.get("answer", ""))


def load_decisions(path: Path) -> dict[tuple, tuple[str, str]]:
    decisions: dict[tuple, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eg = json.loads(line)
            if not all(k in eg for k in KEY):
                continue
            act = action_of(eg)
            if act is None:
                continue
            key = tuple(eg[k] for k in KEY)
            decisions[key] = (act, str(eg.get("edit", "")).strip())  # last annotation wins
    return decisions


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply Prodigy BBQ review decisions to all pair rows.")
    ap.add_argument("--annotations", type=Path, default=Path("data/bbq/stereotypes/bbq_review_annotations.jsonl"))
    ap.add_argument("--pairs_csv", type=Path, default=Path("data/bbq/stereotypes/bbq_pairs_all.csv"))
    ap.add_argument("--out_reviewed", type=Path, default=Path("data/bbq/stereotypes/bbq_pairs_reviewed.csv"))
    ap.add_argument("--out_approved", type=Path, default=Path("data/bbq/stereotypes/bbq_pairs_approved.csv"))
    args = ap.parse_args()

    decisions = load_decisions(args.annotations)
    d = pd.read_csv(args.pairs_csv).fillna("")
    keys = list(zip(*[d[k] for k in KEY]))
    d["review_action"] = [decisions.get(k, ("undecided", ""))[0] for k in keys]
    d["review_edit"] = [decisions.get(k, ("", ""))[1] for k in keys]

    # Apply edits: rewrite predicate + sentences for rows the reviewer corrected.
    edited = 0
    for i, r in d[(d["review_action"] == "edit") & (d["review_edit"] != "")].iterrows():
        new = r["review_edit"].strip()
        pred = pluralize(new) if r["frame"] == "groupnoun" else new
        d.at[i, "predicate"] = pred
        d.at[i, "continuation"] = pred
        d.at[i, "sent_x"] = f"{r['prefix_x']} {pred}"
        d.at[i, "sent_y"] = f"{r['prefix_y']} {pred}"
        edited += 1

    d.to_csv(args.out_reviewed, index=False)
    approved = d[d["review_action"].isin(["approve", "edit"]) &
                 ~((d["review_action"] == "edit") & (d["review_edit"] == ""))]
    approved.to_csv(args.out_approved, index=False)

    print(f"Decisions loaded: {len(decisions)} cards")
    print(f"Wrote {args.out_reviewed}  ({len(d)} rows)")
    print(f"Wrote {args.out_approved}  ({len(approved)} approved/edited rows; {edited} edited)")
    print("\nrows per review_action:")
    print(d["review_action"].value_counts().to_string())
    und = d[d["review_action"] == "undecided"]
    if len(und):
        print(f"\n{und[KEY].drop_duplicates().shape[0]} stereotype-pairs still undecided "
              f"(not yet reviewed / ignored).")


if __name__ == "__main__":
    main()
