#!/usr/bin/env python3
"""Deduplicate the BBQ pairs into one review card per unique stereotype-pair for Prodigy.

The build emits ~11x redundant rows per stereotype (4 names x both frames). For review we collapse
to one card per (category, target, dominant, social_value), showing a representative name-frame pair
AND the group-noun pair, so the reviewer judges each stereotype once. The decision propagates back to
all underlying rows via apply_bbq_review.py (keyed on the same 4 fields).

Outputs:
  data/bbq/stereotypes/bbq_review_cards.jsonl   (Prodigy stream; html + 4-option choice)
  data/bbq/stereotypes/bbq_review_cards.csv      (same cards, flat, for inspection)
"""
from __future__ import annotations

import argparse
import html as _html
import json
from pathlib import Path

import pandas as pd

KEY = ["category", "Group_x", "Group_y", "predicate_label_provisional"]
OPTIONS = [
    {"id": "approve", "text": "✓ Approve"},
    {"id": "reject", "text": "✗ Reject"},
    {"id": "review", "text": "\U0001f50d Mark for review"},
    {"id": "edit", "text": "✏ Mark for edit"},
]


def first_of(g: pd.DataFrame, frame: str, col: str) -> str:
    sub = g[g["frame"] == frame]
    return str(sub.iloc[0][col]) if len(sub) else ""


def render(category, sv, gx, gy, name_x, name_y, grp_x, grp_y, predicate, n_rows) -> str:
    def esc(s):
        return _html.escape(str(s))
    return f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5;">
  <div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.5px;">
    {esc(category)} &middot; {esc(sv) or '(unlabeled)'}
  </div>
  <div style="margin:6px 0 2px;font-size:12px;color:#555;">
    target <b style="color:#c0392b;">{esc(gx)}</b> &nbsp;vs&nbsp; dominant <b style="color:#2c7fb8;">{esc(gy)}</b>
    &nbsp;&middot;&nbsp; predicate: <b>{esc(predicate)}</b> &nbsp;&middot;&nbsp; {n_rows} variants
  </div>
  <table style="margin-top:8px;border-collapse:collapse;font-size:15px;">
    <tr><td style="color:#999;padding:2px 10px 2px 0;">name</td>
        <td style="padding:2px 0;">&ldquo;{esc(name_x)}&rdquo; &nbsp;<span style="color:#bbb;">vs</span>&nbsp; &ldquo;{esc(name_y)}&rdquo;</td></tr>
    <tr><td style="color:#999;padding:2px 10px 2px 0;">group</td>
        <td style="padding:2px 0;">&ldquo;{esc(grp_x)}&rdquo; &nbsp;<span style="color:#bbb;">vs</span>&nbsp; &ldquo;{esc(grp_y)}&rdquo;</td></tr>
  </table>
</div>""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare deduplicated BBQ review cards for Prodigy.")
    ap.add_argument("--pairs_csv", type=Path, default=Path("data/bbq/stereotypes/bbq_pairs_all.csv"))
    ap.add_argument("--out_jsonl", type=Path, default=Path("data/bbq/stereotypes/bbq_review_cards.jsonl"))
    ap.add_argument("--out_csv", type=Path, default=Path("data/bbq/stereotypes/bbq_review_cards.csv"))
    args = ap.parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.pairs_csv).fillna("")
    cards = []
    for (cat, gx, gy, sv), g in d.groupby(KEY, sort=True):
        name_x = first_of(g, "name", "sent_x") or first_of(g, "groupnoun", "sent_x")
        name_y = first_of(g, "name", "sent_y") or first_of(g, "groupnoun", "sent_y")
        grp_x = first_of(g, "groupnoun", "sent_x")
        grp_y = first_of(g, "groupnoun", "sent_y")
        predicate = first_of(g, "name", "predicate") or first_of(g, "groupnoun", "predicate")
        cards.append({
            "category": cat, "Group_x": gx, "Group_y": gy, "predicate_label_provisional": sv,
            "predicate": predicate, "n_rows": int(len(g)),
            "sent_x_name": name_x, "sent_y_name": name_y, "sent_x_group": grp_x, "sent_y_group": grp_y,
        })
    cards_df = pd.DataFrame(cards).sort_values(["category", "Group_x", "predicate_label_provisional"]).reset_index(drop=True)
    cards_df.to_csv(args.out_csv, index=False)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for r in cards_df.to_dict(orient="records"):
            task = {
                **{k: r[k] for k in KEY + ["predicate", "n_rows"]},
                "html": render(r["category"], r["predicate_label_provisional"], r["Group_x"], r["Group_y"],
                               r["sent_x_name"], r["sent_y_name"], r["sent_x_group"], r["sent_y_group"], r["predicate"], r["n_rows"]),
                "options": OPTIONS,
                "meta": {"category": r["category"], "n_rows": r["n_rows"]},
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out_jsonl}  ({len(cards_df)} review cards, collapsing {len(d)} pair rows)")
    print(f"Wrote {args.out_csv}")
    print("\ncards per category:")
    print(cards_df.groupby("category").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
