#!/usr/bin/env python3
"""Custom Prodigy recipe for reviewing the BBQ stereotype-pair cards.

Each card shows one unique stereotype pair (name frame + group-noun frame) and a 4-way choice —
Approve / Reject / Mark for review / Mark for edit — plus a free-text box for corrections or notes.
Hit the green ✓ (accept) to save the selected action; the chosen option lands in eg["accept"] and any
correction in eg["edit"].

Run (Prodigy must be installed in the active env):

    prodigy bbq-review bbq_review \\
        data/bbq/stereotypes/bbq_review_cards.jsonl \\
        -F scripts/prodigy_bbq_review.py

Then export your decisions for apply_bbq_review.py:

    prodigy db-out bbq_review > data/bbq/stereotypes/bbq_review_annotations.jsonl
"""
from __future__ import annotations

import prodigy
from prodigy.components.stream import get_stream


HELP = (
    "Review each stereotype PAIR once (it covers all names + both frames).\n"
    "Pick one action, optionally type a corrected predicate / note, then press the green ✓ to save.\n"
    "  • Approve — the pair is a valid, clean stereotype contrast\n"
    "  • Reject — drop it (not a real stereotype, ungrammatical, wrong groups)\n"
    "  • Mark for review — unsure / revisit later\n"
    "  • Mark for edit — keep the stereotype but fix the predicate (type the fix in the box)"
)


@prodigy.recipe(
    "bbq-review",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Path to bbq_review_cards.jsonl", "positional", None, str),
)
def bbq_review(dataset: str, source: str):
    stream = get_stream(source, rehash=True, dedup=True)

    blocks = [
        {"view_id": "html"},
        {"view_id": "choice", "text": None},
        {"view_id": "text_input", "field_id": "edit", "field_rows": 1,
         "field_label": "Correction / note (used when 'Mark for edit')", "field_autofocus": False},
    ]
    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {
            "blocks": blocks,
            "choice_style": "single",
            "choice_auto_accept": False,
            "buttons": ["accept", "undo"],
            "instructions": HELP,
            "show_flag": True,
            "global_css": ".prodigy-content{max-width:760px;}",
        },
    }
