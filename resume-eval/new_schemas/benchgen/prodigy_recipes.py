"""Prodigy recipe for single-resume QA of generated benchmark examples.

Usage (requires Prodigy installed):
  prodigy resume-qa my_qa_dataset new_schemas/prodigy/resume_qa.jsonl \
      -F new_schemas/benchgen/prodigy_recipes.py

Each screen shows one rendered resume + its intended metadata labels; reviewer
selects QA flags (multi-choice) and can add a note. Accepted/flagged annotations
are stored in the Prodigy dataset for export (prodigy db-out my_qa_dataset).
"""
try:
    import prodigy
    from prodigy.components.stream import get_stream
except Exception:  # allow import without prodigy installed
    prodigy = None


if prodigy is not None:
    @prodigy.recipe(
        "resume-qa",
        dataset=("Prodigy dataset to save to", "positional", None, str),
        source=("Path to resume_qa.jsonl from export_prodigy", "positional", None, str),
    )
    def resume_qa(dataset, source):
        stream = get_stream(source, rehash=True, dedup=True, input_key="html")
        blocks = [
            {"view_id": "html"},
            {"view_id": "choice", "text": None},
            {"view_id": "text_input", "field_id": "note", "field_label": "Note (optional)"},
        ]
        return {
            "dataset": dataset,
            "stream": stream,
            "view_id": "blocks",
            "config": {"blocks": blocks, "choice_style": "multiple",
                       "buttons": ["accept", "reject", "ignore"]},
        }
