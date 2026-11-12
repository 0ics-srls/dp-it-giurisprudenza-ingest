"""plan.schema.json is one file kept in three class repos (convert writes the plan, anonymize completes it, ingest reads it): they must be equal.

The sibling repos are looked up lazily; where they are absent (the ProcMon test container holds this repo only) the comparison is skipped and the file is
still checked to be a valid JSON Schema that describes the plan.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from builders import new_plan


def _own() -> Path:
    return Path(__file__).resolve().parent.parent / "plan.schema.json"


def _sibling(name: str) -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / f"dp-it-giurisprudenza-{name}" / "plan.schema.json"
        if candidate.is_file():
            return candidate
    return None


def test_the_schema_is_valid_and_accepts_an_empty_plan() -> None:
    schema = json.loads(_own().read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(new_plan("cc", "cc:x"))


@pytest.mark.parametrize("name", ["convert", "anonymize"])
def test_the_three_repos_carry_the_same_plan_schema(name: str) -> None:
    other = _sibling(name)
    if other is None:
        pytest.skip(f"dp-it-giurisprudenza-{name} is not next to this repo")

    assert json.loads(other.read_text(encoding="utf-8")) == json.loads(_own().read_text(encoding="utf-8"))
