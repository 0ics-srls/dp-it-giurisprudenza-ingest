"""Proc-protocol loop and manifest of the ingest class."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import main
from builders import FakeConnection, item, new_plan, pronuncia, write_plan

ROOT = Path(__file__).resolve().parent.parent.parent
CONN = "Host=db;Port=5432;Database=lex;Username=ds_it_giurisprudenza_rw;Password=s3cr3t-pw"


def lines_of(text: str, connection: FakeConnection | None = None) -> tuple[int, list[dict[str, Any]]]:
    out = io.StringIO()
    fake = connection or FakeConnection()
    code = main.main(io.StringIO(text), out, lambda _dsn: fake)
    return code, [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def request(action: str, **params: Any) -> str:
    return json.dumps({"_action": action, "_meta": {"executionId": "e1"}, **params}) + "\n"


def execute(directory: Path) -> str:
    return request("execute", items=[item("cc", "cc:pronunce_2001_oggi")], dirIn=str(directory), connString=CONN, schema="ds_it_giurisprudenza")


def test_execute_emits_progress_then_one_result(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))

    code, lines = lines_of(execute(tmp_path))

    assert code == 0
    assert lines[-1]["type"] == "result"
    assert all(m["type"] == "progress" for m in lines[:-1])
    assert lines[-1]["data"]["metrics"]["items"] == 1


def test_a_failing_item_prints_one_error_line_without_the_password(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))

    code, lines = lines_of(execute(tmp_path), FakeConnection(fail_on="INSERT INTO pronunce"))

    errors = [m for m in lines if m["type"] == "error"]
    assert code != 0
    assert len(errors) == 1
    assert errors[0]["code"] == "system_error"
    assert "cc:pronunce_2001_oggi|v1" in errors[0]["message"]
    assert "s3cr3t-pw" not in json.dumps(lines)


def test_bad_line_unknown_action_and_terminate() -> None:
    _, lines = lines_of("not json\n" + request("bogus"))
    assert [m["code"] for m in lines] == ["config_error", "config_error"]

    _, after = lines_of(request("terminate") + request("bogus"))
    assert after == []


def test_manifest_declares_execute_with_a_secret_connection_string() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

    properties = manifest["actions"]["execute"]["input"]["properties"]
    assert manifest["name"] == "dp-it-giurisprudenza-ingest"
    assert {"items", "dirIn", "connString", "schema"} <= set(properties)
    assert properties["connString"]["x-secret"] is True
