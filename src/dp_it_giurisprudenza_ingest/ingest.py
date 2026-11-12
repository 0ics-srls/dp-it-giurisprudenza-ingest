"""The ``execute`` action: loads the plan files of the given items, one transaction per item."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from dp_it_giurisprudenza_ingest.db import role_of
from dp_it_giurisprudenza_ingest.errors import CONFIG_ERROR, SYSTEM_ERROR, TRANSIENT_ERROR, DbConnectError, IngestError
from dp_it_giurisprudenza_ingest.repository import PLAN_SCHEMA_VERSION, TABLES, write_cassazione, write_cc, write_cgt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

    from dp_it_giurisprudenza_ingest.repository import Cursor

KINDS = ("cc", "cassazione", "cgt")
_SCHEMA = re.compile(r"^ds_[a-z0-9_]+$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_ARRAYS = ("pronunce", "massime", "norme", "citations", "cassazione", "cgt", "cgt_skipped")


class Connection(Protocol):
    def transaction(self) -> AbstractContextManager[Any]: ...

    def cursor(self) -> AbstractContextManager[Cursor]: ...

    def execute(self, sql: Any, params: Any = None) -> Any: ...

    def close(self) -> None: ...


def safe_name(item_id: str) -> str:
    """File stem of an item id (identical to the grab, convert and anonymize classes)."""
    return _UNSAFE.sub("_", item_id.replace(" ", "-"))


def _validate(params: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, str, str]:
    items = params.get("items")
    if not isinstance(items, list) or not items:
        raise IngestError(CONFIG_ERROR, "items must be a non-empty array")
    for entry in items:
        meta = entry.get("meta") if isinstance(entry, dict) else None
        if not isinstance(entry, dict) or not entry.get("key") or not isinstance(meta, dict) or meta.get("kind") not in KINDS:
            raise IngestError(CONFIG_ERROR, f"item needs a key and meta.kind one of {list(KINDS)}: {entry!r}")
        if not isinstance(meta.get("itemId"), str) or not meta["itemId"]:
            raise IngestError(CONFIG_ERROR, f"item {entry['key']}: meta.itemId is required")
    directory, conn_string, schema = params.get("dirIn"), params.get("connString"), params.get("schema")
    if not isinstance(directory, str) or not directory:
        raise IngestError(CONFIG_ERROR, "dirIn is required")
    if not isinstance(conn_string, str) or not conn_string:
        raise IngestError(CONFIG_ERROR, "connString is required")
    if not isinstance(schema, str) or not _SCHEMA.match(schema):
        raise IngestError(CONFIG_ERROR, "schema must be a dataset schema name (ds_...)")
    return items, Path(directory), conn_string, schema


def _load(entry: dict[str, Any], directory: Path) -> dict[str, Any]:
    meta = entry["meta"]
    plan = json.loads((directory / f"{safe_name(meta['itemId'])}.plan.json").read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("version") != PLAN_SCHEMA_VERSION or any(not isinstance(plan.get(name), list) for name in _ARRAYS):
        msg = f"unsupported plan file (version {plan.get('version') if isinstance(plan, dict) else None})"
        raise ValueError(msg)
    if plan.get("kind") != meta["kind"]:
        msg = f"the plan holds kind {plan.get('kind')!r}, the item says {meta['kind']!r}"
        raise ValueError(msg)
    return plan


def _write(cursor: Cursor, plan: dict[str, Any]) -> tuple[dict[str, int], int]:
    if plan["kind"] == "cc":
        return write_cc(cursor, plan), 0
    if plan["kind"] == "cassazione":
        return write_cassazione(cursor, plan)
    return write_cgt(cursor, plan), 0


def _items(
    connection: Connection, schema: str, items: list[dict[str, Any]], directory: Path, on_progress: Callable[[int, str], None],
) -> Iterator[tuple[dict[str, Any], tuple[dict[str, int], int, int] | Exception]]:
    for index, entry in enumerate(items):
        try:
            plan = _load(entry, directory)
            with connection.transaction():
                connection.execute(f"SET search_path TO {schema}, public")  # schema is validated against ^ds_[a-z0-9_]+$
                with connection.cursor() as cursor:
                    counts, warnings = _write(cursor, plan)
            yield entry, (counts, warnings, len(plan.get("quarantine") or []))
        except Exception as error:  # noqa: BLE001 - any failure of one item is reported, the next item still runs
            yield entry, error
        on_progress(int((index + 1) * 100 / len(items)), f"{index + 1}/{len(items)} {entry['key']}")


def run_ingest(params: dict[str, Any], connect: Callable[[str], Connection], on_progress: Callable[[int, str], None]) -> dict[str, Any]:
    """Ingests every item; failed items are rolled back and reported together as one system_error after the good ones were committed."""
    items, directory, conn_string, schema = _validate(params)
    try:
        connection = connect(conn_string)
    except (OSError, DbConnectError) as ex:
        msg = f"cannot connect to the dataset database as role {role_of(conn_string)} ({type(ex).__name__})"
        raise IngestError(TRANSIENT_ERROR, msg) from None
    metrics: dict[str, int] = {"items": 0, "quarantined": 0, "warnings.data_decisione_after_deposito": 0}
    metrics.update({f"written.{schema}.{table}": 0 for table in TABLES})
    loaded: list[str] = []
    failures: list[str] = []
    try:
        for entry, outcome in _items(connection, schema, items, directory, on_progress):
            if isinstance(outcome, Exception):
                failures.append(f"{entry['key']} ({type(outcome).__name__}: {outcome})")
                continue
            counts, warnings, quarantined = outcome
            metrics["items"] += 1
            metrics["quarantined"] += quarantined
            metrics["warnings.data_decisione_after_deposito"] += warnings
            for table, value in counts.items():
                metrics[f"written.{schema}.{table}"] += value
            loaded.append(entry["key"])
    finally:
        connection.close()
    if failures:
        msg = f"{len(failures)} of {len(items)} items failed: {'; '.join(failures)}"
        raise IngestError(SYSTEM_ERROR, msg)
    return {"metrics": metrics, "loaded": loaded}
