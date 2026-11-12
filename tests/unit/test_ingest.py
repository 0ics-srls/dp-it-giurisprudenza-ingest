"""run_ingest with a fake connection: statement order, parameters, transactions, the embedding rule, quarantine, secrets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from builders import (
    ECLI,
    GUID,
    UUID,
    FakeConnection,
    cgt,
    citation,
    item,
    massima,
    new_plan,
    norma,
    pronuncia,
    sentenza,
    write_plan,
)
from dp_it_giurisprudenza_ingest.errors import CONFIG_ERROR, SYSTEM_ERROR, TRANSIENT_ERROR, IngestError
from dp_it_giurisprudenza_ingest.ingest import run_ingest

CONN = "Host=db;Port=5432;Database=lex;Username=ds_it_giurisprudenza_rw;Password=s3cr3t-pw"
SCHEMA = "ds_it_giurisprudenza"


def params(directory: Path, *items: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"items": list(items), "dirIn": str(directory), "connString": CONN, "schema": SCHEMA, **extra}


def run(directory: Path, connection: FakeConnection, *items: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return run_ingest(params(directory, *items, **extra), lambda _dsn: connection, lambda *_: None)


def kinds(connection: FakeConnection) -> list[str]:
    return [" ".join(sql.split()[:3]) for _, sql, _ in connection.log]


def statement(connection: FakeConnection, text: str) -> tuple[str, str, Any]:
    return next(entry for entry in connection.log if text in entry[1])


def test_cc_pronunce_item_statement_order_in_one_transaction(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()], citations=[citation()]))
    connection = FakeConnection()

    result = run(tmp_path, connection, item("cc", "cc:pronunce_2001_oggi"))

    assert connection.events == ["begin", "commit"]
    assert kinds(connection) == [
        "SET search_path TO",
        "INSERT INTO pronunce",
        "INSERT INTO passages",
        "DELETE FROM passages",
        "UPDATE norme_impugnate SET",
    ]
    assert connection.closed
    assert result["metrics"][f"written.{SCHEMA}.pronunce"] == 1
    assert result["metrics"][f"written.{SCHEMA}.passages"] == 3  # fatto, diritto and dispositivo (testo is empty)
    assert result["metrics"]["items"] == 1


def test_passages_hold_the_sections_but_never_the_epigrafe_header(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    connection = FakeConnection()

    run(tmp_path, connection, item("cc", "cc:pronunce_2001_oggi"))

    rows = statement(connection, "INSERT INTO passages")[2]
    assert [r[0:3] for r in rows] == [("pronuncia", ECLI, 0), ("pronuncia", ECLI, 1), ("pronuncia", ECLI, 2)]
    assert all("EPIGRAFE" not in r[4] and "EPIGRAFE" not in r[5] for r in rows)
    assert rows[0][5].startswith("[Corte Cost. S n. 95/2026 — fatto]\n")
    assert statement(connection, "DELETE FROM passages")[2] == ("pronuncia", ECLI, [0, 1, 2])


def test_embedding_is_kept_when_the_contextual_text_is_unchanged_and_nulled_when_it_changes() -> None:
    from dp_it_giurisprudenza_ingest import repository  # noqa: PLC0415

    sql = " ".join(repository.UPSERT_PASSAGE.split())

    assert "ON CONFLICT (corpus, doc_key, seq) DO UPDATE" in sql
    assert "embedding = CASE WHEN passages.text_contextual IS DISTINCT FROM EXCLUDED.text_contextual THEN NULL ELSE passages.embedding END" in sql


def test_ecli_via_is_only_written_by_the_citations_statement(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()], citations=[citation()]))
    connection = FakeConnection()

    run(tmp_path, connection, item("cc", "cc:pronunce_2001_oggi"))

    with_via = [sql for _, sql, _ in connection.log if "ecli_via" in sql]
    assert len(with_via) == 1
    assert with_via[0].lstrip().startswith("UPDATE norme_impugnate")
    assert "EXISTS (SELECT 1 FROM pronunce" in with_via[0]  # only when the decision exists
    assert statement(connection, "UPDATE norme_impugnate")[2] == [(ECLI, "epigrafe", "ordinanze", 2025, 173, ECLI)]


def test_massime_item_writes_massime_and_their_passages(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:massime_2001_oggi", massime=[massima(numero=1), massima(numero=2, text="Secondo principio.")]))
    connection = FakeConnection()

    result = run(tmp_path, connection, item("cc", "cc:massime_2001_oggi"))

    assert kinds(connection)[:2] == ["SET search_path TO", "INSERT INTO massime"]
    passages = statement(connection, "INSERT INTO passages")[2]
    assert [(r[0], r[1], r[2]) for r in passages] == [("massima", ECLI, 1), ("massima", ECLI, 2)]
    assert passages[1][4] == passages[1][5] == "Secondo principio."  # embedded text equals the massima text
    assert result["metrics"][f"written.{SCHEMA}.massime"] == 2


def test_norme_item_upserts_flips_missing_rows_and_never_deletes(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(), norma(numero=174)]))
    connection = FakeConnection()

    result = run(tmp_path, connection, item("cc", "cc:norme_ordinanze"))

    upsert = " ".join(statement(connection, "INSERT INTO norme_impugnate")[1].split())
    assert "ON CONFLICT (registro_tipo, registro_anno, registro_numero, ord) DO UPDATE" in upsert
    assert "pending = true" in upsert
    assert "last_seen = now()" in upsert
    assert "first_seen" not in upsert.split("DO UPDATE")[1]  # first_seen is kept
    assert "ecli" not in upsert.split("DO UPDATE")[1].replace("ecli_via", "")  # a resolved decision link survives a reload
    flip = statement(connection, "SET pending = false")
    assert "registro_tipo = %s" in flip[1]
    assert "last_seen < now()" in flip[1]
    assert flip[2] == ("ordinanze",)
    assert not any("DELETE FROM norme_impugnate" in sql for _, sql, _ in connection.log)
    assert result["metrics"][f"written.{SCHEMA}.norme_impugnate"] == 2


def test_an_empty_registry_snapshot_still_flips_its_registry(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:norme_ricorsi"))
    connection = FakeConnection()

    run(tmp_path, connection, item("cc", "cc:norme_ricorsi"))

    assert statement(connection, "SET pending = false")[2] == ("ricorsi",)


def test_cassazione_item_stores_both_texts_and_chunks_the_anonymized_one(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cassazione", "cass:snciv:2025:01", cassazione=[sentenza()]))
    connection = FakeConnection()

    result = run(tmp_path, connection, item("cassazione", "cass:snciv:2025:01"))

    row = statement(connection, "INSERT INTO cassazione_sentenze")[2][0]
    assert row[0] == "snciv2025000001O"
    assert row[5] == 6063  # num_dec as integer
    assert "ORIGINALE" in row[12]  # text_raw_original
    assert "ORIGINALE" not in row[13]  # text_anon
    sql = " ".join(statement(connection, "INSERT INTO cassazione_sentenze")[1].split())
    assert "text_raw_original = EXCLUDED.text_raw_original, text_anon = EXCLUDED.text_anon" in sql
    passages = statement(connection, "INSERT INTO passages")[2]
    assert passages[0][0:2] == ("cassazione", "snciv2025000001O")
    assert all("ORIGINALE" not in r[4] for r in passages)
    assert passages[0][5].startswith("[Cass. snciv Sez. 5 n. 06063/2025]\n")
    assert result["metrics"][f"written.{SCHEMA}.cassazione_sentenze"] == 1


def test_data_decisione_after_deposito_is_counted_as_a_warning_not_refused(tmp_path: Path) -> None:
    bad = sentenza(data_decisione="2025-09-15", data_deposito="2025-09-01")
    write_plan(tmp_path, new_plan("cassazione", "cass:x", cassazione=[bad, sentenza("snciv2025000002O")]))

    result = run(tmp_path, FakeConnection(), item("cassazione", "cass:x"))

    assert result["metrics"]["warnings.data_decisione_after_deposito"] == 1
    assert result["metrics"][f"written.{SCHEMA}.cassazione_sentenze"] == 2


def test_a_sentence_that_was_not_anonymized_is_refused(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cassazione", "cass:x", cassazione=[sentenza(text_anon=None)]))

    with pytest.raises(IngestError) as error:
        run(tmp_path, FakeConnection(), item("cassazione", "cass:x"))

    assert error.value.code == SYSTEM_ERROR
    assert "text_anon" in str(error.value)


def test_cgt_item_writes_decisions_passages_and_skips(tmp_path: Path) -> None:
    skipped = {"guid": "{11111111-2222-3333-4444-555555555555}", "estremi": "Sentenza del 01/12/2024", "reason": "no_text"}
    write_plan(tmp_path, new_plan("cgt", "cgt:2024-12-31:2024-12-31", cgt=[cgt()], cgt_skipped=[skipped]))
    connection = FakeConnection()

    result = run(tmp_path, connection, item("cgt", "cgt:2024-12-31:2024-12-31"))

    row = statement(connection, "INSERT INTO cgt_sentenze")[2][0]
    assert row[0] == UUID  # the DocTrib GUID, braces stripped, lower case
    assert row[6] == 7355  # numero as integer
    passages = statement(connection, "INSERT INTO passages")[2]
    assert passages[0][0:2] == ("cgt", UUID)
    assert passages[0][5].startswith("[Cass. CGT-2 Campania Sez. 2 n. 7355/2024]\n")
    assert statement(connection, "INSERT INTO cgt_skipped")[2] == [("11111111-2222-3333-4444-555555555555", "Sentenza del 01/12/2024", "no_text")]
    assert "DELETE FROM cgt_skipped" in " ".join(sql for _, sql, _ in connection.log)  # a decision that now has text is no longer "skipped"
    assert result["metrics"][f"written.{SCHEMA}.cgt_sentenze"] == 1
    assert result["metrics"][f"written.{SCHEMA}.cgt_skipped"] == 1


def test_quarantined_documents_are_counted_and_never_written(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cgt", "cgt:q", cgt=[cgt()], quarantine=[{"id": "{BBBB}", "reason": "pii_guardrail", "findings": {"cf": 1}}]))
    connection = FakeConnection()

    result = run(tmp_path, connection, item("cgt", "cgt:q"))

    assert result["metrics"]["quarantined"] == 1
    assert all("{BBBB}" not in repr(params) for _, _, params in connection.log)


def test_cc_plan_with_no_content_still_commits(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_1956_1980"))
    connection = FakeConnection()

    run(tmp_path, connection, item("cc", "cc:pronunce_1956_1980"))

    assert connection.events == ["begin", "commit"]


def test_two_items_get_two_transactions_and_a_failure_rolls_back_only_its_item(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    write_plan(tmp_path, new_plan("cc", "cc:massime_2001_oggi", massime=[massima()]))
    connection = FakeConnection(fail_on="INSERT INTO massime")

    with pytest.raises(IngestError) as error:
        run(tmp_path, connection, item("cc", "cc:massime_2001_oggi"), item("cc", "cc:pronunce_2001_oggi"))

    assert error.value.code == SYSTEM_ERROR
    assert "cc:massime_2001_oggi|v1" in str(error.value)
    assert "cc:pronunce_2001_oggi|v1" not in str(error.value)
    assert connection.events == ["begin", "rollback", "begin", "commit"]


def test_missing_and_malformed_plans_fail_the_item(tmp_path: Path) -> None:
    with pytest.raises(IngestError) as missing:
        run(tmp_path, FakeConnection(), item("cc", "cc:pronunce_2001_oggi"))
    assert missing.value.code == SYSTEM_ERROR
    write_plan(tmp_path, {**new_plan("cc", "cc:pronunce_2001_oggi"), "version": 2})
    with pytest.raises(IngestError) as bad:
        run(tmp_path, FakeConnection(), item("cc", "cc:pronunce_2001_oggi"))
    assert "plan" in str(bad.value)
    write_plan(tmp_path, new_plan("cgt", "cc:pronunce_2001_oggi"))
    with pytest.raises(IngestError) as other:
        run(tmp_path, FakeConnection(), item("cc", "cc:pronunce_2001_oggi"))
    assert "kind" in str(other.value)


def test_connection_failure_is_transient_and_never_shows_the_password(tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:x"))

    def refuse(dsn: str) -> Any:
        msg = f"cannot connect with {dsn}"
        raise OSError(msg)

    with pytest.raises(IngestError) as error:
        run_ingest(params(tmp_path, item("cc", "cc:x")), refuse, lambda *_: None)

    assert error.value.code == TRANSIENT_ERROR
    assert "s3cr3t-pw" not in str(error.value)
    assert "ds_it_giurisprudenza_rw" in str(error.value)


def test_the_connection_string_is_never_logged_or_in_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    caplog.set_level(logging.DEBUG)

    with pytest.raises(IngestError) as error:
        run(tmp_path, FakeConnection(fail_on="INSERT INTO pronunce"), item("cc", "cc:pronunce_2001_oggi"))

    assert "s3cr3t-pw" not in str(error.value)
    assert "s3cr3t-pw" not in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        {"items": []},
        {"items": "x"},
        {"items": [{"key": "k"}]},
        {"items": [{"key": "k", "meta": {"kind": "other", "itemId": "a"}}]},
        {"items": [{"key": "k", "meta": {"kind": "cc"}}]},
        {"dirIn": ""},
        {"connString": ""},
        {"schema": "public"},
        {"schema": "ds_x; DROP SCHEMA public"},
    ],
)
def test_bad_parameters_are_config_errors_before_connecting(tmp_path: Path, bad: dict[str, Any]) -> None:
    calls: list[str] = []

    def connect(dsn: str) -> Any:
        calls.append(dsn)
        return FakeConnection()

    with pytest.raises(IngestError) as error:
        run_ingest({**params(tmp_path, item("cc", "cc:x")), **bad}, connect, lambda *_: None)

    assert error.value.code == CONFIG_ERROR
    assert calls == []


def test_guid_helper_is_a_lowercase_uuid() -> None:
    assert GUID.strip("{}").lower() == UUID
