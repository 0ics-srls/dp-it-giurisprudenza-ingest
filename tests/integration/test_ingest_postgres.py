"""Real Postgres + pgvector (Testcontainers) with the package migrations applied; no Jina call (embeddings are set by SQL)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from testcontainers.community.postgres import PostgresContainer

from builders import (
    ECLI,
    UUID,
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
from dp_it_giurisprudenza_ingest.db import connect
from dp_it_giurisprudenza_ingest.errors import IngestError
from dp_it_giurisprudenza_ingest.ingest import run_ingest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration

SCHEMA = "ds_it_giurisprudenza"
VECTOR = "[" + ",".join(["0.25"] * 1024) + "]"


def migrations_dir() -> Path:
    """The package migrations next to this class repo; resolved lazily so collecting the module in a container (no sibling repo) cannot fail."""
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / "dp-it-giurisprudenza" / "schema" / "migrations", parent.parent / "dp-it-giurisprudenza" / "schema" / "migrations"):
            if candidate.is_dir():
                return candidate
    msg = "package migrations (dp-it-giurisprudenza/schema/migrations) not found next to this repo"
    raise FileNotFoundError(msg)


class Db:
    def __init__(self, container: PostgresContainer) -> None:
        self.dsn = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        self.conn_string = (
            f"Host={container.get_container_host_ip()};Port={container.get_exposed_port(5432)};"
            f"Database={container.dbname};Username={container.username};Password={container.password}"
        )

    def scalar(self, sql: str) -> Any:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            row = conn.execute(sql).fetchone()
            assert row is not None
            return row[0]

    def execute(self, sql: str) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(sql)

    def ingest(self, directory: Path, *items: dict[str, Any]) -> dict[str, Any]:
        params = {"items": list(items), "dirIn": str(directory), "connString": self.conn_string, "schema": SCHEMA}
        return run_ingest(params, connect, lambda *_: None)


@pytest.fixture(scope="module")
def container() -> Iterator[PostgresContainer]:
    with PostgresContainer("pgvector/pgvector:pg17", username="lex", password="pw", dbname="lex") as pg:
        yield pg


@pytest.fixture
def db(container: PostgresContainer) -> Db:
    database = Db(container)
    database.execute("CREATE EXTENSION IF NOT EXISTS vector")
    database.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    database.execute(f"CREATE SCHEMA {SCHEMA}")
    with psycopg.connect(database.dsn, autocommit=True) as conn:
        conn.execute(f"SET search_path TO {SCHEMA}, public")
        for migration in sorted(migrations_dir().glob("*.sql")):
            conn.execute(migration.read_text(encoding="utf-8"))
    return database


def count(db: Db, table: str, where: str = "true") -> int:
    return int(db.scalar(f"SELECT count(*) FROM {SCHEMA}.{table} WHERE {where}"))


PRON = item("cc", "cc:pronunce_2001_oggi")
MASS = item("cc", "cc:massime_2001_oggi")
NORM = item("cc", "cc:norme_ordinanze")


def test_first_ingest_writes_the_decision_its_passages_and_the_source_link(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))

    result = db.ingest(tmp_path, PRON)

    assert count(db, "pronunce") == 1
    assert count(db, "passages", "corpus = 'pronuncia'") == 3
    assert db.scalar(f"SELECT source_url FROM {SCHEMA}.pronunce") == "https://www.cortecostituzionale.it/scheda-pronuncia/2026/95"
    assert db.scalar(f"SELECT count(*) FROM {SCHEMA}.passages WHERE text_contextual LIKE '%EPIGRAFE%'") == 0  # the header is not a passage
    assert result["metrics"][f"written.{SCHEMA}.passages"] == 3


def test_second_ingest_of_the_same_plan_is_a_no_op_and_keeps_the_embeddings(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    db.ingest(tmp_path, PRON)
    db.execute(f"UPDATE {SCHEMA}.passages SET embedding = '{VECTOR}'")

    db.ingest(tmp_path, PRON)

    assert count(db, "passages") == 3
    assert count(db, "passages", "embedding IS NOT NULL") == 3
    assert count(db, "pronunce") == 1


def test_changed_passage_text_nulls_only_that_embedding(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    db.ingest(tmp_path, PRON)
    db.execute(f"UPDATE {SCHEMA}.passages SET embedding = '{VECTOR}'")

    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia(fatto="Ritenuto in fatto un fatto DIVERSO.")]))
    db.ingest(tmp_path, PRON)

    assert count(db, "passages", "embedding IS NULL") == 1
    assert db.scalar(f"SELECT seq FROM {SCHEMA}.passages WHERE embedding IS NULL") == 0
    assert db.scalar(f"SELECT text FROM {SCHEMA}.passages WHERE seq = 0").endswith("DIVERSO.")


def test_a_shrinking_document_deletes_the_surplus_passages(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    db.ingest(tmp_path, PRON)
    assert count(db, "passages") == 3

    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia(fatto=None, diritto=None)]))
    db.ingest(tmp_path, PRON)

    assert count(db, "passages") == 1
    assert count(db, "pronunce") == 1  # the decision itself is never deleted


def test_massime_and_their_passages(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:massime_2001_oggi", massime=[massima(numero=1), massima(numero=2, text="Secondo.")]))
    db.ingest(tmp_path, MASS)
    assert count(db, "massime") == 2
    assert count(db, "passages", f"corpus = 'massima' AND doc_key = '{ECLI}'") == 2

    write_plan(tmp_path, new_plan("cc", "cc:massime_2001_oggi", massime=[massima(numero=1)]))
    db.ingest(tmp_path, MASS)
    assert count(db, "passages", "corpus = 'massima'") == 1  # the massima that disappeared from the plan has no passage left


def test_registry_snapshot_without_an_entry_flips_pending_but_keeps_the_row(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=173), norma(numero=174)]))
    db.ingest(tmp_path, NORM)
    assert count(db, "norme_impugnate", "pending") == 2
    first_seen = db.scalar(f"SELECT first_seen FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 173")

    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=173)]))
    db.ingest(tmp_path, NORM)

    assert count(db, "norme_impugnate") == 2  # never deleted
    assert count(db, "norme_impugnate", "pending") == 1
    assert db.scalar(f"SELECT pending FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 174") is False
    assert db.scalar(f"SELECT first_seen FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 173") == first_seen
    assert db.scalar(f"SELECT last_seen > first_seen FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 173") is True


def test_a_reappearing_registry_entry_is_pending_again(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=174)]))
    db.ingest(tmp_path, NORM)
    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze"))
    db.ingest(tmp_path, NORM)
    assert count(db, "norme_impugnate", "pending") == 0

    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=174)]))
    db.ingest(tmp_path, NORM)

    assert count(db, "norme_impugnate", "pending") == 1


def test_other_registries_are_not_touched_by_a_snapshot(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:norme_ricorsi", norme=[norma(tipo="ricorsi", numero=5)]))
    db.ingest(tmp_path, item("cc", "cc:norme_ricorsi"))

    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=173)]))
    db.ingest(tmp_path, NORM)

    assert count(db, "norme_impugnate", "registro_tipo = 'ricorsi' AND pending") == 1


def test_citations_resolve_the_ecli_only_when_the_decision_exists_and_a_reload_keeps_it(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=173), norma(numero=999)]))
    db.ingest(tmp_path, NORM)
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()], citations=[citation(), citation(numero=999, ecli="ECLI:IT:COST:2026:1")]))

    db.ingest(tmp_path, PRON)

    assert db.scalar(f"SELECT ecli FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 173") == ECLI
    assert db.scalar(f"SELECT ecli_via FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 173") == "epigrafe"
    assert db.scalar(f"SELECT ecli FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 999") is None  # that decision is not loaded
    write_plan(tmp_path, new_plan("cc", "cc:norme_ordinanze", norme=[norma(numero=173)]))
    db.ingest(tmp_path, NORM)
    assert db.scalar(f"SELECT ecli FROM {SCHEMA}.norme_impugnate WHERE registro_numero = 173") == ECLI  # a registry reload keeps the link


def test_cassazione_stores_both_texts_and_passages_of_the_anonymized_one(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cassazione", "cass:snciv:2025:01", cassazione=[sentenza(), sentenza("snciv2025000002O", ecli="ECLI:IT:CASS:2025:06063")]))

    db.ingest(tmp_path, item("cassazione", "cass:snciv:2025:01"))

    assert count(db, "cassazione_sentenze") == 2
    assert count(db, "cassazione_sentenze", "ecli = 'ECLI:IT:CASS:2025:06063'") == 2  # the ECLI is not unique
    assert db.scalar(f"SELECT num_dec FROM {SCHEMA}.cassazione_sentenze LIMIT 1") == 6063
    assert "ORIGINALE" in db.scalar(f"SELECT text_raw_original FROM {SCHEMA}.cassazione_sentenze LIMIT 1")
    assert count(db, "passages", "corpus = 'cassazione' AND text LIKE '%ORIGINALE%'") == 0
    assert db.scalar(f"SELECT redaction_report->'text_anon'->>'model' FROM {SCHEMA}.cassazione_sentenze LIMIT 1") == "m"  # jsonb, not a text blob


def test_cgt_decisions_skips_and_the_skip_cleared_once_the_text_arrives(db: Db, tmp_path: Path) -> None:
    skipped = {"guid": "{E69E3100-5261-4D7D-B967-2BC6ABDDAF2A}", "estremi": "Sentenza del 31/12/2024", "reason": "no_text"}
    write_plan(tmp_path, new_plan("cgt", "cgt:a", cgt_skipped=[skipped]))
    db.ingest(tmp_path, item("cgt", "cgt:a"))
    assert count(db, "cgt_skipped") == 1

    write_plan(tmp_path, new_plan("cgt", "cgt:a", cgt=[cgt()]))
    db.ingest(tmp_path, item("cgt", "cgt:a"))

    assert count(db, "cgt_sentenze", f"id = '{UUID}'") == 1
    assert count(db, "cgt_skipped") == 0
    assert count(db, "passages", f"corpus = 'cgt' AND doc_key = '{UUID}'") >= 1


def test_quarantined_documents_are_not_written(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cgt", "cgt:q", cgt=[cgt()], quarantine=[{"id": "{BBBB}", "reason": "pii_guardrail", "findings": {"cf": 1}}]))

    result = db.ingest(tmp_path, item("cgt", "cgt:q"))

    assert count(db, "cgt_sentenze") == 1
    assert result["metrics"]["quarantined"] == 1


def test_failure_in_the_middle_rolls_back_the_item(db: Db, tmp_path: Path) -> None:
    bad = massima(ecli="ECLI:IT:COST:2026:1", text="x")
    bad["numero"] = "not-a-number"  # the INSERT fails after the pronuncia rows were written in the same transaction
    write_plan(tmp_path, new_plan("cc", "cc:massime_2001_oggi", massime=[massima(), bad]))

    with pytest.raises(IngestError):
        db.ingest(tmp_path, MASS)

    assert count(db, "massime") == 0
    assert count(db, "passages") == 0


def test_two_items_in_one_run_and_a_failing_one_does_not_undo_the_other(db: Db, tmp_path: Path) -> None:
    write_plan(tmp_path, new_plan("cc", "cc:pronunce_2001_oggi", pronunce=[pronuncia()]))
    broken = massima()
    broken["numero"] = "x"
    write_plan(tmp_path, new_plan("cc", "cc:massime_2001_oggi", massime=[broken]))

    with pytest.raises(IngestError):
        db.ingest(tmp_path, MASS, PRON)

    assert count(db, "pronunce") == 1
    assert count(db, "massime") == 0
