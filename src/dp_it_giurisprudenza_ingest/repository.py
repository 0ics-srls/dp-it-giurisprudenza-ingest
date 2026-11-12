"""SQL of one item (plan file v1): decisions, massime, passages, registries, Cassazione and CGT sentences.

All statements run inside one transaction opened by the caller; the dataset schema is on the search_path (names are unqualified). The rules:

* decisions, massime and sentences are upserted (a changed text replaces the old one; the text columns of the posture tables change together);
* passages are upserted on ``(corpus, doc_key, seq)``: the embedding is KEPT when ``text_contextual`` is unchanged and set NULL when it changed, and the
  passages of a document beyond its current set are deleted (P79-07/08);
* the norme impugnate registries are ACCUMULATIVE (P79-12): ``pending`` marks the latest snapshot, rows are never deleted, ``first_seen`` and a resolved
  decision link survive a reload;
* the link ``norme_impugnate.ecli`` comes from the ``citations`` of a decision plan only, and only when that decision exists.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Protocol

from psycopg.types.json import Jsonb

from dp_it_giurisprudenza_ingest.chunkers import Piece, chunk_cassazione, chunk_cgt, chunk_massima, chunk_pronuncia

if TYPE_CHECKING:
    from collections.abc import Iterable

PLAN_SCHEMA_VERSION = 1
TABLES = ("pronunce", "massime", "passages", "norme_impugnate", "cassazione_sentenze", "cgt_sentenze", "cgt_skipped")

PRONUNCE_COLUMNS = (
    "ecli", "anno", "numero", "tipologia", "presidente", "relatore", "redattore", "data_decisione", "data_deposito", "collegio", "epigrafe", "fatto",
    "diritto", "testo", "dispositivo", "text_raw",
)
NORME_COLUMNS = (
    "registro_tipo", "registro_anno", "registro_numero", "ord", "norma_descrizione", "norma_numero", "norma_data", "norma_articolo", "norma_comma",
    "norma_specificazione_articolo", "norma_specificazione_comma", "norma_nesso", "norma_urn",
)
CASSAZIONE_COLUMNS = (
    "source_id", "kind", "ecli", "sezione", "anno", "num_dec", "data_decisione", "data_deposito", "presidente", "relatore", "materia", "tipo_prov",
    "text_raw_original", "text_anon", "ocr_dis_anon", "redaction_report", "source_url",
)
CGT_COLUMNS = (
    "id", "tipo", "data", "grado", "luogo", "sezione", "numero", "anno", "estremi", "text_raw_original", "text_anon", "redaction_report", "source_url",
)


def _placeholders(columns: tuple[str, ...]) -> str:
    return ", ".join(["%s"] * len(columns))


def _assignments(columns: tuple[str, ...], skip: int = 1) -> str:
    return ", ".join(f"{c} = EXCLUDED.{c}" for c in columns[skip:])


UPSERT_PRONUNCE = f"""
INSERT INTO pronunce ({", ".join(PRONUNCE_COLUMNS)})
VALUES ({_placeholders(PRONUNCE_COLUMNS)})
ON CONFLICT (ecli) DO UPDATE SET {_assignments(PRONUNCE_COLUMNS)}
"""

UPSERT_MASSIME = """
INSERT INTO massime (ecli, numero, titolo, text_raw) VALUES (%s, %s, %s, %s)
ON CONFLICT (ecli, numero) DO UPDATE SET titolo = EXCLUDED.titolo, text_raw = EXCLUDED.text_raw
"""

# Text changed -> the embedding is stale: NULL it; identical contextual text keeps it.
UPSERT_PASSAGE = """
INSERT INTO passages (corpus, doc_key, seq, kind, text, text_contextual) VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (corpus, doc_key, seq) DO UPDATE SET
    kind = EXCLUDED.kind,
    text = EXCLUDED.text,
    embedding = CASE WHEN passages.text_contextual IS DISTINCT FROM EXCLUDED.text_contextual THEN NULL ELSE passages.embedding END,
    text_contextual = EXCLUDED.text_contextual
"""
DELETE_SURPLUS_PASSAGES = "DELETE FROM passages WHERE corpus = %s AND doc_key = %s AND NOT (seq = ANY(%s::int[]))"

UPSERT_NORME = f"""
INSERT INTO norme_impugnate ({", ".join(NORME_COLUMNS)})
VALUES ({_placeholders(NORME_COLUMNS)})
ON CONFLICT (registro_tipo, registro_anno, registro_numero, ord) DO UPDATE SET
    {_assignments(NORME_COLUMNS, skip=4)}, pending = true, last_seen = now()
"""
FLIP_ABSENT_NORME = "UPDATE norme_impugnate SET pending = false WHERE registro_tipo = %s AND pending AND last_seen < now()"
LINK_CITATION = (
    "UPDATE norme_impugnate SET ecli = %s, ecli_via = %s WHERE registro_tipo = %s AND registro_anno = %s AND registro_numero = %s "
    "AND EXISTS (SELECT 1 FROM pronunce WHERE ecli = %s)"
)

UPSERT_CASSAZIONE = f"""
INSERT INTO cassazione_sentenze ({", ".join(CASSAZIONE_COLUMNS)}, ingested_at)
VALUES ({_placeholders(CASSAZIONE_COLUMNS)}, now())
ON CONFLICT (source_id) DO UPDATE SET {_assignments(CASSAZIONE_COLUMNS[:12])}, text_raw_original = EXCLUDED.text_raw_original, text_anon = EXCLUDED.text_anon,
    {_assignments(CASSAZIONE_COLUMNS[14:], skip=0)}, ingested_at = EXCLUDED.ingested_at
"""

UPSERT_CGT = f"""
INSERT INTO cgt_sentenze ({", ".join(CGT_COLUMNS)}, ingested_at)
VALUES ({_placeholders(CGT_COLUMNS)}, now())
ON CONFLICT (id) DO UPDATE SET {_assignments(CGT_COLUMNS[:9])}, text_raw_original = EXCLUDED.text_raw_original, text_anon = EXCLUDED.text_anon,
    {_assignments(CGT_COLUMNS[11:], skip=0)}, ingested_at = EXCLUDED.ingested_at
"""
CLEAR_SKIPPED = "DELETE FROM cgt_skipped WHERE guid = ANY(%s::uuid[])"
UPSERT_SKIPPED = """
INSERT INTO cgt_skipped (guid, estremi, reason) VALUES (%s, %s, %s)
ON CONFLICT (guid) DO UPDATE SET estremi = EXCLUDED.estremi, reason = EXCLUDED.reason, seen_at = now()
"""


class Cursor(Protocol):
    def execute(self, sql: Any, params: Any = None) -> Any: ...

    def executemany(self, sql: Any, params: Iterable[Any]) -> Any: ...


def _int_or_none(value: Any) -> int | None:
    text = None if value is None else str(value).strip()
    return int(text) if text and text.isdigit() else None


def _guid(value: str) -> str:
    return str(uuid.UUID(value.strip("{}")))


def _json(value: Any) -> Jsonb | None:
    return None if value is None else Jsonb(value)


def replace_passages(cursor: Cursor, documents: dict[tuple[str, str], list[Piece]]) -> int:
    """Upserts the passages of every document in one batch, then deletes the passages of each document beyond its current set."""
    rows = [(corpus, key, p.seq, p.kind, p.text, p.text_contextual) for (corpus, key), pieces in documents.items() for p in pieces]
    if rows:
        cursor.executemany(UPSERT_PASSAGE, rows)
    for (corpus, key), pieces in documents.items():
        cursor.execute(DELETE_SURPLUS_PASSAGES, (corpus, key, [p.seq for p in pieces]))
    return len(rows)


def write_pronunce(cursor: Cursor, plan: dict[str, Any]) -> dict[str, int]:
    pronunce: list[dict[str, Any]] = plan["pronunce"]
    if not pronunce:
        return {}
    cursor.executemany(UPSERT_PRONUNCE, [tuple(p.get(c) for c in PRONUNCE_COLUMNS) for p in pronunce])
    documents = {
        ("pronuncia", p["ecli"]): chunk_pronuncia(
            p["tipologia"], p["numero"], p["anno"], p.get("fatto"), p.get("diritto"), p.get("testo"), p.get("dispositivo"))
        for p in pronunce
    }
    counts = {"pronunce": len(pronunce), "passages": replace_passages(cursor, documents)}
    citations = [(c["ecli"], "epigrafe", c["registro_tipo"], c["registro_anno"], c["registro_numero"], c["ecli"]) for c in plan["citations"]]
    if citations:
        cursor.executemany(LINK_CITATION, citations)
    return counts


def write_massime(cursor: Cursor, plan: dict[str, Any]) -> dict[str, int]:
    massime: list[dict[str, Any]] = plan["massime"]
    if not massime:
        return {}
    cursor.executemany(UPSERT_MASSIME, [(m["ecli"], m["numero"], m.get("titolo"), m["text_raw"]) for m in massime])
    documents: dict[tuple[str, str], list[Piece]] = {}
    for m in massime:
        documents.setdefault(("massima", m["ecli"]), []).extend(chunk_massima(m["ecli"], m["numero"], m["text_raw"]))
    return {"massime": len(massime), "passages": replace_passages(cursor, documents)}


def write_norme(cursor: Cursor, plan: dict[str, Any]) -> dict[str, int]:
    """The registry snapshot of this item: upsert the rows, then flip the rows of the same registry that the snapshot no longer holds."""
    tipo = plan["itemId"].removeprefix("cc:norme_")
    rows = [tuple(n.get(c) for c in NORME_COLUMNS) for n in plan["norme"]]
    if rows:
        cursor.executemany(UPSERT_NORME, rows)
    cursor.execute(FLIP_ABSENT_NORME, (tipo,))
    return {"norme_impugnate": len(rows)}


def write_cc(cursor: Cursor, plan: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if plan["itemId"].startswith("cc:norme_"):
        counts.update(write_norme(cursor, plan))
    for part in (write_pronunce(cursor, plan), write_massime(cursor, plan)):
        for table, value in part.items():
            counts[table] = counts.get(table, 0) + value
    return counts


def _require_anonymized(record: dict[str, Any], identity: str) -> str:
    text = record.get("text_anon")
    if not isinstance(text, str) or not text.strip():
        msg = f"{identity}: text_anon is missing, the plan was not anonymized"
        raise ValueError(msg)
    return text


def write_cassazione(cursor: Cursor, plan: dict[str, Any]) -> tuple[dict[str, int], int]:
    """Returns the row counts and the number of sentences decided after they were deposited (a data warning, never a refusal)."""
    records: list[dict[str, Any]] = plan["cassazione"]
    if not records:
        return {}, 0
    rows = []
    documents: dict[tuple[str, str], list[Piece]] = {}
    warnings = 0
    for r in records:
        text_anon = _require_anonymized(r, r["source_id"])
        rows.append((
            r["source_id"], r["kind"], r.get("ecli"), r.get("sezione"), r["anno"], _int_or_none(r.get("num_dec")),
            r.get("data_decisione"), r.get("data_deposito"),
            r.get("presidente"), r.get("relatore"), r.get("materia"), r.get("tipo_prov"), r["text_raw_original"], text_anon, r.get("ocr_dis_anon"),
            _json(r.get("redaction_report")), r["source_url"],
        ))
        documents[("cassazione", r["source_id"])] = chunk_cassazione(text_anon, r["kind"], r.get("sezione"), r.get("num_dec"), r["anno"]).pieces
        if r.get("data_decisione") and r.get("data_deposito") and r["data_decisione"] > r["data_deposito"]:
            warnings += 1
    cursor.executemany(UPSERT_CASSAZIONE, rows)
    return {"cassazione_sentenze": len(rows), "passages": replace_passages(cursor, documents)}, warnings


def write_cgt(cursor: Cursor, plan: dict[str, Any]) -> dict[str, int]:
    records: list[dict[str, Any]] = plan["cgt"]
    skipped: list[dict[str, Any]] = plan["cgt_skipped"]
    counts: dict[str, int] = {}
    if records:
        rows = []
        documents: dict[tuple[str, str], list[Piece]] = {}
        for r in records:
            guid = _guid(r["guid"])
            text_anon = _require_anonymized(r, r["guid"])
            rows.append((
                guid, r["tipo"], r.get("data"), r["grado"], r.get("luogo"), r.get("sezione"), _int_or_none(r.get("numero")), r["anno"], r["estremi"],
                r["text_raw_original"], text_anon, _json(r.get("redaction_report")), r["source_url"],
            ))
            documents[("cgt", guid)] = chunk_cgt(text_anon, r["grado"], r.get("luogo"), r.get("sezione"), r.get("numero"), r["anno"]).pieces
        cursor.executemany(UPSERT_CGT, rows)
        counts = {"cgt_sentenze": len(rows), "passages": replace_passages(cursor, documents)}
        cursor.execute(CLEAR_SKIPPED, ([row[0] for row in rows],))  # a decision that now has text is no longer "without text"
    if skipped:
        cursor.executemany(UPSERT_SKIPPED, [(_guid(s["guid"]), s.get("estremi"), s["reason"]) for s in skipped])
        counts["cgt_skipped"] = len(skipped)
    return counts
