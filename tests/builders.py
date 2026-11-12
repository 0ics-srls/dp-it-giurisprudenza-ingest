"""Test builders: plan documents (the convert/anonymize output) and a recording fake database connection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ECLI = "ECLI:IT:COST:2026:95"
GUID = "{E69E3100-5261-4D7D-B967-2BC6ABDDAF2A}"
UUID = "e69e3100-5261-4d7d-b967-2bc6abddaf2a"


def new_plan(kind: str, item_id: str, **arrays: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "version": 1, "kind": kind, "itemId": item_id, "counts": {}, "skipped": {},
        "pronunce": [], "massime": [], "norme": [], "citations": [], "cassazione": [], "cgt": [], "cgt_skipped": [],
    }
    plan.update(arrays)
    return plan


def pronuncia(numero: int = 95, anno: int = 2026, fatto: str | None = "Ritenuto in fatto che il giudice.", diritto: str | None = "Considerato in diritto che la questione.") -> dict[str, Any]:
    ecli = f"ECLI:IT:COST:{anno}:{numero}"
    sections = [s for s in ("EPIGRAFE", fatto, diritto, "dispositivo") if s]
    return {
        "ecli": ecli, "anno": anno, "numero": numero, "tipologia": "S", "presidente": "P", "relatore": "R", "redattore": None,
        "data_decisione": "2026-05-04", "data_deposito": "2026-05-20", "collegio": "C", "epigrafe": "EPIGRAFE", "fatto": fatto, "diritto": diritto,
        "testo": None, "dispositivo": "dispositivo", "text_raw": " ".join(sections),
    }


def massima(ecli: str = ECLI, numero: int = 1, text: str = "Principio di diritto.") -> dict[str, Any]:
    return {"ecli": ecli, "numero": numero, "titolo": "Titolo", "text_raw": text}


def norma(tipo: str = "ordinanze", anno: int = 2025, numero: int = 173, ord_: int = 0, urn: str | None = "urn:nir:stato:legge:2022-12-29;197~art1") -> dict[str, Any]:
    return {
        "registro_tipo": tipo, "registro_anno": anno, "registro_numero": numero, "ord": ord_, "norma_descrizione": "legge", "norma_numero": "197",
        "norma_data": "29/12/2022", "norma_articolo": "1", "norma_comma": None, "norma_specificazione_articolo": None, "norma_specificazione_comma": None,
        "norma_nesso": None, "norma_urn": urn,
    }


def citation(ecli: str = ECLI, tipo: str = "ordinanze", anno: int = 2025, numero: int = 173) -> dict[str, Any]:
    return {"ecli": ecli, "registro_tipo": tipo, "registro_anno": anno, "registro_numero": numero}


def sentenza(source_id: str = "snciv2025000001O", text: str = "Testo anonimizzato della sentenza. " * 5, **extra: Any) -> dict[str, Any]:
    record = {
        "source_id": source_id, "kind": "snciv", "anno": 2025, "num_dec": "06063", "sezione": "5", "tipo_prov": "Sentenza",
        "data_decisione": "2025-03-10", "data_deposito": "2025-04-11", "materia": "TRIBUTI", "presidente": "P", "relatore": "R", "ecli": "ECLI:IT:CASS:2025:06063",
        "text_raw_original": "Testo ORIGINALE con nomi. " * 5, "text_anon": text, "ocr_dis": "P.Q.M.", "ocr_dis_anon": "P.Q.M. anon",
        "redaction_report": {"text_anon": {"entities": [], "absoluteCategory": False, "gazetteerHits": 0, "model": "m"}},
        "source_url": "https://www.italgiure.giustizia.it/sncass/#/ricerca?id=" + source_id,
    }
    record.update(extra)
    return record


def cgt(guid: str = GUID, text: str = "SVOLGIMENTO DEL PROCESSO. Testo della sentenza CGT. " * 3, **extra: Any) -> dict[str, Any]:
    record = {
        "guid": guid, "tipo": "Sentenza", "data": "2024-12-31", "grado": 2, "luogo": "Campania", "sezione": "2", "numero": "7355", "anno": 2024,
        "estremi": "Sentenza del 31/12/2024 n. 7355", "text_raw_original": text, "text_anon": text,
        "redaction_report": {"sourceAnonymized": True, "findings": []}, "source_url": "https://def.finanze.it/x?id=" + guid,
    }
    record.update(extra)
    return record


def stem_of(item_id: str) -> str:
    return item_id.replace(":", "_")


def write_plan(directory: Path, plan: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem_of(plan['itemId'])}.plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")


def item(kind: str, item_id: str) -> dict[str, Any]:
    return {"key": f"{item_id}|v1", "meta": {"kind": kind, "itemId": item_id, "version": "1"}}


class FakeCursor:
    def __init__(self, log: list[tuple[str, str, Any]], fail_on: str | None) -> None:
        self._log = log
        self._fail_on = fail_on

    def execute(self, sql: Any, params: Any = None) -> None:
        text = str(sql)
        self._log.append(("execute", text, params))
        if self._fail_on and self._fail_on in text:
            msg = f"boom on {self._fail_on}"
            raise RuntimeError(msg)

    def executemany(self, sql: Any, params: Any) -> None:
        text = str(sql)
        self._log.append(("executemany", text, list(params)))
        if self._fail_on and self._fail_on in text:
            msg = f"boom on {self._fail_on}"
            raise RuntimeError(msg)


class FakeConnection:
    """Records statements and transaction boundaries; ``fail_on`` makes the statement containing that text raise."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.log: list[tuple[str, str, Any]] = []
        self.events: list[str] = []
        self.closed = False
        self._fail_on = fail_on

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.events.append("begin")
        try:
            yield
        except BaseException:
            self.events.append("rollback")
            raise
        self.events.append("commit")

    @contextmanager
    def cursor(self) -> Iterator[FakeCursor]:
        yield FakeCursor(self.log, self._fail_on)

    def execute(self, sql: Any, params: Any = None) -> None:
        FakeCursor(self.log, self._fail_on).execute(sql, params)

    def close(self) -> None:
        self.closed = True
