"""Passage chunkers: pronuncia, massima, Cassazione and CGT (ports of PronunciaChunker, CassazioneChunker and the CGT backfill)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dp_it_giurisprudenza_ingest.chunkers import chunk_cassazione, chunk_cgt, chunk_massima, chunk_pronuncia
from dp_it_giurisprudenza_ingest.segmenter import MAX_CHARS

# file, kind, sezione, num_dec, anno, header length (0 = no boundary-anchored marker at a position > 0), marker found at all
GOLDEN = [
    ("considerato_snpen_1_06063_2025.txt", "snpen", "1", "06063", 2025, 2400, True),
    ("fatti_snciv_3_21631_2021.txt", "snciv", "3", "21631", 2021, 1059, True),
    ("motivi_snciv_L_22186_2021.txt", "snciv", "L", "22186", 2021, 1858, True),
    ("nomarker_snciv_5_25994_2021.txt", "snciv", "5", "25994", 2021, 0, False),
    ("oversize_snciv_2_23082_2021.txt", "snciv", "2", "23082", 2021, 0, False),
    ("pdfbox_snciv_5_02123_2026.txt", "snciv", "5", "02123", 2026, 605, True),
    ("ragioni_snciv_5_21495_2021.txt", "snciv", "5", "21495", 2021, 855, True),
    ("rilevato_snciv_5_20663_2021.txt", "snciv", "5", "20663", 2021, 711, True),
    ("ritenuto_snpen_7_32009_2025.txt", "snpen", "7", "32009", 2025, 235, True),
    ("svolgimento_snciv_L_31345_2021.txt", "snciv", "L", "31345", 2021, 1709, True),
]


def _fixture(*parts: str) -> str:
    return (Path(__file__).resolve().parent.parent.joinpath("fixtures", *parts)).read_text(encoding="utf-8")


def _strip_pdfbox(text: str) -> str:
    return re.sub(r"^\s*PDFBox:\s*", "", text)


def test_pronuncia_sections_in_order_with_prefix() -> None:
    pieces = chunk_pronuncia("S", 1, 2001, fatto="Ritenuto in fatto che il giudice.", diritto="Considerato in diritto che la questione.",
                             testo="La motivazione.", dispositivo="dichiara non fondata la questione")

    assert [p.seq for p in pieces] == [0, 1, 2, 3]
    assert [p.kind for p in pieces] == ["body"] * 4
    assert pieces[0].text == "Ritenuto in fatto che il giudice."
    assert pieces[0].text_contextual == "[Corte Cost. S n. 1/2001 — fatto]\nRitenuto in fatto che il giudice."
    assert pieces[1].text_contextual.startswith("[Corte Cost. S n. 1/2001 — diritto]\n")
    assert pieces[2].text_contextual.startswith("[Corte Cost. S n. 1/2001 — testo]\n")
    assert pieces[3].text_contextual.startswith("[Corte Cost. S n. 1/2001 — dispositivo]\n")


def test_pronuncia_without_fatto_diritto() -> None:
    pieces = chunk_pronuncia("O", 5, 2020, fatto=None, diritto="  ", testo="Corpo dell'ordinanza.", dispositivo="Per questi motivi.")

    assert [(p.seq, p.text_contextual.split("\n")[0]) for p in pieces] == [
        (0, "[Corte Cost. O n. 5/2020 — testo]"),
        (1, "[Corte Cost. O n. 5/2020 — dispositivo]"),
    ]


def test_pronuncia_long_section_is_segmented_and_loses_nothing() -> None:
    diritto = "Il collegio osserva che la questione e' fondata per le ragioni esposte. " * 200

    pieces = chunk_pronuncia("S", 9, 2024, fatto=None, diritto=diritto, testo=None, dispositivo=None)

    assert len(pieces) > 1
    assert "".join(p.text for p in pieces) == diritto
    assert [p.seq for p in pieces] == list(range(len(pieces)))
    assert all(len(p.text) <= MAX_CHARS for p in pieces)


def test_pronuncia_without_any_section_has_no_passage() -> None:
    assert chunk_pronuncia("S", 1, 2001, fatto=None, diritto=None, testo=None, dispositivo=None) == []


def test_massima_single_passage_text_equals_raw() -> None:
    pieces = chunk_massima("ECLI:IT:COST:2001:10", 2, "Principio di diritto affermato dalla Corte.")

    assert len(pieces) == 1
    assert pieces[0].seq == 2  # the passage seq is the massima number
    assert pieces[0].kind == "body"
    assert pieces[0].text == "Principio di diritto affermato dalla Corte."
    assert pieces[0].text_contextual == pieces[0].text  # the embedded text equals the legacy TextRaw


def test_massima_blank_has_no_passage() -> None:
    assert chunk_massima("ECLI:IT:COST:2001:10", 1, "   ") == []


@pytest.mark.parametrize(("name", "kind", "sezione", "num_dec", "anno", "header_len", "marker"), GOLDEN)
def test_cassazione_fixtures(name: str, kind: str, sezione: str, num_dec: str, anno: int, header_len: int, marker: bool) -> None:
    raw = _fixture("cassazione", name)

    result = chunk_cassazione(raw, kind, sezione, num_dec, anno)

    assert len(result.header) == header_len
    assert result.marker_found is marker  # no marker => no header, everything is body, and the caller can count it
    assert result.pieces
    assert [p.seq for p in result.pieces] == list(range(len(result.pieces))), "seq restarts per document and counts only stored passages"
    assert all(p.kind == "body" for p in result.pieces)
    assert all(len(p.text) <= MAX_CHARS for p in result.pieces)
    prefix = f"[Cass. {kind} Sez. {sezione} n. {num_dec}/{anno}]\n"
    assert all(p.text_contextual == prefix + p.text for p in result.pieces)


def test_cassazione_ritenuto_body_starts_at_the_marker() -> None:
    raw = _fixture("cassazione", "ritenuto_snpen_7_32009_2025.txt")

    result = chunk_cassazione(raw, "snpen", "7", "32009", 2025)

    assert result.pieces[0].text.startswith("RITENUTO IN FATTO")


def test_cassazione_pdfbox_prefix_is_stripped() -> None:
    raw = _fixture("cassazione", "pdfbox_snciv_5_02123_2026.txt")
    assert raw.startswith("PDFBox:")

    result = chunk_cassazione(raw, "snciv", "5", "02123", 2026)

    assert "PDFBox:" not in result.header + "".join(p.text for p in result.pieces)


@pytest.mark.parametrize(("name", "kind", "sezione", "num_dec", "anno", "header_len", "marker"), GOLDEN)
def test_cassazione_no_text_lost_after_header(name: str, kind: str, sezione: str, num_dec: str, anno: int, header_len: int, marker: bool) -> None:
    raw = _fixture("cassazione", name)

    result = chunk_cassazione(raw, kind, sezione, num_dec, anno)

    assert result.header + "".join(p.text for p in result.pieces) == _strip_pdfbox(raw)
    assert (header_len, marker) == (len(result.header), result.marker_found)


def test_cassazione_blank_text_has_no_passage() -> None:
    result = chunk_cassazione("   \n\t ", "snciv", "1", "00001", 2025)

    assert result.pieces == []
    assert result.header == ""


def test_cassazione_marker_inside_a_sentence_is_not_a_section_start() -> None:
    text = "Il ricorrente ritiene che va considerato che la domanda sia fondata. " * 30

    result = chunk_cassazione(text, "snciv", "1", "00001", 2025)

    assert result.marker_found is False
    assert result.header == ""


def test_cgt_fixture_cgt_7355() -> None:
    raw = _fixture("cgt", "cgt-7355-textraw.txt")

    result = chunk_cgt(raw, grado=2, luogo="Campania", sezione="2", numero="7355", anno=2024)

    assert result.marker_found is True  # "SVOLGIMENTO DEL PROCESSO" is the very first line: no header, all body
    assert result.header == ""
    assert "".join(p.text for p in result.pieces) == raw
    assert result.pieces[0].text_contextual.startswith("[Cass. CGT-2 Campania Sez. 2 n. 7355/2024]\n")
    assert [p.seq for p in result.pieces] == list(range(len(result.pieces)))


def test_chunks_are_deterministic() -> None:
    raw = _fixture("cassazione", "oversize_snciv_2_23082_2021.txt")

    first = chunk_cassazione(raw, "snciv", "2", "23082", 2021)
    second = chunk_cassazione(raw, "snciv", "2", "23082", 2021)

    assert first == second
    assert chunk_pronuncia("S", 1, 2001, "a. b.", "c. d.", None, "e.") == chunk_pronuncia("S", 1, 2001, "a. b.", "c. d.", None, "e.")
