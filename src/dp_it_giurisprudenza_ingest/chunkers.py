"""Passage chunkers (P79-07): the embeddable pieces of every corpus.

A header (the epigrafe of a pronuncia, the procedural head of a sentence) is NOT a passage: the embed class embeds every row with an empty vector
and has no filter, and the header text already lives in the parent row. ``seq`` restarts per document and counts only stored passages.
The context prefixes are the legacy ones, so the embedded text stays comparable with the legacy chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from dp_it_giurisprudenza_ingest.segmenter import segment

# Substantial-section markers, enumerated from real data (legacy CassazioneChunker). Matched case-insensitively and anchored to a boundary
# (start of text, or after . ; : or a newline), so "va considerato che" in the middle of a sentence is not taken for a section start.
SECTION_MARKERS = (
    "RITENUTO IN FATTO",
    "FATTI DI CAUSA",
    "SVOLGIMENTO DEL PROCESSO",
    "CONSIDERATO IN DIRITTO",
    "RAGIONI DELLA DECISIONE",
    "MOTIVI DELLA DECISIONE",
    "RILEVATO CHE",
    "CONSIDERATO CHE",
)
_PDFBOX = re.compile(r"^\s*PDFBox:\s*")
_BOUNDARY_BEFORE = re.compile(r"[.;:\n]\s*$")
_MARKER_PATTERN = re.compile("|".join(re.escape(m) for m in SECTION_MARKERS), re.IGNORECASE)


@dataclass(frozen=True)
class Piece:
    """One stored passage."""

    seq: int
    kind: str
    text: str
    text_contextual: str


@dataclass(frozen=True)
class Chunked:
    """Passages of a sentence plus what the chunker dropped: the procedural header (not stored) and whether a section marker was found."""

    pieces: list[Piece]
    header: str
    marker_found: bool


def chunk_pronuncia(  # noqa: PLR0913, PLR0917
    tipologia: str,
    numero: int,
    anno: int,
    fatto: str | None,
    diritto: str | None,
    testo: str | None,
    dispositivo: str | None,
) -> list[Piece]:
    """Sections fatto, diritto, testo, dispositivo in order, each segmented; the epigrafe is a header and not a passage."""
    pieces: list[Piece] = []
    for name, text in (("fatto", fatto), ("diritto", diritto), ("testo", testo), ("dispositivo", dispositivo)):
        if text is None or not text.strip():
            continue
        prefix = f"[Corte Cost. {tipologia} n. {numero}/{anno} — {name}]\n"
        pieces.extend(Piece(len(pieces), "body", part, prefix + part) for part in segment(text))
    return pieces


def chunk_massima(ecli: str, numero: int, text_raw: str) -> list[Piece]:  # noqa: ARG001 - the ecli is the document key of the caller
    """A massima is one passage: its seq is the massima number and the embedded text equals text_raw."""
    if not text_raw.strip():
        return []
    return [Piece(numero, "body", text_raw, text_raw)]


def _first_marker(text: str) -> int:
    best = -1
    for match in _MARKER_PATTERN.finditer(text):
        index = match.start()
        if (index == 0 or _BOUNDARY_BEFORE.search(text[:index])) and (best == -1 or index < best):
            best = index
    return best


def chunk_cassazione(text_anon: str, kind: str, sezione: str | None, num_dec: str | None, anno: int) -> Chunked:
    """Splits the anonymized text of a sentence: header (start to the first boundary-anchored section marker, dropped) and body passages."""
    if not text_anon.strip():
        return Chunked([], "", marker_found=False)
    stripped = _PDFBOX.sub("", text_anon)
    if not stripped.strip():
        return Chunked([], "", marker_found=False)
    prefix = f"[Cass. {kind} Sez. {sezione} n. {num_dec}/{anno}]\n"
    marker = _first_marker(stripped)
    header = stripped[:marker] if marker > 0 else ""
    body = stripped[marker:] if marker > 0 else stripped
    pieces = [Piece(seq, "body", part, prefix + part) for seq, part in enumerate(segment(body))]
    return Chunked(pieces, header, marker_found=marker >= 0)


def chunk_cgt(text_anon: str, grado: int, luogo: str | None, sezione: str | None, numero: str | None, anno: int) -> Chunked:  # noqa: PLR0913, PLR0917
    """CGT decisions use the Cassazione chunker with the label ``CGT-<grado> <luogo>`` in place of the kind (legacy CgtChunkBackfillService)."""
    return chunk_cassazione(text_anon, f"CGT-{grado} {luogo}", sezione, numero, anno)
