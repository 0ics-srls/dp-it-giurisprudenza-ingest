"""Sentence-boundary segmenter: golden cases ported from the legacy TextSegmenterTests."""
from __future__ import annotations

from dp_it_giurisprudenza_ingest.segmenter import MAX_CHARS, MIN_CHARS, TARGET_CHARS, segment


def test_constants() -> None:
    assert (TARGET_CHARS, MIN_CHARS, MAX_CHARS) == (3000, 500, 8000)


def test_sentence_boundaries_concatenation_is_exactly_the_input() -> None:
    text = "".join(f"La frase numero {i} del documento porta un contenuto di prova. " for i in range(1, 51))

    pieces = segment(text)

    assert pieces
    assert "".join(pieces) == text


def test_twenty_thousand_chars_no_piece_over_max() -> None:
    sentence = "Il collegio osserva che la questione dedotta e' infondata per le ragioni esposte. "
    text = sentence * 250
    assert len(text) > 20000

    pieces = segment(text)

    assert all(len(p) <= MAX_CHARS for p in pieces)
    assert "".join(pieces) == text


def test_short_tail_is_merged_with_the_previous_piece() -> None:
    big = "a" * 2900 + ". "
    stub = "b" * 300 + "."

    pieces = segment(big + stub)

    assert pieces
    if len(pieces) > 1:
        assert all(len(p) >= MIN_CHARS for p in pieces)
    assert pieces[-1].endswith(stub)
    assert "".join(pieces) == big + stub


def test_oversize_text_splits_at_sentence_boundaries_into_target_sized_pieces() -> None:
    sentence = "x" * 99 + ". "
    text = sentence * 100  # 10,000 characters

    pieces = segment(text)

    assert len(pieces) >= 3
    assert all(len(p) <= MAX_CHARS for p in pieces)
    assert all(p.endswith(". ") for p in pieces)  # every cut is on a sentence boundary
    assert "".join(pieces) == text


def test_no_sentence_boundary_single_piece() -> None:
    text = "x" * 4000

    assert segment(text) == [text]


def test_no_boundary_over_max_is_hard_split() -> None:
    text = "y" * 9000

    pieces = segment(text)

    assert all(len(p) <= MAX_CHARS for p in pieces)
    assert "".join(pieces) == text


def test_empty_returns_empty_list() -> None:
    assert segment("") == []
