"""Sentence-boundary segmentation shared by every chunker (port of the legacy TextSegmenter).

Accumulates sentences into pieces of about ``TARGET_CHARS``, merges a tail shorter than ``MIN_CHARS`` into the previous piece and hard-cuts
anything above ``MAX_CHARS`` (the embedding input limit). Lossless: ``"".join(segment(text)) == text``.
"""
from __future__ import annotations

import re

TARGET_CHARS = 3000
MIN_CHARS = 500
MAX_CHARS = 8000

_SENTENCE_END = re.compile(r"\.\s+")


def _split_sentences(text: str) -> list[str]:
    """Cut after every "dot + whitespace"; each segment keeps its own delimiter so the concatenation reproduces the input."""
    segments: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        segments.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        segments.append(text[start:])
    return segments


def segment(text: str) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if current and len(current) + len(sentence) > TARGET_CHARS and len(current) >= MIN_CHARS:
            pieces.append(current)
            current = ""
        current += sentence
    if current:
        if pieces and len(current) < MIN_CHARS:
            pieces[-1] += current
        else:
            pieces.append(current)

    capped: list[str] = []
    for piece in pieces:
        if len(piece) <= MAX_CHARS:
            capped.append(piece)
            continue
        capped.extend(piece[i : i + MAX_CHARS] for i in range(0, len(piece), MAX_CHARS))
    return capped
