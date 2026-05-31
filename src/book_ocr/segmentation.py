from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


LETTER_SEPARATOR = "-----"


def _normalize_paragraphs(text: str) -> str:
    """Insert blank lines around letter structural elements the LLM tends to merge."""
    lines = text.split("\n")
    result: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        prev = lines[i - 1].strip() if i > 0 else ""

        result.append(line)

        is_header = bool(re.match(
            r"^[A-ZÁ-Ž][a-záčďéěíňóřšťúůýž]+,\s+\d{1,2}\.\s+"
            r"[a-záčďéěíňóřšťúůýž]+\s+\d{4}$", stripped
        ))
        is_greeting = bool(re.match(
            r"^(Drahá|Milá|Vážená)\s+[A-ZÁ-Ž]", stripped
        ))
        is_closing = stripped.lower() in (
            "s úctou", "s pozdravem", "vaše", "tvá", "s láskou",
        )
        is_signature = bool(re.match(
            r"^[A-ZÁ-Ž][a-záčďéěíňóřšťúůýž]+\s+[A-ZÁ-Ž]\.\s+[A-ZÁ-Ž]\.$", stripped
        ))

        if is_header:
            result.append("")
        elif is_greeting and i > 0:
            result.append("")
        elif is_closing:
            result.append("")
        elif is_signature:
            result.append("")

    return "\n".join(result)


def segment_text(text: str) -> list[dict[str, Any]]:
    """Split full manuscript text into logical segments matching milena.json structure."""
    text = _normalize_paragraphs(text)
    segments: list[dict[str, Any]] = []
    raw_blocks = re.split(r"\n\n+", text.strip())

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        if block == LETTER_SEPARATOR:
            segments.append({
                "source": LETTER_SEPARATOR,
                "target": LETTER_SEPARATOR,
                "status": "approved",
            })
            continue

        sub_segments = _split_long_block(block)
        for sub in sub_segments:
            segments.append({
                "source": sub,
                "target": "",
                "status": "",
            })

    return segments


def _split_long_block(block: str, max_chars: int = 600) -> list[str]:
    """Split a long block into smaller sentence-level units."""
    if len(block) <= max_chars:
        return [block]

    parts: list[str] = []
    pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = pattern.split(block)

    current = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current += " " + sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)

    return parts if parts else [block]


def write_segments_json(segments: list[dict[str, Any]], output_path: Path) -> None:
    data = {
        "title": "",
        "source_lang": "cz",
        "target_lang": "ru",
        "translator": "",
        "segments": segments,
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
