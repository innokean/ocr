from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    print("Missing 'openai' package. Run: uv add openai")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

SEGMENTS_SOURCE = "segments"


def _build_client():
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
    if az_key and az_endpoint:
        return (
            AzureOpenAI(
                api_key=az_key,
                azure_endpoint=az_endpoint,
                api_version=az_version,
            ),
            os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        )
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "")
    if base_url and api_key:
        return OpenAI(base_url=base_url, api_key=api_key), model
    raise RuntimeError(
        "Set AZURE_OPENAI_* or OPENAI_BASE_URL/API_KEY env vars"
    )


SUMMARY_MAX_TOKENS = 16384
TRANSLATE_MAX_TOKENS = 16384


def _call_llm(system: str, user: str, *, max_completion_tokens: int = 8192) -> str:
    client, model = _build_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_completion_tokens,
    )
    msg = resp.choices[0].message
    finish = resp.choices[0].finish_reason
    if finish == "length":
        print(f"  WARNING: response truncated at {max_completion_tokens} completion tokens", flush=True)
    content = msg.content
    if finish == "length" and content is not None:
        content += "\n[TRUNCATED]"
    return content or ""


def load_source(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs: list[dict[str, Any]] = data[SEGMENTS_SOURCE]
    for s in segs:
        s.pop("target", None)
        s.pop("status", None)
    return segs


LETTER_SEPARATOR = "-----"


def group_letters(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split segments into letters at ----- boundaries."""
    letters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for s in segments:
        if s["source"] == LETTER_SEPARATOR:
            if current:
                letters.append(current)
                current = []
        else:
            current.append(s)
    if current:
        letters.append(current)
    return letters


def build_manuscript(segments: list[dict[str, Any]]) -> str:
    parts = []
    for s in segments:
        src = s["source"].strip()
        if src == LETTER_SEPARATOR:
            parts.append("\n-----\n")
        else:
            parts.append(src)
    return "\n\n".join(parts)


def generate_summary(full_text: str) -> str:
    system = (
        "You are a Czech literary analyst preparing context for a translator. "
        "Read the full book below, then write a thorough analytical summary in Czech. "
        "Include:\n"
        "- All main and secondary characters, their relationships, and how they relate to Virginia Woolf\n"
        "- Key themes, motifs, and recurring imagery\n"
        "- Stylistic notes: register (formal/intimate), tone shifts, notable phrasing\n"
        "- Historical and cultural references that appear\n"
        "- Any recurring names, places, terms that need consistent translation\n"
        "- Structure: how many letters, the arc, letter-writing conventions used\n\n"
        "Be as detailed as possible. This summary will be the ONLY context "
        "the translator has for each letter, so it must capture every nuance "
        "that could affect a faithful Russian translation."
    )
    content = _call_llm(system, full_text, max_completion_tokens=SUMMARY_MAX_TOKENS)
    return content.strip()


TRANSLATE_SYSTEM_TMPL = (
    "You are a literary translator from Czech to Russian.\n\n"
    "Context — Czech summary of the whole book:\n{summary}\n\n"
    "Below is letter {letter_num} of {total_letters} from this book. "
    "Translate each segment below from Czech to Russian. "
    "Preserve names, style, and literary quality.\n\n"
    "Respond with ONLY a JSON array. "
    "Each element must have exactly these keys:\n"
    '  "source" — original Czech text\n'
    '  "target" — Russian translation\n'
    '  "status" — always "draft"\n\n'
    "Example:\n"
    '[{{"source":"Praha, 1. srpna 2022","target":"Прага, 1 августа 2022","status":"draft"}}]\n\n'
    "Do not include any text before or after the JSON array."
)


def translate_letter(
    letter_num: int,
    total_letters: int,
    summary: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = _call_translate(letter_num, total_letters, summary, segments)
    result = _parse_translation(raw, letter_num, len(segments), segments)
    if result is not None:
        return result

    # Retry: split into halves
    print(f"  Retrying letter {letter_num} in 2 chunks...", end="", flush=True)
    half = len(segments) // 2
    parts = [segments[:half], segments[half:]]
    merged = []
    for part in parts:
        raw = _call_translate(letter_num, total_letters, summary, part)
        result = _parse_translation(raw, letter_num, len(part), part)
        if result is None:
            merged.extend(
                {"source": s["source"], "target": "", "status": "draft"}
                for s in part
            )
        else:
            merged.extend(result)
    return merged


def _call_translate(
    letter_num: int, total_letters: int, summary: str, segments: list[dict[str, Any]]
) -> str:
    system = TRANSLATE_SYSTEM_TMPL.format(
        summary=summary, letter_num=letter_num, total_letters=total_letters
    )
    numbered = "\n".join(
        f"[{i+1}] {s['source']}" for i, s in enumerate(segments)
    )
    user = "Translate these segments. Respond ONLY with a JSON array:\n\n" + numbered
    raw = _call_llm(system, user, max_completion_tokens=TRANSLATE_MAX_TOKENS)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        re.sub(r"\s*```$", "", raw)
    return raw


def _parse_translation(
    raw: str, letter_num: int, expected: int, segments: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: invalid JSON for letter {letter_num}", flush=True)
        return None

    if not isinstance(result, list):
        print(f"  WARNING: non-list response for letter {letter_num}", flush=True)
        return None

    if len(result) != expected:
        print(
            f"  WARNING: expected {expected} segments, got {len(result)} for letter {letter_num}",
            flush=True,
        )
        return None

    out = []
    for i, s in enumerate(segments):
        item = result[i] if i < len(result) else {}
        out.append({
            "source": s["source"],
            "target": item.get("target", "") if isinstance(item, dict) else "",
            "status": "draft",
        })
    return out


def save_letter(letter_num: int, segments: list[dict[str, Any]], letters_dir: Path) -> None:
    path = letters_dir / f"letter_{letter_num:02d}.json"
    path.write_text(
        json.dumps({"letter_index": letter_num, "segments": segments}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_joined(letters: list[list[dict[str, Any]]], output_path: Path) -> None:
    all_segments: list[dict[str, Any]] = []
    for i, letter in enumerate(letters):
        if i > 0:
            all_segments.append({
                "source": LETTER_SEPARATOR,
                "target": LETTER_SEPARATOR,
                "status": "approved",
            })
        all_segments.extend(letter)

    data = {
        "source_lang": "cz",
        "target_lang": "ru",
        "segments": all_segments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python3 translate_letters.py <input.json> <output.json> "
            "[--letters-dir <dir>] [--summary-file <path>]"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    letters_dir = Path(output_path.parent / "letters")
    summary_file: Path | None = None

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--letters-dir" and i + 1 < len(sys.argv):
            letters_dir = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--summary-file" and i + 1 < len(sys.argv):
            summary_file = Path(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    print("=== Loading source segments ===")
    segments = load_source(input_path)
    letters = group_letters(segments)
    print(f"  {len(segments)} segments, {len(letters)} letters")

    print("\n=== Generating book summary (Czech) ===")
    if summary_file and summary_file.exists():
        summary = summary_file.read_text(encoding="utf-8")
        print(f"  Using cached summary ({len(summary)} chars)")
    else:
        full_text = build_manuscript(segments)
        print(f"  Sending {len(full_text)} chars to GPT...")
        t0 = time.time()
        summary = generate_summary(full_text)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s ({len(summary)} chars)")
        if summary_file:
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            summary_file.write_text(summary + "\n", encoding="utf-8")
    print(f"  Summary:\n{summary}\n")

    print(f"=== Translating {len(letters)} letters ===")
    letters_dir.mkdir(parents=True, exist_ok=True)

    translated: list[list[dict[str, Any]]] = []
    for idx, letter_segs in enumerate(letters):
        letter_num = idx + 1
        print(f"  Letter {letter_num}/{len(letters)} ({len(letter_segs)} segments)...", end="", flush=True)
        t0 = time.time()
        result = translate_letter(letter_num, len(letters), summary, letter_segs)
        dt = time.time() - t0
        save_letter(letter_num, result, letters_dir)
        translated.append(result)
        print(f" {dt:.1f}s")

    print("\n=== Saving joined output ===")
    save_joined(translated, output_path)
    print(f"  Joined {sum(len(l) for l in translated)} segments → {output_path}")
    print(f"  Per-letter files → {letters_dir}/letter_XX.json")
    print("Done")


if __name__ == "__main__":
    main()
