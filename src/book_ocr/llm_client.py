from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class LlmConfig:
    url: str = "http://192.168.1.14:11434"
    model: str = "qwen3.6-27b"
    temperature: float = 0.0
    max_tokens: int = 16384


def _call_llm(
    config: LlmConfig,
    system: str,
    user: str,
) -> str:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    req = urllib.request.Request(
        f"{config.url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        if not content and msg.get("reasoning_content"):
            return ""
        return content
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc


DIACRITIC_SYSTEM = (
    "Fix Czech OCR diacritics and join hyphenated words. "
    "Restore missing carons: ť/ď/ů/ř/š/č/ž/ě. "
    "Fix ù→ů, d'→ď, t'→ť. "
    "Join words broken by hyphens across line breaks (e.g. romá-nu → románu). "
    "Merge lines into flowing paragraphs. Separate paragraphs by a blank line. "
    "Keep all wording, grammar, and punctuation. "
    "Do NOT paraphrase, translate, or modernize. "
    "Output only the corrected text."
)


def fix_diacritics(config: LlmConfig, text: str) -> str:
    return _call_llm(config, DIACRITIC_SYSTEM, text)


TRANSLATE_SYSTEM = (
    "Translate the Czech text below to Russian. "
    "Preserve the exact meaning, style, and punctuation. "
    "Do not paraphrase or add anything. "
    "Output only the translation, no explanations."
)


def translate_text(config: LlmConfig, text: str) -> str:
    return _call_llm(config, TRANSLATE_SYSTEM, text)


BATCH_TRANSLATE_SYSTEM = (
    "You will receive numbered Czech text segments. "
    "Translate each one to Russian preserving the exact meaning and style.\n"
    "Output each translation on its own line, prefixed with the same number in brackets.\n"
    "Example:\n"
    "[1] Praha, 1. srpna 2022\n"
    "→\n"
    "[1] Прага, 1 августа 2022\n"
    "Output only the numbered translations, one per line."
)


BATCH_SIZE = 10


def translate_batch(config: LlmConfig, texts: list[str]) -> list[str]:
    """Translate multiple short texts in a single LLM call."""
    numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    raw = _call_llm(config, BATCH_TRANSLATE_SYSTEM, numbered)
    result: list[str] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"\[\d+\]\s*(.*)", line)
        if m:
            result.append(m.group(1))
    if len(result) != len(texts):
        msg = f"Expected {len(texts)} translations, got {len(result)}. Falling back to individual calls."
        print(f"  WARNING: {msg}", flush=True)
        return [translate_text(config, t) for t in texts]
    return result
