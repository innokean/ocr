from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .llm_client import BATCH_SIZE, LlmConfig, fix_diacritics, translate_batch
from .segmentation import LETTER_SEPARATOR, segment_text


def _is_letter_header(text: str) -> bool:
    """Detect if text looks like a letter header: City, DD. Month YYYY"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    pattern = re.compile(
        r"^[A-ZÁ-Ž][a-záčďéěíňóřšťúůýž]+,\s+\d{1,2}\.\s+"
        r"[a-záčďéěíňóřšťúůýž]+\s+\d{4}$"
    )
    return bool(pattern.match(lines[0]))


def insert_letter_separators(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect letter headers and insert ----- separators between letters."""
    result: list[dict[str, Any]] = []
    letter_count = 0
    for seg in segments:
        src = seg["source"].strip()
        if _is_letter_header(src):
            letter_count += 1
            if letter_count > 1:
                result.append({
                    "source": LETTER_SEPARATOR,
                    "target": LETTER_SEPARATOR,
                    "status": "approved",
                })
        result.append(seg)
    return result


def correct_pages(
    input_dir: Path,
    output_dir: Path,
    llm_config: LlmConfig,
    *,
    max_pages: int | None = None,
    max_workers: int = 4,
) -> list[str]:
    """Correct all pages with LLM, saving intermediate results. Returns corrected texts."""
    pages_dir = input_dir / "pages"
    page_txts = sorted(pages_dir.glob("page_*.txt"))
    if max_pages:
        page_txts = page_txts[:max_pages]

    corrected_dir = output_dir / "corrected_pages"
    corrected_dir.mkdir(parents=True, exist_ok=True)

    corrected_texts = [""] * len(page_txts)

    def _process(i: int, pf: Path) -> tuple[int, str, str]:
        page_text = pf.read_text(encoding="utf-8").strip()
        corrected_path = corrected_dir / pf.name

        if corrected_path.exists():
            corrected = corrected_path.read_text(encoding="utf-8")
        else:
            corrected = fix_diacritics(llm_config, page_text)
            corrected_path.write_text(corrected + "\n", encoding="utf-8")

        return i, page_text, corrected

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_process, i, pf): i for i, pf in enumerate(page_txts)}
        for fut in as_completed(futs):
            i, _, corrected = fut.result()
            corrected_texts[i] = corrected
            print(f"  Corrected page {i+1}/{len(page_txts)}", flush=True)

    return corrected_texts


def translate_segments(
    segments: list[dict[str, Any]],
    llm_config: LlmConfig,
    *,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Translate all non-empty, non-separator segments in batches."""
    to_translate = [
        (i, s) for i, s in enumerate(segments)
        if s["source"].strip() and s["source"].strip() != LETTER_SEPARATOR
    ]
    total = len(to_translate)

    batches = [
        to_translate[j:j + BATCH_SIZE]
        for j in range(0, total, BATCH_SIZE)
    ]

    def _translate_batch(batch: list[tuple[int, Any]]) -> None:
        texts = [s["source"] for _, s in batch]
        try:
            translations = translate_batch(llm_config, texts)
        except Exception as exc:
            print(f"  Batch failed ({exc}), retrying individually", flush=True)
            from .llm_client import translate_text
            translations = [translate_text(llm_config, t) for t in texts]
        for (i, s), t in zip(batch, translations):
            segments[i]["target"] = t
            segments[i]["status"] = "draft"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_translate_batch, b): b for b in batches}
        for fut in as_completed(futs):
            _ = fut.result()
            batch = futs[fut]
            first_idx = batch[0][0]
            print(f"  Translated batch starting at segment {first_idx+1}/{total}", flush=True)

    return segments


def run_llm_pipeline(
    input_dir: Path,
    output_path: Path,
    llm_config: LlmConfig | None = None,
    *,
    skip_translate: bool = False,
    max_pages: int | None = None,
    max_workers: int = 4,
) -> None:
    """Full pipeline: correct diacritics → segment → translate → output JSON."""
    if llm_config is None:
        llm_config = LlmConfig()

    log = []

    print("=== Stage 1: LLM diacritic correction per page ===")
    t0 = time.time()
    corrected_texts = correct_pages(
        input_dir, input_dir, llm_config,
        max_pages=max_pages, max_workers=max_workers,
    )
    dt = time.time() - t0
    log.append({"stage": "diacritic_correction", "pages": len(corrected_texts),
                 "seconds": round(dt, 1)})
    print(f"  Done: {dt:.1f}s\n")

    print("=== Stage 2: Merging and segmenting ===")
    t0 = time.time()
    manuscript = "\n\n".join(corrected_texts)
    # Normalize multiple blank lines
    manuscript = re.sub(r"\n{3,}", "\n\n", manuscript.strip())
    segments = segment_text(manuscript)
    segments = insert_letter_separators(segments)
    dt = time.time() - t0
    log.append({"stage": "segmentation", "segments": len(segments),
                 "seconds": round(dt, 1)})
    print(f"  {len(segments)} segments in {dt:.1f}s\n")

    if skip_translate:
        for s in segments:
            if s["source"].strip() and s["source"].strip() != LETTER_SEPARATOR:
                s["target"] = ""
                s["status"] = ""
        _write_output(segments, log, output_path)
        return

    print("=== Stage 3: Translation to Russian ===")
    t0 = time.time()
    segments = translate_segments(segments, llm_config, max_workers=max_workers)
    dt = time.time() - t0
    log.append({"stage": "translation", "segments": len(segments),
                 "seconds": round(dt, 1)})
    print(f"  Done: {dt:.1f}s\n")

    _write_output(segments, log, output_path)
    total = sum(e["seconds"] for e in log)
    print(f"=== Complete: {total:.1f}s total ===")
    print(f"Output: {output_path}")


def _write_output(
    segments: list[dict[str, Any]],
    log: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "title": "",
        "source_lang": "cz",
        "target_lang": "ru",
        "translator": "",
        "pipeline": log,
        "segments": segments,
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
