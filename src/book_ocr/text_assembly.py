from __future__ import annotations

import json
from pathlib import Path

from .models import PageOcrResult


def sort_lines_for_reading(lines):
    return sorted(lines, key=lambda line: (round(line.top / 12.0), line.left))


def assemble_manuscript(results: list[PageOcrResult]) -> str:
    page_texts = [result.text for result in results if result.text]
    return "\n\n".join(page_texts).strip() + ("\n" if page_texts else "")


def write_outputs(output_dir: Path, results: list[PageOcrResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "pages.jsonl").open("w", encoding="utf-8") as jsonl:
        for result in results:
            page_id = f"page_{result.page.index:06d}"
            (pages_dir / f"{page_id}.txt").write_text(result.text + "\n", encoding="utf-8")
            page_json = result.to_json()
            (pages_dir / f"{page_id}.json").write_text(
                json.dumps(page_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            jsonl.write(json.dumps(page_json, ensure_ascii=False) + "\n")

    (output_dir / "raw_manuscript.txt").write_text(
        assemble_manuscript(results),
        encoding="utf-8",
    )


def build_report(results: list[PageOcrResult], low_confidence: float) -> dict:
    low_confidence_lines = []
    sparse_pages = []

    for result in results:
        if len(result.lines) < 5:
            sparse_pages.append(
                {
                    "page_index": result.page.index,
                    "path": str(result.page.path),
                    "line_count": len(result.lines),
                }
            )

        for line_number, line in enumerate(result.lines, start=1):
            if line.confidence < low_confidence:
                low_confidence_lines.append(
                    {
                        "page_index": result.page.index,
                        "line_number": line_number,
                        "confidence": line.confidence,
                        "text": line.text,
                    }
                )

    mean_confidences = [
        result.mean_confidence for result in results if result.mean_confidence is not None
    ]
    return {
        "page_count": len(results),
        "total_line_count": sum(len(result.lines) for result in results),
        "mean_confidence": (
            sum(mean_confidences) / len(mean_confidences) if mean_confidences else None
        ),
        "low_confidence_threshold": low_confidence,
        "low_confidence_line_count": len(low_confidence_lines),
        "low_confidence_lines": low_confidence_lines[:200],
        "sparse_pages": sparse_pages,
    }
