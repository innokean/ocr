from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .discovery import build_manifest, discover_pages
from .models import PageOcrResult
from .paddle_engine import PaddleOcrEngine
from .text_assembly import build_report, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR sequential Czech scanned book-page JPEGs into raw text."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/dev/shm/milena"),
        help="Directory containing numbered JPEG scans.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for OCR output.",
    )
    parser.add_argument("--lang", default="cs", help="PaddleOCR language code.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run PaddleOCR on CPU instead of GPU.",
    )
    parser.add_argument(
        "--no-angle-cls",
        action="store_true",
        help="Disable PaddleOCR angle classification.",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.85,
        help="Confidence threshold for report warnings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover pages and write manifest; do not run OCR.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    image_paths = discover_pages(args.input_dir)
    pages = build_manifest(image_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps([page.to_json() for page in pages], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"Discovered {len(pages)} JPEG page(s). Manifest written to {args.output_dir}.")
        return 0

    engine = PaddleOcrEngine(
        lang=args.lang,
        use_gpu=not args.cpu,
        use_angle_cls=not args.no_angle_cls,
    )

    results: list[PageOcrResult] = []
    for page in pages:
        print(f"OCR {page.index}/{len(pages)}: {page.path}", flush=True)
        results.append(engine.recognize_page(page))

    write_outputs(args.output_dir, results)
    report = build_report(results, low_confidence=args.low_confidence)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "OCR complete: "
        f"{report['page_count']} page(s), {report['total_line_count']} line(s). "
        f"Raw text: {args.output_dir / 'raw_manuscript.txt'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"book-ocr: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
