from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .discovery import build_manifest, discover_pages
from .llm_client import LlmConfig
from .llm_stage import run_llm_pipeline
from .models import PageOcrResult
from .paddle_engine import PaddleOcrEngine
from .text_assembly import build_report, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR sequential Czech scanned book-page JPEGs into raw text."
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help="LLM API base URL (e.g. http://192.168.1.14:11434). Enables LLM post-processing.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3.6-27b",
        help="LLM model name (default: qwen3.6-27b).",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip the translation pass (output only corrected segments).",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=4,
        help="Max concurrent LLM requests (default: 4).",
    )
    parser.add_argument(
        "--from-ocr-dir",
        type=Path,
        default=None,
        help="Skip OCR, run LLM pipeline on existing OCR output directory.",
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
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Disable automatic text fixes (dehyphenation, spacing, common errors).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Drop OCR lines below this confidence (default: 0.5). Set 0 to keep all.",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=None,
        help="PaddleOCR detection threshold (default: model default).",
    )
    parser.add_argument(
        "--det-box-thresh",
        type=float,
        default=None,
        help="PaddleOCR detection box threshold (default: model default).",
    )
    parser.add_argument(
        "--rec-score-thresh",
        type=float,
        default=None,
        help="PaddleOCR recognition score threshold (default: model default).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    ocr_input = args.from_ocr_dir or args.input_dir

    if args.from_ocr_dir is None:
        # OCR stage
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
            text_det_thresh=args.det_thresh,
            text_det_box_thresh=args.det_box_thresh,
            text_rec_score_thresh=args.rec_score_thresh,
        )

        results: list[PageOcrResult] = []
        for page in pages:
            print(f"OCR {page.index}/{len(pages)}: {page.path}", flush=True)
            result = engine.recognize_page(page)
            if args.min_confidence > 0:
                result = result.filter_lines(min_conf=args.min_confidence)
            results.append(result)

        write_outputs(args.output_dir, results, apply_fixes=not args.no_fix)
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

    # LLM post-processing stage (if --llm-url provided or --from-ocr-dir)
    if args.llm_url or args.from_ocr_dir:
        llm_config = LlmConfig(
            url=args.llm_url or "http://192.168.1.14:11434",
            model=args.llm_model,
        )
        output_path = args.output_dir / "milena.json"
        run_llm_pipeline(
            ocr_input,
            output_path,
            llm_config,
            skip_translate=args.no_translate,
            max_workers=args.llm_workers,
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
