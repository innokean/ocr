# book-ocr — Agent guide

## Project
Local OCR pipeline for scanned Czech book pages. Single package `book-ocr` under `src/` layout.

## Commands
- Run OCR: `uv run book-ocr --input-dir /dev/shm/milena --output-dir output`
- Dry-run (discover only): `uv run book-ocr --input-dir /dev/shm/milena --dry-run`
- Run tests: `uv run pytest` (no config, one test file `tests/test_pipeline_helpers.py`)
- Add deps: `uv add <package>`
- Sync: Repo uses `uv` — do not use `pip install` or `poetry`.

## CLI entrypoint
`book_ocr.cli:main` → `src/book_ocr/cli.py`.
Args: `--input-dir` (default `/dev/shm/milena`), `--output-dir` (default `output`), `--lang` (default `cs`), `--cpu`, `--no-angle-cls`, `--low-confidence` (default `0.85`), `--dry-run`,
`--no-fix` (disable auto dehyphenation/spacing fixes), `--min-confidence` (default `0.5`, set 0 to keep all),
`--det-thresh`, `--det-box-thresh`, `--rec-score-thresh` (PaddleOCR predict kwargs).

LLM pipeline flags:
- `--llm-url <URL>`: enable LLM post-processing (default: none). E.g. `http://192.168.1.14:11434`
- `--llm-model <name>`: model name (default `qwen3.6-27b`)
- `--no-translate`: skip Russian translation pass (output only corrected segments)
- `--llm-workers <N>`: concurrent LLM requests (default 4)
- `--from-ocr-dir <DIR>`: skip OCR, run LLM pipeline on existing OCR output directory

## Setup quirk
PaddleOCR requires a CUDA-enabled PaddlePaddle wheel matching the local CUDA driver. PaddlePaddle is **not** listed in `pyproject.toml` dependencies. Install it manually first (official PaddlePaddle guide), then `uv sync`.

## Output layout
- `output/raw_manuscript.txt` — fixed text (pages joined by blank line; dehyphenated, spacing fixed)
- `output/pages/page_000001.txt` / `page_000001.json` — per-page `.txt` has fixes, `.json` preserves raw OCR data
- `output/pages.jsonl` — all pages as JSONL (raw)
- `output/manifest.json` — input file manifest with SHA-256
- `output/report.json` — confidence/summary stats
- `output/milena.json` — LLM post-processed output (corrected segments + optional translations)
- `output/corrected_pages/` — per-page LLM-corrected text (cached, reuse on re-run)

## Post-processing fixes
Automatic (disable with `--no-fix`):
- **Dehyphenation**: `(\w+)-\n(\w+)` → `\1\2` joins book-hyphenated words
- **Diacritic fix**: `ù`/`Ù` → `ů`/`Ů` (PaddleOCR mixes these)
- **Spacing fix**: `Súctou` → `S úctou` (known book-specific error)
- **Confidence filter**: drops lines below `--min-confidence` (default 0.5)

## Known PaddleOCR limitations
- `ť` at word endings is consistently missed: "sít" → should be "síť" (context-dependent; LLM/VLM needed)
- `ď` rendered as `'`: "Ted'" → should be "Teď" (model lacks ď in output; `ù`/`ů` confusion also common)
- Top-of-page noise lines (e.g. "gialviha") are common — the min-confidence filter handles most
- These are recognition model limits; no preprocessing or threshold tuning fixes them

## Architecture notes
- Scan discovery sorts JPEGs by first numeric sequence in filename (`natural_page_key`).
- PaddleOCR kwargs differ between v2 (`use_gpu`/`use_angle_cls`/`show_log`) and v3 (`device`/`use_textline_orientation`). Engine inspects `PaddleOCR.__init__` signature to pick the right set.
- Both PaddleOCR v2 list-of-lists and v3 `OCRResult` page-dict result shapes are handled.
- Recognition lines are sorted top-to-bottom then left-to-right using `(round(top/12), left)`.
- Raw guard: `_first_present()` tries multiple key names (`rec_texts`, `text_recognition_texts`, etc.) across PaddleOCR versions.
- No linter, formatter, or typechecker config exists.

## LLM post-processing (Qwen 3.6 27B)
Stage runs after v5 OCR + mechanical fixes when `--llm-url` is provided:
1. **Diacritic correction**: per-page LLM call, fixes `ť`/`ď`/`ů`/`ř`/`š`/`č`/`ž`/`ě`. Caches to `corrected_pages/`.
2. **Segmentation**: splits into paragraph/sentence segments, detects letter headers to insert `-----` separators.
3. **Translation**: batch-translates segments to Russian (skip with `--no-translate`). Outputs `milena.json`.

LLM uses `llama.cpp` server at `--llm-url`, endpoint `/v1/chat/completions`. Default model `qwen3.6-27b`.
Pages are processed in parallel with `--llm-workers` (default 4).
Translation batches 10 segments per LLM call to reduce overhead (BATCH_SIZE in `llm_client.py`).
Server started with `-np 1` — requests are serial; use `-np N` for parallel.

## Azure GPT-5.3 translation (alternate)
Standalone script `translate_letters.py` translates letter-by-letter using Azure OpenAI (GPT-5.3-chat).
Reads `output/milena.json` segments, groups by letter, generates a Czech summary for context, then translates each letter in one call.

Usage:
```
uv run python3 translate_letters.py output/milena.json output/milena_new.json [--letters-dir output/letters] [--summary-file output/summary.txt]
```
Output: `letter_01.json`–`letter_19.json` (per-letter), `milena_new.json` (joined).
Requires `AZURE_OPENAI_API_KEY`/`ENDPOINT`/`VERSION`/`DEPLOYMENT` or `OPENAI_BASE_URL`/`API_KEY`/`MODEL` env vars.
