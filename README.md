# Local Czech Book OCR

Local OCR pipeline for sequential JPEG scans of Czech literary book pages.

## Setup

Install a CUDA-enabled PaddlePaddle build appropriate for the machine, then install the project dependencies. PaddleOCR currently uses PaddlePaddle rather than PyTorch for the traditional OCR stage.

Example:

```bash
uv sync
```

If the selected PaddlePaddle wheel is not available through the default index for the installed CUDA version, install PaddlePaddle manually using the official PaddlePaddle instructions, then install this project.

## Usage

Dry-run page discovery:

```bash
uv run book-ocr --input-dir /dev/shm/milena --dry-run
```

Run OCR:

```bash
uv run book-ocr --input-dir /dev/shm/milena --output-dir output
```

Main output:

- `output/raw_manuscript.txt`: one continuous raw OCR text body.
- `output/pages/page_000001.txt`: raw text for each page.
- `output/pages/page_000001.json`: OCR records for each page.
- `output/pages.jsonl`: all page OCR records.
- `output/manifest.json`: input file manifest.
- `output/report.json`: basic quality statistics.

## Notes

- Czech language is configured with PaddleOCR language code `cs`.
- The first pass intentionally preserves raw line breaks and punctuation. Cleanup, dehyphenation, and paragraph merging belong in the later VLM stage.
- More JPEG files can be added later; rerun the same command to rebuild outputs from the current directory contents.
