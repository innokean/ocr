# Local Czech Book OCR Plan

## Method Review

The proposed pipeline is sound for scanned literary pages:

`scanned page image -> PaddleOCR -> raw text/layout -> VLM post-processing -> clean Czech manuscript`

Important caveats:

- PaddleOCR uses PaddlePaddle, not PyTorch. This is fine for the OCR stage, but the environment should install a CUDA-enabled PaddlePaddle build that matches the local CUDA driver. PyTorch/CUDA remains relevant for a local Qwen VLM cleanup stage.
- The first OCR pass should preserve evidence rather than over-clean it. For literary text, corrections are safer after keeping line boxes, confidence scores, page order, and raw line breaks.
- VLM cleanup should be constrained by the OCR output and page images. It should correct OCR errors, hyphenation, and paragraph flow, but should not modernize spelling, paraphrase, translate, or normalize punctuation.
- Gemini is useful for high quality post-processing but is not local. Qwen2.5-VL/Qwen3-VL style models are the local option and fit the target RTX 5090 hardware better.

## Target Output

- A single text body preserving Czech spelling and punctuation.
- Per-page raw OCR text for auditability.
- Per-page structured OCR records with bounding boxes and confidences.
- A manifest that allows reruns as more numbered JPEG files appear in `/dev/shm/milena/`.

## Repository Layout

- `src/book_ocr/`: OCR pipeline code.
- `PLAN.md`: this implementation plan.
- `README.md`: setup and usage.
- `output/`: default generated OCR output, ignored by git if a gitignore is added later.

## Stage 1: Scan Discovery

1. Read JPEG files from `/dev/shm/milena/` by default.
2. Accept `--input-dir` so the pipeline can be tested elsewhere.
3. Sort files naturally by the first number in the filename, then by full name.
4. Validate that at least one `.jpg` or `.jpeg` exists.
5. Write a manifest containing path, size, modified time, and SHA-256.

## Stage 2: Traditional OCR With PaddleOCR

1. Initialize PaddleOCR with Czech language code `cs`.
2. Prefer GPU execution by default.
3. Run detection, optional angle classification, and recognition for each page.
4. Normalize PaddleOCR result shapes across package versions.
5. Sort recognized text lines top-to-bottom and left-to-right using bounding boxes.
6. Save:
   - `pages/page_000001.txt`
   - `pages/page_000001.json`
   - `pages.jsonl`
   - `raw_manuscript.txt`

## Stage 3: Raw Layout Assembly

1. Keep line breaks inside each page.
2. Join pages with a blank line by default to form one continuous body.
3. Do not add visible page markers to the manuscript text unless explicitly requested.
4. Optionally dehyphenate and paragraphize later, but not in the initial raw OCR pass.

## Stage 4: Quality Checks

1. Summarize low-confidence lines.
2. Report pages with unusually few recognized lines.
3. Compare page count with numbered filenames and warn about gaps.
4. Keep enough metadata for manual inspection.

## Stage 5: VLM Cleanup

1. Feed page image plus raw OCR text to a VLM.
2. Use a strict prompt:
   - preserve Czech spelling and punctuation;
   - do not translate;
   - do not paraphrase;
   - repair OCR errors only when supported by the image/context;
   - merge broken lines and remove scan artifacts;
   - preserve paragraphing.
3. Run page-level cleanup first.
4. Run a second pass over adjacent page boundaries for hyphenation and paragraph continuity.
5. Record all prompts, model names, temperatures, and outputs.

## Stage 6: Local VLM Option

1. Use a PyTorch/CUDA stack with a Qwen vision-language model.
2. Use quantization only if VRAM pressure requires it; 32 GB VRAM should handle a useful local VLM tier.
3. Batch cautiously because high-resolution book pages can consume memory quickly.

## Stage 7: Gemini Option

1. Use Gemini only when cloud processing is acceptable.
2. Store API outputs separately from local outputs.
3. Keep the same strict no-translation/no-modernization prompt.

## Current Implementation Scope

This first implementation covers stages 1 through 3 and basic stage 4 reporting. VLM post-processing will be added after raw OCR output exists and can be inspected.
