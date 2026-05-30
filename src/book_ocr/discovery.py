from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import PageImage

JPEG_SUFFIXES = {".jpg", ".jpeg"}


def natural_page_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    number = int(match.group(0)) if match else 10**12
    return number, path.name.lower()


def discover_pages(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    pages = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES
    ]
    pages.sort(key=natural_page_key)
    if not pages:
        raise FileNotFoundError(f"No JPEG files found in: {input_dir}")
    return pages


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: list[Path]) -> list[PageImage]:
    pages: list[PageImage] = []
    for index, path in enumerate(paths, start=1):
        stat = path.stat()
        pages.append(
            PageImage(
                index=index,
                path=path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=sha256_file(path),
            )
        )
    return pages
