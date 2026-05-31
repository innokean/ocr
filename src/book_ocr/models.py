from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageImage:
    index: int
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: list[list[float]]

    @property
    def top(self) -> float:
        return min(point[1] for point in self.box) if self.box else 0.0

    @property
    def left(self) -> float:
        return min(point[0] for point in self.box) if self.box else 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PageOcrResult:
    page: PageImage
    lines: list[OcrLine]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines).strip()

    @property
    def mean_confidence(self) -> float | None:
        if not self.lines:
            return None
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def filter_lines(self, min_conf: float = 0.5) -> PageOcrResult:
        return replace(self, lines=[line for line in self.lines if line.confidence >= min_conf])

    def to_json(self) -> dict[str, Any]:
        return {
            "page": self.page.to_json(),
            "line_count": len(self.lines),
            "mean_confidence": self.mean_confidence,
            "text": self.text,
            "lines": [line.to_json() for line in self.lines],
        }
