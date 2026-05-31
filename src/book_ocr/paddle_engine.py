from __future__ import annotations

import inspect
from typing import Any

from .models import OcrLine, PageImage, PageOcrResult
from .text_assembly import sort_lines_for_reading


class PaddleOcrEngine:
    def __init__(
        self,
        lang: str = "cs",
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        show_log: bool = False,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_rec_score_thresh: float | None = None,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install project dependencies and a compatible "
                "CUDA-enabled PaddlePaddle build before running OCR."
            ) from exc

        signature = inspect.signature(PaddleOCR)
        parameters = signature.parameters

        if "use_textline_orientation" in parameters:
            kwargs = {
                "lang": lang,
                "device": "gpu:0" if use_gpu else "cpu",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": use_angle_cls,
            }
        else:
            kwargs = {
                "lang": lang,
                "use_gpu": use_gpu,
                "use_angle_cls": use_angle_cls,
                "show_log": show_log,
            }

        try:
            self._ocr = PaddleOCR(**kwargs)
        except TypeError:
            # Some minor releases accept fewer logging/device arguments than documented.
            kwargs.pop("show_log", None)
            self._ocr = PaddleOCR(**kwargs)
        self._use_predict = hasattr(self._ocr, "predict")

        self._predict_kwargs = {}
        if text_det_thresh is not None:
            self._predict_kwargs["text_det_thresh"] = text_det_thresh
        if text_det_box_thresh is not None:
            self._predict_kwargs["text_det_box_thresh"] = text_det_box_thresh
        if text_rec_score_thresh is not None:
            self._predict_kwargs["text_rec_score_thresh"] = text_rec_score_thresh

    def recognize_page(self, page: PageImage) -> PageOcrResult:
        if self._use_predict:
            raw_result = self._ocr.predict(str(page.path), **self._predict_kwargs)
        else:
            raw_result = self._ocr.ocr(str(page.path), cls=True)
        lines = sort_lines_for_reading(parse_paddle_result(raw_result))
        return PageOcrResult(page=page, lines=lines)


def parse_paddle_result(raw_result: Any) -> list[OcrLine]:
    page_level = _parse_page_level_result(raw_result)
    if page_level:
        return page_level

    records = _flatten_result(raw_result)
    lines: list[OcrLine] = []
    for record in records:
        page_level = _parse_page_level_result(record)
        if page_level:
            lines.extend(page_level)
            continue

        parsed = _parse_record(record)
        if parsed is not None:
            lines.append(parsed)
    return lines


def _parse_page_level_result(value: Any) -> list[OcrLine]:
    data = _as_mapping(value)
    if not data:
        return []

    texts = _first_present(data, "rec_texts", "text_recognition_texts")
    scores = _first_present(data, "rec_scores", "text_recognition_scores")
    boxes = _first_present(data, "rec_polys", "dt_polys", "rec_boxes")
    if scores is None:
        scores = []
    if boxes is None:
        boxes = []
    if texts is None:
        return []

    lines: list[OcrLine] = []
    for index, text in enumerate(_to_list(texts)):
        line_text = str(text).strip()
        if not line_text:
            continue
        confidence = _safe_float(_get_index(scores, index), default=0.0)
        box = _parse_box(_get_index(boxes, index))
        lines.append(OcrLine(text=line_text, confidence=confidence, box=box))
    return lines


def _flatten_result(raw_result: Any) -> list[Any]:
    if raw_result is None:
        return []

    if isinstance(raw_result, list) and len(raw_result) == 1 and isinstance(raw_result[0], list):
        return raw_result[0]

    if isinstance(raw_result, list):
        return raw_result

    return []


def _parse_record(record: Any) -> OcrLine | None:
    if not isinstance(record, (list, tuple)) or len(record) < 2:
        return None

    box = _parse_box(record[0])
    text, confidence = _parse_text_confidence(record[1])
    if not text:
        return None
    return OcrLine(text=text, confidence=confidence, box=box)


def _parse_box(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        else:
            return []

    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        left, top, right, bottom = [float(item) for item in value]
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    if not isinstance(value, (list, tuple)):
        return []

    box: list[list[float]] = []
    for point in value:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            box.append([float(point[0]), float(point[1])])
    return box


def _parse_text_confidence(value: Any) -> tuple[str, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first, second = value[0], value[1]
        if isinstance(first, str):
            return first.strip(), float(second)
        if isinstance(second, str):
            return second.strip(), float(first)

    if isinstance(value, dict):
        text = str(value.get("text") or value.get("rec_text") or "").strip()
        confidence = value.get("confidence", value.get("score", value.get("rec_score", 0.0)))
        return text, float(confidence)

    return "", 0.0


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    json_value = getattr(value, "json", None)
    if isinstance(json_value, dict):
        return json_value

    if callable(json_value):
        try:
            result = json_value()
        except TypeError:
            result = None
        if isinstance(result, dict):
            return result

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result

    return {}


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else [value]


def _get_index(value: Any, index: int) -> Any:
    items = _to_list(value)
    if index >= len(items):
        return None
    return items[index]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None
