from pathlib import Path

from book_ocr.discovery import natural_page_key
from book_ocr.models import OcrLine
from book_ocr.paddle_engine import parse_paddle_result
from book_ocr.text_assembly import sort_lines_for_reading


def test_natural_page_key_sorts_by_first_number():
    names = [Path("10.jpg"), Path("2.jpg"), Path("001.jpg")]
    assert sorted(names, key=natural_page_key) == [
        Path("001.jpg"),
        Path("2.jpg"),
        Path("10.jpg"),
    ]


def test_parse_paddle_v2_shape():
    raw = [
        [
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                ("Příliš žluťoučký kůň.", 0.98),
            ]
        ]
    ]
    lines = parse_paddle_result(raw)
    assert len(lines) == 1
    assert lines[0].text == "Příliš žluťoučký kůň."
    assert lines[0].confidence == 0.98


def test_parse_paddle_v3_page_shape():
    raw = [
        {
            "rec_texts": ["První řádek.", "Druhý řádek."],
            "rec_scores": [0.91, 0.92],
            "rec_polys": [
                [[0, 0], [100, 0], [100, 10], [0, 10]],
                [[0, 20], [100, 20], [100, 30], [0, 30]],
            ],
        }
    ]
    lines = parse_paddle_result(raw)
    assert [line.text for line in lines] == ["První řádek.", "Druhý řádek."]
    assert [line.confidence for line in lines] == [0.91, 0.92]


def test_sort_lines_for_reading_uses_top_then_left():
    right = OcrLine("right", 0.9, [[50, 0], [60, 0], [60, 10], [50, 10]])
    left = OcrLine("left", 0.9, [[0, 0], [10, 0], [10, 10], [0, 10]])
    lower = OcrLine("lower", 0.9, [[0, 50], [10, 50], [10, 60], [0, 60]])
    assert [line.text for line in sort_lines_for_reading([lower, right, left])] == [
        "left",
        "right",
        "lower",
    ]
