"""Module 6 - Result.

Packages the final output for the UI and produces the exportable formats
(TXT / JSON) offered by the result screen.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class OCRResult:
    image_name: str
    stored_name: str
    text: str
    confidence: float
    word_count: int
    region_count: int
    engine: str
    skew_angle: float = 0.0
    preprocess_steps: list[str] = field(default_factory=list)
    postprocess_steps: list[str] = field(default_factory=list)
    corrections: list[tuple[str, str]] = field(default_factory=list)
    removed_tokens: int = 0
    elapsed_ms: int = 0
    ocr_id: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    preview_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_txt(self) -> str:
        header = (
            f"Image      : {self.image_name}\n"
            f"Date       : {self.created_at}\n"
            f"Engine     : {self.engine}\n"
            f"Confidence : {self.confidence:.2f}%\n"
            f"Words      : {self.word_count}\n"
            + "-" * 48
            + "\n"
        )
        return header + self.text + "\n"

    @property
    def quality_label(self) -> str:
        if self.confidence >= 80:
            return "High"
        if self.confidence >= 60:
            return "Medium"
        return "Low"
