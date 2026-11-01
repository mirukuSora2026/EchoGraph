"""설정과 메타데이터를 표준 JSON으로 읽고 저장한다."""

import json
from pathlib import Path
from typing import Any

from .files import atomic_write_text

def _reject_constant(value: str) -> None:
    """표준 JSON에 없는 NaN과 Infinity를 거부한다."""
    raise ValueError(f"JSON에서 사용할 수 없는 값: {value}")

def read_json(path: str | Path) -> Any:
    """JSON을 읽는다. 파일 누락과 형식 오류는 호출한 쪽에 전달한다."""
    text = Path(path).expanduser().read_text(encoding="utf-8")

    return json.loads(text, parse_constant=_reject_constant)

def write_json(
        path: str | Path,
        data: Any,
        *,
        pretty: bool = True,
) -> Path:
    """한글을 유지하고 완성된 JSON으로 파일을 교체한다."""
    # 변환을 먼저 수행해, 잘못된 데이터로 기존 파일 손상을 막는다.
    text = json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
    )

    return atomic_write_text(path, text + "\n")
