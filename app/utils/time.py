"""저장용 UTC 시각과 화면에 표시할 경과 시간을 처리한다."""

import math
from datetime import UTC, datetime

def utc_now() -> datetime:
    """시간대 정보가 포함된 현재 UTC시각을 반환한다."""
    return datetime.now(UTC)

def to_utc_iso(value: datetime) -> str:
    """시각을 밀리초 단위 UTC ISO 문자열로 변환한다."""
    if value.tzinfo is None or value.utcoffset()is None:
        raise ValueError("시간대 정보가 있는 datetime이 필요합니다.")

    return (
        value.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

def format_timestamp(seconds: int | float) -> str:
    """초를 HH:MM:SS.mmm으로 변환한다. (Python Round 규칙 적용)"""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError("시간은 초 단위의 숫자여야 합니다.")

    try:
        milliseconds = seconds * 1000
        valid = seconds >= 0 and math.isfinite(milliseconds)
    except OverflowError:
        valid = False

    if not valid:
        raise ValueError("시간이 유효한 범위를 벗어났습니다.")

    # 밀리초로 반올림하여 초 / 분 경계에서도 올바르게 표시한다.
    total = round(milliseconds)

    hours, remaining = divmod(total, 3_600_000)
    minutes, remaining = divmod(remaining, 60_000)
    whole_seconds, fraction = divmod(remaining, 1_000)

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{whole_seconds:02d}.{fraction:03d}"
    )
