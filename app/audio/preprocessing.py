"""오디오를 분석용 모노 PCM으로 읽고, 겹치는 시간 구간으로 나눈다."""

import math
import shutil
import subprocess
from _collections_abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mpmath import isint
from numpy.typing import NDArray

class AudioPreprocessingError(RuntimeError):
    """오디오 디코딩 실패 또는 처리 시간 초과를 나타낸다."""

@dataclass(frozen=True, eq=False)
class AudioData:
    """샘플 좌표와 길이는 변환된 분석용 PCM을 기준으로 한다."""
    source: Path
    sample_rate: int
    samples: NDArray[np.float32] = field(repr=False)

    @property
    def duration_seconds(self) -> float:
        """디코딩 된 오디오의 길이를 초 단위로 변환"""
        return len(self.samples) / self.sample_rate

@dataclass(frozen=True, eq=False)
class AudioSegment:
    """구간 번호는 0부터 시작하고, 끝 샘플은 포함하지 않음"""

    index: int
    sample_rate: int
    start_sample: int
    end_sample: int
    samples: NDArray[np.float32] = field(repr=False)

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return (
            self.end_sample - self.start_sample
        ) / self.sample_rate

def _positive_seconds(value: int | float, name: str) -> float:
    """시간 인자가 유한한 양수인지 확인"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}은 숫자여야 합니다.")

    try:
        number = float(value)
    except OverflowError as e:
        raise ValueError(f"{name}이 너무 큽니다.") from e

    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{name}은 0보다 큰 유한한 숫자여야 합니다."
        )

    return number

def _validate_rate(sample_rate: int) -> None:
    """샘플레이트가 양의 정수인지 확인한다."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("샘플레이트는 정수여야 합니다.")

    if sample_rate <= 0:
        raise ValueError("샘플레이트는 0보다 커야 합니다.")

def load_audio(
        path: str | Path,
        *,
        sample_rate: int = 48_000,
        timeout_seconds: float = 120.0
) -> AudioData:
    """FFmpeg 모노 float32 PCM을 읽는다."""
    _validate_rate(sample_rate)
    timeout = _positive_seconds(
        timeout_seconds,
        "처리 제한 시간"
    )

    source = Path(path).expanduser().resolve(strict=True)

    if not source.is_file():
        raise ValueError("오리오 파일 경로가 필요합니다.")

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise AudioPreprocessingError(
            "FFmpeg 실행 파일을 찾을 수 없습니다."
        )

    # 첫 오디오 스트림을 모노로 혼합하고 샘플레이트를 변환
    # 임의의 음량 정규화나 무음 제거는 적용하지 않음.
    command = [
        ffmpeg,
        "-nostdin",
        "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-i", str(source),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
        "-f", "f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False
        )
    except subprocess.TimeoutExpired as e:
        raise AudioPreprocessingError(
            "오디오 변환 제한 시간을 초과했습니다."
        ) from e

    if result.returncode != 0:
        detail = result.stderr.decode(
            "utf-8",
            errors="replace"
        ).strip()

        raise AudioPreprocessingError(
            f"오디오를 읽지 못했습니다 : {detail[-2000:]}"
        )

    # FFmpeg가 출력한 little-endian float32 데이터를 읽는다.
    # bytes를 공유하므로 배열은 읽기 전용
    samples = np.frombuffer(result.stdout, dtype="<f4")

    if samples.size == 0:
        raise AudioPreprocessingError(
            "오디오에 NaN 또는 무한대 값이 있습니다."
        )

    return AudioData(
        source=source,
        sample_rate=sample_rate,
        samples=samples
    )

def split_audio(
        audio: AudioData,
        *,
        window_seconds: float = 10.0,
        hop_seconds: float = 5.0,
) -> Iterator[AudioSegment]:
    """겹치는 구간을 반환, 짮은 마지막 구간은 패딩하지 않음"""
    _validate_rate(audio.sample_rate)

    if audio.samples.ndim != 1 or audio.samples.size == 0:
        raise ValueError(
            "비어 있지 않은 모노 오디오 배열이 필요합니다."
        )

    if not np.isfinite(audio.samples).all():
        raise ValueError(
            "오디오 배열에 NaN 또는 무한대 값이 있습니다."
        )

    window = (
        _positive_seconds(window_seconds, "구간 길이")
        * audio.sample_rate
    )

    hop = (
        _positive_seconds(hop_seconds, "이동 간격")
        * audio.sample_rate
    )

    if not math.isfinite(window) or not math.isfinite(hop):
        raise ValueError("구간 설정이 너무 큽니다.")

    # 정수 샘플 좌표를 사용하여 반복적인 시간 덧셈 오차를 피한다.
    window_samples = round(window)
    hop_samples = round(hop)

    if not 0 < hop_samples <= window_samples:
        raise ValueError(
            "샘플 기준으로 0 < 이동 간격 <= 구간 길이여야 합니다."
        )

    total = len(audio.samples)

    for idx, start in enumerate(range(0, total, hop_samples)):
        end = min(start + window_samples, total)

        yield AudioSegment(
            index=idx,
            sample_rate=audio.sample_rate,
            start_sample=start,
            end_sample=end,
            samples=audio.samples[start:end],
        )

        if end == total:
            break

