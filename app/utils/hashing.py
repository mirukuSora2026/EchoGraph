"""큰 파일도 메모리에 전부 올리지 않고 해시를 계산한다."""

import hashlib
from pathlib import Path

def sha256_file(
        path: str | Path,
        chunk_size: int = 1024 * 1024,
) -> str:
    """파일의 SHA-256을 64자리 문자열로 변환한다."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("읽기 크기는 정수여야 합니다.")

    if chunk_size <= 0:
        raise ValueError("읽기 크기는 0보다 커야합니다.")

    digest = hashlib.sha256()

    # 기본적으로 1MiB씩 읽어 큰 파일의 메모리 사용량을 줄인다.
    with Path(path).expanduser().open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()
