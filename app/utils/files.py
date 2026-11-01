"""여러 모듈에서 공통으로 사용하는 경로 / 파일 저장 기능이다."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

def ensure_directory(path: str | Path) -> Path:
    """폴더가 없으면 생성, 있으면 절대 경로를 반환한다."""
    directory = Path(path).expanduser().resolve()
    # expanduser:
    #   경로 시작 부분에 물결표(~)가 있을 경우,
    #   이를 현재 로그인한 사용자의 홈 디렉터리 절대 경로로 치환
    # resolve:
    #   경로에 포함된 상대 경로 기호를 계산하고,
    #   표준 절대 경로로 변환
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def resolve_within(base: str | Path, relative: str | Path) -> Path:
    """상대 경로를 해석하고 기준 폴더 밖을 가리키면 거부한다."""
    root = Path(base).expanduser().resolve()
    child = Path(relative)

    if child.is_absolute():
        # is_absolute: 절대 경로인지 여부를 판단하는 속성
        raise ValueError("상대 경로만 사용할 수 있습니다.")

    # 기존 심볼릭 링크도 해석한 후 경로를 확인한다.
    target = (root / child).resolve()

    if not target.is_relative_to(root):
        # is_relative_to: 다른 기준 경로의 하위(포함) 경로인지 확인
        raise ValueError("기준 폴더 밖의 경로에는 접근할 수 없습니다.")

    return target

def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """임시 파일을 완성한 뒤 교체하여 저장 중인 내용의 노출을 막는다."""
    if not isinstance(data, bytes):
        # isinstance:
        #   특정 객체가 어떤 클래스의 인스턴스(또는 하위 클래스의 인스턴스)가 맞는지 확인
        raise TypeError("data는 bytes여야 합니다.")

    target = Path(path).expanduser().resolve()
    ensure_directory(target.parent)

    temporary: Path | None = None

    try:
        # 같은 파일시스템에서 교체하도록 대상과 같은 폴더에 생성한다.
        with NamedTemporaryFile(
            mode="wb",
            dir=target.parent,        # 기본적으로 지정 안하면 OS의 temp에 들어가서 느려짐
            prefix=".echograph-",     # 파일 이름의 맨 앞에 붙는 시작 문구(접두사)
            delete=False,             # with 블록이 끝날 때 임시 파일이 자동으로 지워지는 것을 막음
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()            # 파이썬 버퍼의 데이터를 OS 버퍼로 밀어 넣는다.
            os.fsync(stream.fileno()) # 실제 물리 디스크 저장 장치에 완전히 기록한다.

        # 교체
        os.replace(temporary, target) # 임시 파일을 정식 파일 이름으로 덮어쓴다.
    finally:
        # 저장이나 교체에 실패해도 남은 임시 파일을 정리한다.
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return target

def atomic_write_text(path: str | Path, text: str) -> Path:
    """문자열을 UTF-8으로 교체하여 저장한다."""
    return atomic_write_bytes(path, text.encode("utf-8"))