"""API와 분석 워커에서 공통으로 쓰는 로그 설정이다."""

import logging
import time
from threading import RLock

_LOCK = RLock()
_LOGGER_NAME = "echograph"
_HANDLER_MARKER = "_echograph_console_handler"

def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """프로세스 시작 시 호출하며, 자체 출력 핸들러의 중복을 방지한다."""
    with _LOCK:
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(
            level.upper() if isinstance(level, str) else level
        )

        # 상위 로거를 통해 같은 메시지 재출력을 방지한다.
        logger.propagate = False

        # 이 모듈이 만든 핸들러가 있으면 재사용한다.
        handler = next(
            (
                item
                for item in logger.handlers
                if getattr(item, _HANDLER_MARKER, False)
            ),
            None,
        )

        if handler is None:
            handler = logging.StreamHandler()
            setattr(handler, _HANDLER_MARKER, True)
            logger.addHandler(handler)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

        # 로그 시각도 UTC로 통일
        formatter.converter = time.gmtime

        handler.setFormatter(formatter)
        handler.setLevel(logger.level)

        return logger

def get_logger(name: str | None = None) -> logging.Logger:
    """모듈별 로거를 가져온다."""
    full_name = f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME
    return logging.getLogger(full_name)
