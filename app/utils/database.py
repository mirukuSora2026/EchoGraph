"""SQLite 요청을 순차적으로 처리하는 API이다."""

import asyncio
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, TypeVar
from .files import ensure_directory
from .logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


@dataclass(frozen=True)
class Statement:
    """실행할 SQL과 바인딩 할 값을 보관한다."""
    sql: str
    params: tuple[object, ...] = ()


@dataclass
class QueryResult:
    """커서 대신 다른 스레드에서도 사용할 일반 데이터를 반환한다."""

    rows: list[dict[str, object]]
    rowcount: int  # SELECT에선 -1


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

        # 순차처리용 작업 스레드
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="echograph-db",
        )

        # 접수와 종료가 동시에 발생하는 상황 보호
        self._lock = Lock()
        self._closing = False
        self._close_future: Future[None] | None = None

        # 연결의 생성 / 사용 / 종료는 모두 한 스레드에서 수행
        self._connection: sqlite3.Connection | None = None

    async def start(self) -> None:
        """DB 연결 준비"""
        try:
            await self._wait(self._submit(self._connect))
        except BaseException:
            await self.close()
            raise

    def _connect(self) -> None:
        """작업 스레드 안에서 SQLite 연결 생성"""
        if self._connection is not None:
            return

        ensure_directory(self.path.parent)

        # 자동 트랜젝션 대신 BEGIN, COMMIT을 직접 관리
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )

        try:
            # 조회 결과를 열 이름으로 접근할 수 있도록 연결 속성에 설정한다.
            connection.row_factory = sqlite3.Row

            for sql in (
                    "PRAGMA foreign_keys = ON",
                    "PRAGMA journal_mode = WAL",
            ):
                cursor = connection.execute(sql)
                try:
                    cursor.fetchall()
                finally:
                    cursor.close()

        except BaseException:
            connection.close()
            raise

        self._connection = connection

    def _submit(self, operation: Callable[[], T]) -> Future[T]:
        """종료 여부를 확인하고 작업 큐에 추가"""
        with self._lock:
            if self._closing:
                raise RuntimeError("종료 중인 데이터베이스에는 작업을 추가할 수 없습니다.")

            return self._executor.submit(operation)

    @staticmethod
    async def _wait(future: Future[T]) -> T:
        """이벤트 루프를 막지 않고 작업 결과를 기다린다."""
        wrapped = asyncio.wrap_future(future)

        # 요청이 취소된 뒤 작업이 실패하더라도 예외를 회수한다.
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )

        # 요청 취소는 결과 대기만 중단한다.
        # 이미 접수된 작업은 끝까지 처리한다.
        return await asyncio.shield(wrapped)

    async def execute(
            self,
            sql: str,
            params: tuple[object, ...] = (),
    ) -> QueryResult:
        results = await self.transaction(
            (Statement(sql, params),)
        )

        return results[0]

    async def transaction(
            self,
            statements: tuple[Statement, ...],
    ) -> list[QueryResult]:
        """여러 SQL을 묶어 처리하고, 하나라도 실패 시 전체 롤백을 진행"""
        snapshot = tuple(statements)

        if not snapshot:
            raise ValueError("처리할 SQL이 필요합니다.")

        # 묶음 전체를 큐의 작업 하나로 넣는다.
        # 따라서 다른 요청의 SQL이 중간에 끼어들지 않는다.
        return await self._wait(
            self._submit(lambda: self._run(snapshot))
        )

    def _run(
            self,
            statements: tuple[Statement, ...],
    ) -> list[QueryResult]:
        """작업 스레드에서 SQL 묶음을 실행"""
        # 연결 객체를 가져온다. 연결 객체를 함수처럼 호출하지 않는다.
        connection = self._connection

        if connection is None:
            raise RuntimeError("사용 가능한 DB 연결이 없습니다.")

        try:
            connection.execute("BEGIN IMMEDIATE").close()

            results: list[QueryResult] = []

            for statement in statements:
                cursor = connection.execute(
                    statement.sql,
                    statement.params
                )

                try:
                    rows = (
                        [dict(row) for row in cursor.fetchall()]
                        if cursor.description is not None
                        else []
                    )

                    results.append(
                        QueryResult(
                            rows=rows,
                            rowcount=cursor.rowcount
                        )
                    )
                finally:
                    cursor.close()

            connection.commit()
            return results

        except Exception as e:
            logger.exception('DB 작업이 실패하여 롤백합니다.')

            try:
                connection.rollback()
            except Exception as rollback_e:
                self._connection = None

                try:
                    connection.close()
                except Exception:
                    logger.exception("오류가 발생한 DB 연결의 종료에도 실패했습니다.")

                raise ExceptionGroup(
                    "SQL 처리와 롤백이 모두 실패했습니다.",
                    [e, rollback_e]
                ) from None

            raise

    def _disconnect(self) -> None:
        """작업 스레드 종료"""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    async def close(self) -> None:
        """접수된 작업을 처리한 다음 연결과 작업 스레드를 종료"""
        with self._lock:
            if self._close_future is None:
                self._closing = True

                # 기존 작업 뒤에 종료 작업을 추가
                self._close_future = self._executor.submit(
                    self._disconnect
                )

            # 동시에 close()가 호출돼도 같은 종료 작업을 대기
            future = self._close_future

        try:
            await self._wait(future)

        finally:
            # 스레드 종료를 기다리느라 FastAPI 이벤트 루프를 막지 않음
            await asyncio.to_thread(
                self._executor.shutdown,
                wait=True,
                cancel_futures=False,
            )
