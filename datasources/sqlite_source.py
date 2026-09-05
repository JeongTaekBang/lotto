"""SQLite 데이터 소스 - bjt-blog 저장소의 data/lotto.db (draws 테이블) 읽기 전용

블로그 저장소가 관리하는 SQLite 파일을 원본으로 삼는다. 이 모듈은 절대 DB에 쓰지 않으며
(`mode=ro` URI 연결), DB보다 최신 회차가 동행복권 API에 있으면 메모리에서만 보충한다.
"""
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

from core.base_datasource import BaseDataSource
from core.types import LottoRecord

# 저장소 루트 (datasources/의 부모)
REPO_ROOT = Path(__file__).resolve().parent.parent

# 기본 DB 경로 (저장소 루트 기준 상대 경로)
DEFAULT_DB_PATH = "../bjt-blog/data/lotto.db"

# 동행복권 회차 정보 API
LOTTO_API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"

_SELECT_DRAWS = """
    SELECT round, n1, n2, n3, n4, n5, n6, bonus, rank1_winners, rank1_amount
    FROM draws
    ORDER BY round ASC
"""


def _optional_int(value) -> Optional[int]:
    """None을 허용하는 int 변환"""
    return None if value is None else int(value)


def resolve_db_path(db_path=None) -> Path:
    """DB 경로 결정: 인자 → LOTTO_DB_PATH 환경변수 → 기본값 (../bjt-blog/data/lotto.db)

    Args:
        db_path: 명시적 경로 (None이면 환경변수/기본값 사용)

    Returns:
        절대 경로 Path (존재 여부는 확인하지 않음)
    """
    load_dotenv()

    if db_path:
        return Path(db_path).expanduser().resolve()

    env_path = os.getenv('LOTTO_DB_PATH')
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (REPO_ROOT / DEFAULT_DB_PATH).resolve()


class SQLiteDataSource(BaseDataSource):
    """bjt-blog 저장소의 SQLite(draws 테이블) 기반 로또 데이터 소스 (읽기 전용)"""

    def __init__(self, db_path=None, fetch_missing: bool = True,
                 api_url: str = LOTTO_API_URL, timeout: int = 10):
        """
        Args:
            db_path: DB 경로 (None이면 LOTTO_DB_PATH → 기본값 순으로 결정)
            fetch_missing: DB보다 최신인 회차를 API로 메모리 보충할지 여부
            api_url: 동행복권 회차 정보 API URL
            timeout: API 요청 타임아웃(초)
        """
        super().__init__()
        self.db_path = resolve_db_path(db_path)
        self.fetch_missing = fetch_missing
        self.api_url = api_url
        self.timeout = timeout
        self.fetched_rounds: List[int] = []  # API로 보충한 회차 목록

    # ------------------------------------------------------------------
    # 로드
    # ------------------------------------------------------------------

    def load(self) -> List[LottoRecord]:
        """SQLite에서 데이터 로드 (필요 시 API로 최신 회차 메모리 보충)

        Raises:
            FileNotFoundError: DB 파일이 없을 때
            ValueError: draws 테이블 조회/파싱 실패 시
        """
        records = self._read_db()
        self.fetched_rounds = []

        if self.fetch_missing:
            records.extend(self._fetch_missing_records(records))

        self._records = records
        self._loaded = True

        suffix = f" (DB {len(records) - len(self.fetched_rounds)}회 + API 보충 {len(self.fetched_rounds)}회)" \
            if self.fetched_rounds else ""
        print(f"데이터 로드 완료: {len(records)}회차{suffix}")
        return self._records

    def _read_db(self) -> List[LottoRecord]:
        """SQLite draws 테이블을 읽기 전용으로 조회"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"로또 SQLite DB를 찾을 수 없습니다: {self.db_path}\n"
                f"LOTTO_DB_PATH 환경변수로 경로를 지정하거나, "
                f"bjt-blog 저장소를 {(REPO_ROOT / '../bjt-blog').resolve()} 에 두세요."
            )

        # mode=ro: 쓰기 시도 자체가 불가능하므로 블로그 DB 파일이 변경되지 않는다
        conn = sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute(_SELECT_DRAWS).fetchall()
        except sqlite3.Error as e:
            raise ValueError(f"draws 테이블 조회 실패 ({self.db_path}): {e}") from e
        finally:
            conn.close()

        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row) -> LottoRecord:
        """draws 행 → LottoRecord"""
        round_num, n1, n2, n3, n4, n5, n6, bonus, winners, prize = row
        numbers = [n1, n2, n3, n4, n5, n6]

        if bonus is None or any(n is None for n in numbers):
            raise ValueError(f"{round_num}회차 당첨번호에 빈 값이 있습니다: {row}")

        return LottoRecord(
            round_num=int(round_num),
            numbers=sorted(int(n) for n in numbers),
            bonus=int(bonus),
            winners=_optional_int(winners),
            prize=_optional_int(prize),
        )

    # ------------------------------------------------------------------
    # 동행복권 API 보충 (메모리 전용)
    # ------------------------------------------------------------------

    def fetch_latest_round(self) -> Optional[int]:
        """동행복권 API의 최신 회차 (조회 실패 시 None)"""
        info = self._request()
        return int(info['ltEpsd']) if info else None

    def _fetch_missing_records(self, records: List[LottoRecord]) -> List[LottoRecord]:
        """DB에 없는 최신 회차를 API에서 읽어 메모리로만 보충"""
        db_last = records[-1].round_num if records else 0
        latest = self.fetch_latest_round()

        if latest is None or latest <= db_last:
            return []

        fetched = []
        for round_num in range(db_last + 1, latest + 1):
            record = self._fetch_round(round_num)
            if record is None:
                # 회차가 끊기면 중단 (시퀀스 모델은 연속 회차를 전제로 한다)
                print(f"[경고] {round_num}회차 조회 실패 — {db_last + len(fetched)}회차까지만 사용합니다.")
                break
            fetched.append(record)
            self.fetched_rounds.append(round_num)

        if fetched:
            print(f"API 보충: {fetched[0].round_num}~{fetched[-1].round_num}회차 (메모리 전용, DB 미변경)")
        return fetched

    def _fetch_round(self, round_num: int) -> Optional[LottoRecord]:
        """특정 회차를 API에서 조회"""
        info = self._request({'srchLtEpsd': round_num})
        if not info or int(info.get('ltEpsd', 0)) != round_num:
            return None

        try:
            return LottoRecord(
                round_num=round_num,
                numbers=sorted(int(info[f'tm{i}WnNo']) for i in range(1, 7)),
                bonus=int(info['bnsWnNo']),
                winners=_optional_int(info.get('rnk1WnNope')),
                prize=_optional_int(info.get('rnk1WnAmt')),
            )
        except (KeyError, TypeError, ValueError) as e:
            print(f"[경고] {round_num}회차 응답 파싱 실패: {e}")
            return None

    def _request(self, params: dict = None) -> Optional[dict]:
        """API 호출 후 첫 회차 항목 반환 (실패 시 경고 후 None)"""
        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            items = (response.json().get('data') or {}).get('list') or []
        except (requests.RequestException, ValueError) as e:
            print(f"[경고] 동행복권 API 조회 실패: {e} (DB 데이터만 사용)")
            return None

        return items[0] if items else None


if __name__ == "__main__":
    # 테스트
    source = SQLiteDataSource()
    records = source.load()

    print(f"\nDB 경로: {source.db_path}")
    print(f"\n최근 5회차:")
    for rec in source.get_latest(5):
        print(f"  {rec.round_num}회: {rec.numbers} + {rec.bonus}")

    print(f"\n총 {len(source)}회차 로드됨")
