"""SQLiteDataSource 테스트 - 스키마 파싱, 정렬, 매핑, 경로 결정, API 보충"""
import hashlib
import sqlite3
from pathlib import Path

import pytest

from core.types import LottoRecord
from datasources import sqlite_source
from datasources.sqlite_source import (
    DEFAULT_DB_PATH,
    REPO_ROOT,
    SQLiteDataSource,
    resolve_db_path,
)

# bjt-blog data/lotto.db 와 동일한 스키마
SCHEMA = """
CREATE TABLE draws (
  round         INTEGER PRIMARY KEY,
  draw_date     TEXT    NOT NULL,
  n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER, n6 INTEGER,
  bonus         INTEGER NOT NULL,
  rank1_winners INTEGER,
  rank1_amount  INTEGER,
  total_sales   INTEGER
);
"""

# 일부러 회차 순서를 섞고(2, 1, 3), 2회차 번호도 정렬되지 않은 상태로 넣는다
ROWS = [
    (2, '2002-12-14', 9, 2, 25, 32, 42, 14, 16, 1, 2002006800, 3_000_000_000),
    (1, '2002-12-07', 10, 23, 29, 33, 37, 40, 16, 0, 0, 4_000_000_000),
    (3, '2002-12-21', 11, 16, 19, 21, 27, 31, 30, None, None, None),
]


@pytest.fixture
def temp_db(tmp_path) -> Path:
    """임시 sqlite DB (draws 테이블)"""
    db_path = tmp_path / "lotto.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO draws VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", ROWS
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def source(temp_db) -> SQLiteDataSource:
    """API 보충을 끈 데이터 소스"""
    return SQLiteDataSource(db_path=temp_db, fetch_missing=False)


def _no_network(*args, **kwargs):
    raise AssertionError("네트워크 호출이 발생하면 안 됩니다")


# ============================================================
# 스키마 파싱 / 정렬 / 매핑
# ============================================================

class TestSchemaParsing:

    def test_load_returns_all_rounds(self, source):
        records = source.load()
        assert len(records) == 3
        assert all(isinstance(r, LottoRecord) for r in records)

    def test_records_sorted_by_round_ascending(self, source):
        records = source.load()
        assert [r.round_num for r in records] == [1, 2, 3]

    def test_numbers_are_sorted(self, source):
        records = source.load()
        # DB에는 (9, 2, 25, 32, 42, 14) 순서로 저장되어 있다
        assert records[1].numbers == [2, 9, 14, 25, 32, 42]
        assert all(r.numbers == sorted(r.numbers) for r in records)

    def test_bonus_mapping(self, source):
        records = source.load()
        assert [r.bonus for r in records] == [16, 16, 30]

    def test_winners_and_prize_mapping(self, source):
        records = source.load()
        assert (records[0].winners, records[0].prize) == (0, 0)
        assert (records[1].winners, records[1].prize) == (1, 2002006800)
        # NULL 은 None 으로 매핑된다
        assert (records[2].winners, records[2].prize) == (None, None)

    def test_get_raw_data_shape(self, source):
        raw = source.get_raw_data()
        assert raw.shape == (3, 8)
        assert list(raw[1]) == [2, 2, 9, 14, 25, 32, 42, 16]

    def test_malformed_row_raises(self, tmp_path):
        db_path = tmp_path / "broken.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO draws VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (1, '2002-12-07', 10, None, 29, 33, 37, 40, 16, 0, 0, 0),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ValueError, match="빈 값"):
            SQLiteDataSource(db_path=db_path, fetch_missing=False).load()

    def test_missing_table_raises_value_error(self, tmp_path):
        db_path = tmp_path / "empty.db"
        sqlite3.connect(db_path).close()

        with pytest.raises(ValueError, match="draws"):
            SQLiteDataSource(db_path=db_path, fetch_missing=False).load()


# ============================================================
# 경로 결정 / 파일 부재
# ============================================================

class TestPathResolution:

    def test_explicit_path_wins(self, temp_db, monkeypatch):
        monkeypatch.setenv('LOTTO_DB_PATH', '/nonexistent/other.db')
        assert resolve_db_path(temp_db) == temp_db.resolve()

    def test_env_path_used(self, temp_db, monkeypatch):
        monkeypatch.setenv('LOTTO_DB_PATH', str(temp_db))
        assert resolve_db_path() == temp_db.resolve()

    def test_env_path_loads_that_file(self, temp_db, monkeypatch):
        monkeypatch.setenv('LOTTO_DB_PATH', str(temp_db))
        ds = SQLiteDataSource(fetch_missing=False)
        assert ds.db_path == temp_db.resolve()
        assert len(ds.load()) == 3

    def test_default_path_is_blog_repo(self, monkeypatch):
        # .env 로 인한 오염 없이 기본값만 검증한다
        monkeypatch.setattr(sqlite_source, 'load_dotenv', lambda *a, **kw: None)
        monkeypatch.delenv('LOTTO_DB_PATH', raising=False)
        assert resolve_db_path() == (REPO_ROOT / DEFAULT_DB_PATH).resolve()

    def test_missing_file_raises_with_path(self, tmp_path):
        missing = tmp_path / "does_not_exist.db"
        ds = SQLiteDataSource(db_path=missing, fetch_missing=False)

        with pytest.raises(FileNotFoundError) as excinfo:
            ds.load()
        assert str(missing.resolve()) in str(excinfo.value)


# ============================================================
# API 보충 (fetch_missing)
# ============================================================

class TestFetchMissing:

    def test_fetch_missing_false_skips_network(self, temp_db, monkeypatch):
        monkeypatch.setattr(sqlite_source.requests, 'get', _no_network)

        records = SQLiteDataSource(db_path=temp_db, fetch_missing=False).load()

        assert len(records) == 3
        assert [r.round_num for r in records] == [1, 2, 3]

    def test_fetch_missing_false_reports_no_supplement(self, source):
        source.load()
        assert source.fetched_rounds == []

    def test_supplements_only_missing_rounds(self, temp_db, monkeypatch):
        api_rounds = {
            4: {'ltEpsd': 4, 'tm1WnNo': 40, 'tm2WnNo': 3, 'tm3WnNo': 12,
                'tm4WnNo': 22, 'tm5WnNo': 30, 'tm6WnNo': 44, 'bnsWnNo': 7,
                'rnk1WnNope': 2, 'rnk1WnAmt': 1_000_000_000},
            5: {'ltEpsd': 5, 'tm1WnNo': 1, 'tm2WnNo': 5, 'tm3WnNo': 9,
                'tm4WnNo': 13, 'tm5WnNo': 25, 'tm6WnNo': 45, 'bnsWnNo': 8,
                'rnk1WnNope': None, 'rnk1WnAmt': None},
        }

        def fake_request(self, params=None):
            if params is None:
                return api_rounds[5]  # 최신 회차
            return api_rounds.get(params['srchLtEpsd'])

        monkeypatch.setattr(SQLiteDataSource, '_request', fake_request)

        ds = SQLiteDataSource(db_path=temp_db, fetch_missing=True)
        records = ds.load()

        assert [r.round_num for r in records] == [1, 2, 3, 4, 5]
        assert ds.fetched_rounds == [4, 5]
        assert records[3].numbers == [3, 12, 22, 30, 40, 44]
        assert (records[3].winners, records[3].prize) == (2, 1_000_000_000)
        assert (records[4].winners, records[4].prize) == (None, None)

    def test_no_supplement_when_db_is_current(self, temp_db, monkeypatch):
        monkeypatch.setattr(
            SQLiteDataSource, '_request',
            lambda self, params=None: {'ltEpsd': 3} if params is None else None
        )

        ds = SQLiteDataSource(db_path=temp_db, fetch_missing=True)

        assert len(ds.load()) == 3
        assert ds.fetched_rounds == []

    def test_api_failure_falls_back_to_db(self, temp_db, monkeypatch):
        monkeypatch.setattr(SQLiteDataSource, '_request', lambda self, params=None: None)

        ds = SQLiteDataSource(db_path=temp_db, fetch_missing=True)

        assert len(ds.load()) == 3
        assert ds.fetched_rounds == []

    def test_stops_at_first_gap(self, temp_db, monkeypatch):
        def fake_request(self, params=None):
            if params is None:
                return {'ltEpsd': 6}
            if params['srchLtEpsd'] == 4:
                return {'ltEpsd': 4, 'tm1WnNo': 1, 'tm2WnNo': 2, 'tm3WnNo': 3,
                        'tm4WnNo': 4, 'tm5WnNo': 5, 'tm6WnNo': 6, 'bnsWnNo': 7,
                        'rnk1WnNope': 1, 'rnk1WnAmt': 1}
            return None  # 5회차부터 조회 실패

        monkeypatch.setattr(SQLiteDataSource, '_request', fake_request)

        ds = SQLiteDataSource(db_path=temp_db, fetch_missing=True)
        records = ds.load()

        # 연속성이 끊기면 중단해 회차가 이어지도록 유지한다
        assert [r.round_num for r in records] == [1, 2, 3, 4]
        assert ds.fetched_rounds == [4]


# ============================================================
# 읽기 전용 보장
# ============================================================

class TestReadOnly:

    def test_db_file_unchanged_after_load(self, temp_db):
        digest_before = hashlib.sha256(temp_db.read_bytes()).hexdigest()

        SQLiteDataSource(db_path=temp_db, fetch_missing=False).load()

        assert hashlib.sha256(temp_db.read_bytes()).hexdigest() == digest_before

    def test_connection_rejects_writes(self, temp_db):
        conn = sqlite3.connect(f"{temp_db.resolve().as_uri()}?mode=ro", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM draws")
        finally:
            conn.close()
