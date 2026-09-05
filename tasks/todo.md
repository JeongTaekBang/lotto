# TODO — MySQL → bjt-blog SQLite 데이터 소스 전환 (Coop WorkUnit #184)

## 배경
AWS MySQL 의존을 제거하고, bjt-blog 저장소가 관리하는 `data/lotto.db`(`draws` 테이블)를
읽기 전용 데이터 소스로 사용한다. 블로그 DB 파일에는 절대 쓰지 않는다.

## 작업 항목

- [x] `datasources/sqlite_source.py` — `SQLiteDataSource(BaseDataSource)` 신규
  - [x] 경로 결정: 생성자 인자 → `LOTTO_DB_PATH` → 기본값 `../bjt-blog/data/lotto.db`(저장소 루트 기준)
  - [x] `sqlite3` URI `mode=ro` 읽기 전용 연결 (쓰기 불가 보장)
  - [x] `draws` 스키마 파싱 → `LottoRecord` (회차 오름차순, 번호 정렬, winners/prize 매핑)
  - [x] 파일 부재 시 경로를 포함한 `FileNotFoundError`
  - [x] `fetch_missing=True`일 때 동행복권 API로 빠진 회차만 메모리 보충 (연속성 유지, 실패 시 경고 후 DB만 사용)
- [x] `core/base_datasource.py` — `get_raw_data()`를 베이스로 이동 (MySQL 전용 아님)
- [x] `datasources/__init__.py` — `SQLiteDataSource` export
- [x] `main_new.py` — 데이터 소스 교체, `--no-fetch` 옵션, `crawl` 재정의
  - [x] `crawl`: DB·API 회차 차이 보고 + 블로그 저장소 `git pull --ff-only` 시도(`--no-pull`로 생략)
- [x] `analysis/statistics_report.py`, `analysis/visualize_grid.py`, `scripts_xgboost_mode.py` 전환
- [x] `datasources/mysql_source.py`, `crawling.py` 삭제
- [x] `requirements.txt`에서 MySQL 드라이버 의존성 제거, `.env.example`의 DB 접속 항목 → `LOTTO_DB_PATH`
- [x] `readme.md`, `CLAUDE.md`, `CHANGELOG.md` 갱신
- [x] `tests/test_sqlite_source.py` 신규 + `pytest tests` 전체 통과

## 수용 기준 검증
- [x] 1) `predict --ensemble --sets 5`가 DB_* 없이 완주하고 로드 회차 수가 기대치와 일치
- [x] 2) `LOTTO_DB_PATH`로 다른 경로 사용 / 파일 부재 시 경로 포함 `FileNotFoundError`
- [x] 3) 실행 후 블로그 저장소 `data/lotto.db` 무변경
- [x] 4) `pytest tests` 전부 통과
- [x] 5) MySQL 드라이버 잔재 grep(.venv 제외) 결과 없음
- [x] 6) `main_new.py crawl`이 MySQL 없이 오류 없이 종료

## Review

### 구현 요약
- `datasources/sqlite_source.py` 신설. `sqlite3` URI `mode=ro` 연결이라 블로그 DB에 쓰기 자체가 불가능하다.
  경로는 생성자 인자 → `LOTTO_DB_PATH` → 저장소 루트 기준 `../bjt-blog/data/lotto.db` 순으로 결정하고
  절대 경로로 정규화해 오류 메시지에 그대로 싣는다.
- API 보충은 메모리 전용이며 회차 연속성이 끊기면 중단한다(시퀀스 모델 전제). API 실패는 경고 후 DB만 사용해
  예측이 네트워크에 의존하지 않는다.
- `get_raw_data()`는 MySQL 전용 로직이 아니라 `_records` 파생 헬퍼여서 `BaseDataSource`로 옮겼다.
- `crawl`은 DB·API 회차 차이를 보고하고 블로그 저장소에서 `git pull --ff-only`를 시도한다.
  git 저장소가 아니거나 pull이 실패해도 종료 코드는 0을 유지한다.
- `crawl` 자체가 API 보충을 쓰면 "부족한 회차"가 가려지므로 이 명령만 `fetch_missing=False`로 DB 원본을 읽는다.

### 검증 결과 (conda lotto / Python 3.12.9)
| 기준 | 결과 |
|---|---|
| 1) `predict --ensemble --sets 5` | 통과. DB_* 없이 완주, 1239회차 로드(DB 1237 + API 보충 1238~1239) = API 최신 회차 1239, 1240회차 예측 |
| 2) `LOTTO_DB_PATH` 지정 / 파일 부재 | 통과. 지정 경로에서 1237회차 로드, 부재 시 절대 경로 포함 `FileNotFoundError` |
| 3) 블로그 DB 무변경 | 통과. sha256 동일(`41cd34ae…`), `git status -- data/lotto.db` 비어 있음, `-wal`/`-shm` 미생성 |
| 4) `pytest tests` | 통과. 231개 전부 통과 (신규 `test_sqlite_source.py` 21개 포함) |
| 5) MySQL 드라이버 잔재 grep | 통과. 결과 0건 |
| 6) `main_new.py crawl` | 통과. exit 0. pull 성공/실패/비-git 3가지 경로 모두 확인 |

추가 스모크 테스트: `analyze`, `evaluate --no-fetch`, `scripts_xgboost_mode.py`, `analysis/visualize_grid.py` 모두 정상.
`visualize_grid.py`가 재생성한 `analysis/output/*.png`는 이번 작업 범위가 아니라 원본으로 되돌렸다.

### 남은 참고 사항
- 실제 블로그 저장소 대상 `git pull --ff-only`는 이 관리 세션 샌드박스가 `.git/FETCH_HEAD` 쓰기를 막아
  실패 경로로 종료했다(코드 결함 아님, HEAD 불변 확인). 성공 경로는 격리된 git 픽스처에서 fast-forward로 검증했다.
- `requirements.txt`의 `beautifulsoup4`/`lxml`은 현재 어디에서도 import되지 않지만 이번 범위 밖이라 두었다.
