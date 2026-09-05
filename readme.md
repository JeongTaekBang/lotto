# Lotto AI - 멀티 모델 앙상블 로또 번호 예측

다양한 ML/DL 모델과 규칙 기반 필터를 조합하여 로또 당첨번호 패턴을 분석하고 예측하는 시스템입니다.

📖 **[Q&A 문서](QA.md)** - 자주 묻는 질문과 상세 설명

## 기능

- **멀티 모델**: Transformer, LSTM, GRU, XGBoost, RandomForest, Markov Chain, CNN Grid
- **앙상블 예측**: 성능 기반 자동 가중치로 여러 모델 조합
- **규칙 기반 필터**: 홀짝/고저 비율, AC값, 빈도 분석 등 통계적 필터링
- **메뉴 기반 CLI**: lotto.bat으로 간편 사용
- **백테스팅**: 과거 데이터로 모델 성능 비교

## 빠른 시작

### 메뉴 방식 (권장)
```bash
# lotto.bat 더블클릭
```

### 명령줄 방식
```bash
conda activate lotto
python main_new.py train --model=all           # 전체 모델 학습
python main_new.py predict --ensemble --auto-weight  # 앙상블 예측
```

## 설치

```bash
conda activate lotto
pip install -r requirements.txt
```

## 환경 설정

당첨번호 데이터는 [bjt-blog](https://github.com/JeongTaekBang/bjt-blog) 저장소의
`data/lotto.db`(SQLite)를 **읽기 전용**으로 사용합니다. 별도의 DB 서버가 필요 없습니다.

기본 경로는 이 저장소 기준 `../bjt-blog/data/lotto.db`이며, 다른 위치에 두었다면
`.env` 파일이나 환경변수로 지정합니다:

```
LOTTO_DB_PATH=/path/to/bjt-blog/data/lotto.db
```

DB보다 최신 회차가 동행복권에 있으면 실행 시 API로 **메모리에만** 보충하며,
블로그 저장소의 DB 파일은 절대 변경하지 않습니다. 보충을 끄려면 `--no-fetch`를 씁니다.

## lotto.bat 메뉴

```
==================================================
      Lotto AI - Multi-Model Prediction System
==================================================

  [1] Predict - Single Model       # 모델 선택 후 예측
  [2] Predict - Ensemble (Auto-Weight)  # 앙상블 (자동 가중치)
  [3] Train - Single Model         # 단일 모델 학습
  [4] Train - All Models           # 6개 모델 학습 (XGBoost 제외)
  [5] Compare Models               # 모델 성능 비교
  [6] Backtest                     # 백테스팅
  [7] Check Data Status            # DB·API 회차 차이 확인 + 블로그 저장소 갱신
  [8] Statistics                   # 통계 분석
  [0] Exit
```

## 모델 성능 비교

### 확장 피처 (74차원) - 권장
| 순위 | 모델 | 평균 적중 | 3개+ 적중률 |
|:---:|------|:---:|:---:|
| 1 | **gru** | **0.82** | 1.0% |
| 2 | transformer | 0.81 | 2.0% |
| 3 | random_forest | 0.80 | 1.0% |
| 4 | markov | 0.77 | 1.0% |
| 5 | lstm | 0.73 | 0.0% |
| 6 | xgboost | 0.70 | 1.0% |

### 기본 피처 (45차원)
| 순위 | 모델 | 평균 적중 |
|:---:|------|:---:|
| 1 | xgboost | **0.94** |
| 2~6 | 나머지 | 0.68~0.77 |

**참고**: 랜덤 기대값 = 0.80개

## CLI 명령어

```bash
# 모델 학습
python main_new.py train --model=xgboost      # 단일 모델
python main_new.py train --model=all          # 전체 모델
python main_new.py train --model=all --feature-mode=extended  # 확장 피처

# 피처 모드 옵션
--feature-mode=basic     # 45차원 (기본)
--feature-mode=extended  # 74차원 (전체 확장 피처)
--feature-mode=xgboost   # 66차원 (XGBoost 최적화)

# 번호 예측
python main_new.py predict --model=xgboost    # 단일 모델
python main_new.py predict --ensemble --auto-weight  # 앙상블 + 자동 가중치
python main_new.py predict --ensemble --auto-weight --filter  # + 필터

# 평가
python main_new.py compare --rounds=100       # 전체 모델 비교
python main_new.py evaluate --model=xgboost   # 단일 모델 평가

# 유틸리티
python main_new.py list               # 모델 목록
python main_new.py crawl              # DB·API 회차 차이 확인 + 블로그 저장소 git pull --ff-only
python main_new.py crawl --no-pull    # 상태 확인만
python main_new.py analyze            # 통계 분석

# 데이터 옵션
--no-fetch   # DB보다 최신인 회차를 API로 보충하지 않음 (train/predict/evaluate/compare)
```

## 프로젝트 구조

```
lotto/
├── lotto.bat                # 메뉴 기반 CLI (권장)
├── main_new.py              # CLI (멀티 모델 지원)
│
├── core/                    # 추상 클래스 (ABC)
├── models/                  # 모델 구현체 (7종)
├── ensemble/                # 앙상블 시스템
├── filters/                 # 규칙 기반 필터
├── training/                # 학습/평가
├── datasources/             # 데이터 소스
└── saved_models/            # 저장된 모델
```

## 자동 가중치 시스템

`--auto-weight` 옵션 사용 시 백테스팅 성능 기반으로 가중치 자동 계산:

### 확장 피처 (권장)
```
  gru            : 0.19 (성능: 0.82)  ← 최고
  transformer    : 0.18 (성능: 0.81)
  random_forest  : 0.17 (성능: 0.80)
  markov         : 0.17 (성능: 0.77)
  lstm           : 0.15 (성능: 0.73)
  xgboost        : 0.14 (성능: 0.70)
```

### 기본 피처
```
  xgboost        : 0.24 (성능: 0.94)  ← 압도적
```

## 앙상블 전략

- **Weighted Average**: 각 모델의 확률 분포를 가중 평균 (기본값)
- **Voting**: Top-K 번호에 대한 다수결 투표
- **Stacking**: 메타 모델이 각 모델의 예측을 조합

## 필터 시스템

- **PatternFilter**: 홀짝 비율 (2~4), 고저 비율 (2~4), 연번 제한
- **StatisticalFilter**: AC값 (7~10), 합계 범위 (100~175)
- **FrequencyFilter**: Hot/Cold 번호 균형

## 아키텍처 설계

### 핵심 패턴
- **Factory Pattern**: 모델 생성 (`ModelFactory.create()`)
- **Strategy Pattern**: 앙상블 전략 교체
- **Composite Pattern**: 필터 체인 조합
- **Dependency Injection**: 유연한 컴포넌트 주입

### 확장성
새 모델 추가 시:
1. `core.base_model.BaseModel` 상속
2. `train()`, `predict_proba()`, `save()`, `load()` 구현
3. `models/factory.py`에 등록

## 데이터베이스

bjt-blog 저장소의 SQLite 파일 `data/lotto.db`, 테이블 `draws` (읽기 전용)

| 필드 | 설명 |
|------|------|
| round | 회차 |
| draw_date | 추첨일 (YYYY-MM-DD) |
| n1~n6 | 당첨번호 |
| bonus | 보너스번호 |
| rank1_winners | 1등 당첨자 수 |
| rank1_amount | 1등 1인당 당첨금 |
| total_sales | 회차 총 판매액 |
