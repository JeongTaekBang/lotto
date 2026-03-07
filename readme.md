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
# 필요한 패키지 설치
pip install torch numpy pymysql python-dotenv requests beautifulsoup4 lxml

# 멀티 모델용 추가 패키지
pip install xgboost scikit-learn
```

## 환경 설정

`.env` 파일 생성:
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=lotto
```

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
  [7] Crawl Data                   # 데이터 크롤링
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
python main_new.py list     # 모델 목록
python main_new.py crawl    # 데이터 크롤링
python main_new.py analyze  # 통계 분석
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

테이블: `lotto`

| 필드 | 설명 |
|------|------|
| count | 회차 |
| 1~6 | 당첨번호 |
| 7 | 보너스번호 |
| person | 당첨자 수 |
| amount | 당첨금액 |
