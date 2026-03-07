# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

멀티 모델 + 앙상블 기반 로또 번호 예측 시스템. 다양한 ML/DL 모델과 규칙 기반 필터를 조합하여 예측 정확도를 높인다.

**관련 문서:**
- [QA.md](QA.md) - 자주 묻는 질문 (모델 원리, 피처 설명, 사용법 등)

## Quick Start

### 메뉴 방식 (권장)
```bash
# lotto.bat 더블클릭 또는
lotto.bat
```

### 명령줄 방식
```bash
conda activate lotto
python main_new.py train --model=all      # 전체 모델 학습
python main_new.py predict --ensemble --auto-weight  # 앙상블 예측
```

## lotto.bat 메뉴 구조

```
==================================================
      Lotto AI - Multi-Model Prediction System
==================================================

  [1] Predict - Single Model       # 모델 선택 후 예측
  [2] Predict - Ensemble (Auto-Weight)  # 앙상블 (5개 모델, XGBoost 제외)
  [3] Train - Single Model         # 단일 모델 학습
  [4] Train - All Models           # 5개 모델 학습 (XGBoost 제외)
  [5] Compare Models               # 모델 성능 비교
  [6] Backtest                     # 백테스팅
  [7] Crawl Data                   # 데이터 크롤링
  [8] Statistics                   # 통계 분석
  [0] Exit
```

### 모델 선택 시
```
Available Models:
  [1] gru           (Sequence)
  [2] transformer   (Attention)
  [3] random_forest (Ensemble)
  [4] markov        (Statistical)
  [5] lstm          (Sequence)
  [6] xgboost       (Boosting)  # 단일 모델 전용
```

### 피처 모드 (자동 결정)
- **앙상블/Train All**: Extended (74차원), XGBoost 제외
- **XGBoost 단독**: Basic (45차원)
- **기타 단일 모델**: Extended (74차원)

## Commands (CLI)
```bash
# 모델 학습 (피처 모드 자동 결정)
python main_new.py train --model=xgboost         # XGBoost: Basic 45차원
python main_new.py train --model=gru             # 기타: Extended 74차원
python main_new.py train --model=all             # 5개 모델 (XGBoost 제외)

# 번호 예측
python main_new.py predict --model=xgboost       # XGBoost 단독 (Basic)
python main_new.py predict --ensemble --auto-weight  # 앙상블 5개 모델 (Extended)
python main_new.py predict --ensemble --auto-weight --filter  # + 필터

# 자동 가중치 옵션
python main_new.py predict --ensemble --auto-weight --weight-method=softmax  # 기본값
python main_new.py predict --ensemble --auto-weight --weight-method=linear   # 성능 비례
python main_new.py predict --ensemble --auto-weight --weight-method=rank     # 순위 기반

# 모델 평가/비교
python main_new.py evaluate --model=xgboost --rounds=100
python main_new.py compare --rounds=100

# 유틸리티
python main_new.py list    # 모델 목록
python main_new.py crawl   # 데이터 크롤링
python main_new.py analyze # 통계 분석
```

## 모델 성능 비교

### 확장 피처 (74차원) - 권장
| 순위 | 모델 | 평균 적중 | 3개+ 적중률 | 특징 |
|:---:|------|:---:|:---:|------|
| 1 | **gru** | **0.82** | 1.0% | LSTM 경량화, 시퀀스 모델 |
| 2 | transformer | 0.81 | 2.0% | 장기 의존성, Attention |
| 3 | random_forest | 0.80 | 1.0% | 랜덤 포레스트 앙상블 |
| 4 | markov | 0.77 | 1.0% | 전이확률 기반 |
| 5 | lstm | 0.73 | 0.0% | 시퀀스 패턴 학습 |
| 6 | xgboost | 0.70 | 1.0% | Gradient Boosting |

### 기본 피처 (45차원)
| 순위 | 모델 | 평균 적중 |
|:---:|------|:---:|
| 1 | xgboost | 0.94 |
| 2 | markov | 0.77 |
| 3~6 | 나머지 | 0.68~0.69 |

**참고**: 랜덤 기대값 = 0.80개

### 피처 모드 자동 결정

| 상황 | 피처 모드 | 모델 |
|------|-----------|------|
| Train All / Ensemble | Extended (74차원) | 5개 (XGBoost 제외) |
| XGBoost 단독 | Basic (45차원) | XGBoost만 |
| 기타 단일 모델 | Extended (74차원) | 선택한 모델 |

## 자동 가중치 시스템

앙상블 예측 시 `--auto-weight` 옵션으로 성능 기반 가중치 자동 계산:

```
자동 가중치 계산 완료:
  gru            : 0.22 (성능: 0.82)  ← 최고 가중치
  transformer    : 0.21 (성능: 0.81)
  random_forest  : 0.20 (성능: 0.80)
  markov         : 0.19 (성능: 0.77)
  lstm           : 0.18 (성능: 0.73)
```

## Claude Code 실행 방법

Claude Code에서 Python 실행 시 conda lotto 환경의 python을 직접 호출:

```bash
# 권장 실행 방법
powershell.exe -Command "& 'C:\Users\since\anaconda3\envs\lotto\python.exe' 'C:\Users\since\Dropbox\bjtPersonalProjects\lotto\main_new.py' <command>"
```

예시:
```bash
# 전체 모델 학습
powershell.exe -Command "& 'C:\Users\since\anaconda3\envs\lotto\python.exe' 'C:\Users\since\Dropbox\bjtPersonalProjects\lotto\main_new.py' train --model=all"

# 앙상블 예측 (자동 가중치)
powershell.exe -Command "& 'C:\Users\since\anaconda3\envs\lotto\python.exe' 'C:\Users\since\Dropbox\bjtPersonalProjects\lotto\main_new.py' predict --ensemble --auto-weight --filter"

# 모델 비교
powershell.exe -Command "& 'C:\Users\since\anaconda3\envs\lotto\python.exe' 'C:\Users\since\Dropbox\bjtPersonalProjects\lotto\main_new.py' compare --rounds=100"
```

## Architecture

### 디렉토리 구조
```
lotto/
├── lotto.bat                # 메뉴 기반 CLI (권장)
├── main_new.py              # CLI (멀티 모델 지원)
│
├── core/                    # 핵심 추상 클래스 (ABC)
│   ├── base_model.py       # BaseModel - 모든 모델의 부모
│   ├── base_datasource.py  # BaseDataSource - 데이터 소스 인터페이스
│   ├── base_ensemble.py    # BaseEnsemble - 앙상블 전략 인터페이스
│   ├── base_filter.py      # BaseFilter - 필터 인터페이스
│   ├── base_loss.py        # BaseLoss - 손실함수 인터페이스
│   └── types.py            # Prediction, ProbabilityDistribution 등
│
├── models/                  # 모델 구현체
│   ├── factory.py          # ModelFactory (팩토리 패턴)
│   ├── transformer.py      # Transformer Encoder
│   ├── lstm.py             # LSTM
│   ├── gru.py              # GRU
│   ├── xgboost_model.py    # XGBoost (45개 이진분류기) ★ Best
│   ├── random_forest.py    # RandomForest
│   └── markov.py           # 마르코프 체인
│
├── ensemble/                # 앙상블 시스템
│   ├── manager.py          # EnsembleManager (자동 가중치 지원)
│   ├── voting.py           # 다수결 투표
│   ├── weighted_average.py # 가중 평균
│   └── stacking.py         # 메타 모델 스태킹
│
├── filters/                 # 규칙 기반 필터
│   ├── pattern_filter.py   # 홀짝, 고저, 연번, 합계
│   ├── statistical_filter.py # AC값, 끝수 분포
│   ├── frequency_filter.py # 빈도 기반 (hot/cold)
│   └── composite_filter.py # 복합 필터 체인
│
├── datasources/             # 데이터 소스
│   └── mysql_source.py     # MySQL 데이터 로더
│
├── training/                # 학습 관련
│   ├── trainer.py          # UnifiedTrainer (모든 모델 학습)
│   └── evaluator.py        # ModelEvaluator (백테스팅)
│
├── losses/                  # 손실함수
│   ├── bce.py, focal.py, ranking.py, combined.py
│
├── analysis/                # 통계 분석
│   └── statistics_report.py
│
└── saved_models/            # 저장된 모델들
    ├── transformer.pt, lstm.pt, gru.pt
    └── xgboost.pkl, random_forest.pkl, markov.pkl
```

### 핵심 설계 패턴

1. **Factory Pattern**: `ModelFactory.create('xgboost')` - 모델 생성
2. **Strategy Pattern**: `EnsembleManager.set_strategy()` - 앙상블 전략 교체
3. **Composite Pattern**: `CompositeFilter` - 필터 체인 조합
4. **Dependency Injection**: 트레이너에 데이터소스/모델 주입

### 데이터 흐름
```
MySQLDataSource → UnifiedTrainer → Model.train()
                                       ↓
                              saved_models/*.pt|pkl
                                       ↓
EnsembleManager ← Model.predict_proba() ← Model.load()
       ↓
   auto_weight_by_performance() → 성능 기반 가중치
       ↓
   Filters (optional)
       ↓
   Predictions
```

## Database

MySQL `lotto` 테이블:
- `count`: 회차
- `1`~`6`: 당첨번호
- `7`: 보너스번호
- `person`: 당첨자 수 (쉼표 포함 문자열 지원)
- `amount`: 당첨금액 (쉼표 포함 문자열 지원)

## Environment

`.env` 파일에 DB 연결 정보 설정:
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=username
DB_PASSWORD=password
DB_NAME=lotto
```

## Dependencies

```bash
# 기본
pip install torch numpy pymysql python-dotenv requests beautifulsoup4 lxml

# 멀티 모델용 추가
pip install xgboost scikit-learn
```

## 코드 예시

### 앙상블 예측 (권장)
```python
from models.factory import ModelFactory
from ensemble.manager import EnsembleManager
from ensemble.weighted_average import WeightedAverageEnsemble
from training.trainer import UnifiedTrainer
from datasources.mysql_source import MySQLDataSource

# 데이터 준비 (Extended 피처)
datasource = MySQLDataSource()
trainer = UnifiedTrainer(datasource, feature_mode='extended')
X, y = trainer.prepare_sequences()

# 모델 로드 및 등록 (XGBoost 제외)
manager = EnsembleManager()
for name in ['gru', 'transformer', 'random_forest', 'markov', 'lstm']:
    model = ModelFactory.create(name, {'input_dim': trainer.feature_dim})
    ext = '.pkl' if name in ['random_forest', 'markov'] else '.pt'
    model.load(f'saved_models/{name}{ext}')
    manager.register_model(name, model)

# 앙상블 전략 및 자동 가중치
manager.set_strategy(WeightedAverageEnsemble())
manager.auto_weight_by_performance(X[-100:], y[-100:], method='softmax')

# 예측
predictions = manager.predict(trainer.get_latest_sequence(), num_sets=5)
for p in predictions:
    print(f"{p.numbers} (신뢰도: {p.confidence:.3f})")
```

### XGBoost 단독 사용
```python
from models.factory import ModelFactory
from training.trainer import UnifiedTrainer
from datasources.mysql_source import MySQLDataSource

# Basic 피처로 트레이너 설정
trainer = UnifiedTrainer(MySQLDataSource(), feature_mode='basic')

model = ModelFactory.create('xgboost')
model.load('saved_models/xgboost.pkl')
predictions = model.predict_numbers(trainer.get_latest_sequence(), num_sets=5)
```
