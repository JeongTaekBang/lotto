# Changelog

이 프로젝트의 주요 변경 사항을 기록합니다.

---

## [2.1.0] - 2026-03-20

CNN Grid 모델을 v2로 업그레이드. 패턴 인식 능력 강화 및 학습 안정성 개선.

### Added
- **CBAM (Channel + Spatial Attention)**: 각 CNN branch와 fusion에 적용하여 중요한 채널/위치에 집중
- **Temporal Attention**: 20개 타임스텝에 learnable weight 적용, 최근 회차에 recency bias 초기화
- **Residual Connections**: 모든 branch + fusion + FC head에 skip connection 추가
- **Focal Loss** (alpha=0.75, gamma=2.0): 6/45 class imbalance 해결, 당첨 번호(positive)에 3배 가중
- **V1 호환 로딩**: `model_version` 키로 자동 분기, 기존 cnn_grid.pt 로드 가능
- **구 V2 체크포인트 마이그레이션**: BatchNorm→LayerNorm 전환 시 state_dict 키 자동 변환

### Changed
- CNN Grid 네트워크: `LottoCNNGridNet` (v2) 전면 교체, 기존은 `LottoCNNGridNetV1`으로 보존
- 5x5, 1x1 branch에 두 번째 conv layer 추가 (deeper feature extraction)
- FC Head: `Linear→ReLU→Dropout→Linear` → `Linear→LayerNorm→ReLU→Dropout→Linear→LayerNorm→ReLU→(residual)→Dropout→Linear`
- FC 정규화: BatchNorm1d → LayerNorm (batch size=1에서도 학습 가능)
- 파라미터: ~97K → ~146K (1.5x, 여전히 lightweight)
- 테스트: 16개 → 39개

---

## [2.0.0] - 2026-03-07

멀티 모델 앙상블 시스템으로 전면 재구축. 단일 스크립트(`lotto_predict.py`) 구조에서 모듈화된 OOP 아키텍처로 전환.

### Added
- **7종 예측 모델**: Transformer, LSTM, GRU, CNN Grid, XGBoost, RandomForest, Markov Chain
- **CNN Grid 모델**: 당첨번호를 7x7 이미지로 변환하여 공간 패턴을 학습하는 CNN (멀티 브랜치 3x3/5x5/1x1)
- **앙상블 시스템** (`ensemble/`): Weighted Average, Voting, Stacking 전략 지원
- **자동 가중치**: 백테스팅 성능 기반 모델 가중치 자동 계산 (softmax/linear/rank)
- **MMR 후보 선택** (`core/selector.py`): 대량 생성 → 스코어링 → 다양성 리랭킹. safe/balanced/aggressive 프리셋
- **필터 파이프라인** (`filters/`): PatternFilter, StatisticalFilter, FrequencyFilter, CompositeFilter
- **확장 피처**: 45차원(multi-hot) → 74차원(홀짝, AC값, 연번, 끝수분포, 주기성 등 29개 추가)
- **코어 추상화** (`core/`): BaseModel, BaseDataSource, BaseFilter, BaseEnsemble ABC 클래스
- **ModelFactory**: 플러그인 방식 모델 등록 및 생성
- **통합 트레이너** (`training/trainer.py`): 모든 모델의 학습/평가 파이프라인 통합
- **모델 평가기** (`training/evaluator.py`): 백테스팅, 모델 간 성능 비교
- **CLI 메뉴**: `lotto.bat` (Windows), `lotto.sh` (macOS) — conda 자동 감지, 패키지 자동 설치
- **Grid 시각화** (`analysis/visualize_grid.py`): 최근 회차 그리드, 빈도 히트맵, 예측 확률 히트맵, 종합 대시보드
- **통계 분석** (`analysis/statistics_report.py`): 번호별 출현 빈도, 구간별 분포 등
- **183개 유닛 테스트** (`tests/`): 모델, 앙상블, 필터, 피처 추출, 셀렉터, 트레이너, 통합 테스트
- **타입 시스템** (`core/types.py`): Prediction, ProbabilityDistribution, LottoRecord, EvaluationResult 등 dataclass
- **커스텀 손실함수** (`losses/`): FocalLoss, RankingLoss, CombinedLoss
- **QA 문서** (`QA.md`): 모델별 작동 원리, 피처 설명, 앙상블/필터 사용법 등 상세 Q&A

### Changed
- `lotto_predict.py` (단일 2000줄 스크립트) → 모듈화된 패키지 구조로 전환
- `lotto_crawling.py` → `crawling.py`로 리네임 및 리팩토링
- `readme.md` 전면 재작성 — 멀티 모델 아키텍처, CLI 명령어, 성능 비교표 반영
- 엔트리포인트: `lotto_predict.py` → `main_new.py`

### Removed
- `lotto_predict.py` — 단일 스크립트 방식 폐기
- `lotto_crawling.py` — `crawling.py`로 대체
- `create_database.md` — DB 생성 가이드 (readme에 통합)
- `LICENSE` — 라이선스 파일 제거
- `lotto_results.csv` — CSV 데이터 파일 (MySQL로 전환)
- 위치별 분포 PNG 이미지 6장 및 전체 분포 이미지

---

## [1.0.0] - 2025-03-02

초기 안정화. 데이터 정리 및 문서 업데이트.

### Changed
- `lotto_predict.py` 대폭 확장 (132줄 → 2237줄): 다양한 예측 로직 추가
- `lotto_crawling.py` 리팩토링
- `readme.md` 업데이트
- `.env.example`, `.gitignore`, `requirements.txt` 추가

### Removed
- 불필요한 파일 정리: CSV 데이터, 분포 이미지, `.gitignore` 재구성

---

## [0.1.0] - 2025-01-27

프로젝트 최초 생성.

### Added
- `lotto_predict.py` — 기본 로또 번호 예측 스크립트
- `lotto_crawling.py` — 당첨번호 크롤링
- `lotto_results.csv` — 역대 당첨번호 데이터
- `create_database.md` — MySQL 테이블 생성 가이드
- 위치별(1~6번째) 번호 분포 시각화 PNG
- `predict_lotto.md` — 예측 방법 설명 문서
