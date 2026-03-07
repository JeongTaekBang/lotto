"""프로젝트 전역 상수 정의"""
from typing import Dict, Set


# === 로또 게임 규칙 ===
LOTTO_MIN_NUMBER = 1
LOTTO_MAX_NUMBER = 45
LOTTO_NUMBERS_COUNT = 6  # 선택하는 번호 개수
LOTTO_POOL_SIZE = 45     # 전체 번호 풀


# === 피처 차원 ===
FEATURE_DIM_BASIC = 45       # multi-hot only
FEATURE_DIM_EXTENDED = 74    # 전체 확장 피처 (45 + 29)
FEATURE_DIM_XGBOOST = 66     # 중복 제거된 확장 피처 (45 + 21)
FEATURE_DIM_BONUS = 1        # 보너스 번호 피처


# === 시퀀스 설정 ===
DEFAULT_SEQ_LENGTH = 20      # 기본 시퀀스 길이
DEFAULT_PERIOD = 52          # 주기성 피처 기간 (주)


# === 학습 설정 ===
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_RATIO = 0.1


# === 당첨금 테이블 ===
PRIZE_TABLE: Dict[int, int] = {
    3: 5_000,           # 5등: 5천원
    4: 50_000,          # 4등: 5만원
    5: 1_500_000,       # 3등: 150만원 (고정)
    6: 2_000_000_000,   # 1등: 20억 (평균 추정)
}


# === 수학적 상수 ===
PRIMES: Set[int] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


# === 모델 저장 경로 ===
DEFAULT_MODEL_DIR = "saved_models"


# === 모델별 확장자 ===
MODEL_EXTENSIONS: Dict[str, str] = {
    'transformer': '.pt',
    'lstm': '.pt',
    'gru': '.pt',
    'cnn_grid': '.pt',
    'xgboost': '.pkl',
    'random_forest': '.pkl',
    'markov': '.pkl',
}


# === 앙상블 기본 모델 (XGBoost 제외) ===
ENSEMBLE_DEFAULT_MODELS = ['gru', 'transformer', 'random_forest', 'markov', 'lstm', 'cnn_grid']


# === 피처 모드 ===
class FeatureMode:
    BASIC = 'basic'
    EXTENDED = 'extended'
    XGBOOST = 'xgboost'


def get_feature_dim(feature_mode: str, use_bonus: bool = False) -> int:
    """피처 모드에 따른 차원 반환"""
    dims = {
        FeatureMode.BASIC: FEATURE_DIM_BASIC,
        FeatureMode.EXTENDED: FEATURE_DIM_EXTENDED,
        FeatureMode.XGBOOST: FEATURE_DIM_XGBOOST,
    }
    base_dim = dims.get(feature_mode, FEATURE_DIM_BASIC)
    return base_dim + (FEATURE_DIM_BONUS if use_bonus else 0)
