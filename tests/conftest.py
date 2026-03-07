"""pytest 공통 fixtures"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import LottoRecord, Prediction, ProbabilityDistribution
from core.base_model import BaseModel
from core.base_datasource import BaseDataSource


# ============================================================
# 샘플 데이터 Fixtures
# ============================================================

@pytest.fixture
def sample_numbers() -> List[int]:
    """샘플 로또 번호 (6개)"""
    return [1, 10, 20, 30, 40, 45]


@pytest.fixture
def sample_numbers_list() -> List[List[int]]:
    """여러 샘플 로또 번호 세트"""
    return [
        [1, 10, 20, 30, 40, 45],
        [3, 12, 23, 34, 41, 44],
        [5, 15, 25, 35, 42, 43],
        [7, 17, 27, 37, 38, 39],
        [2, 11, 22, 33, 36, 45],
    ]


@pytest.fixture
def sample_record() -> LottoRecord:
    """샘플 LottoRecord"""
    return LottoRecord(
        round_num=1000,
        numbers=[1, 10, 20, 30, 40, 45],
        bonus=7,
        winners=10,
        prize=1500000000
    )


@pytest.fixture
def sample_records() -> List[LottoRecord]:
    """샘플 LottoRecord 리스트 (30개)"""
    records = []
    np.random.seed(42)
    for i in range(30):
        nums = sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())
        bonus = np.random.choice([n for n in range(1, 46) if n not in nums])
        records.append(LottoRecord(
            round_num=1000 + i,
            numbers=nums,
            bonus=int(bonus),
            winners=int(np.random.randint(5, 20)),
            prize=int(np.random.randint(1000000, 3000000) * 1000)  # 10억~30억
        ))
    return records


@pytest.fixture
def sample_prediction() -> Prediction:
    """샘플 Prediction"""
    return Prediction(
        numbers=[1, 10, 20, 30, 40, 45],
        confidence=0.85,
        model_name="test_model"
    )


@pytest.fixture
def sample_predictions() -> List[Prediction]:
    """샘플 Prediction 리스트"""
    return [
        Prediction(numbers=[1, 10, 20, 30, 40, 45], confidence=0.85, model_name="model_a"),
        Prediction(numbers=[2, 11, 21, 31, 41, 44], confidence=0.80, model_name="model_b"),
        Prediction(numbers=[3, 12, 22, 32, 42, 43], confidence=0.75, model_name="model_c"),
    ]


@pytest.fixture
def sample_proba() -> ProbabilityDistribution:
    """샘플 확률 분포"""
    np.random.seed(42)
    probs = np.random.rand(45)
    probs = probs / probs.sum()  # 정규화
    return ProbabilityDistribution(
        probabilities=probs,
        model_name="test_model"
    )


# ============================================================
# Mock Datasource Fixture
# ============================================================

class MockDataSource(BaseDataSource):
    """테스트용 Mock 데이터소스"""

    def __init__(self, records: List[LottoRecord] = None):
        super().__init__()
        self._mock_records = records or []

    def load(self) -> List[LottoRecord]:
        """Mock 데이터 로드"""
        self._records = self._mock_records
        self._loaded = True
        return self._records


@pytest.fixture
def mock_datasource(sample_records) -> MockDataSource:
    """Mock 데이터소스 fixture"""
    return MockDataSource(sample_records)


# ============================================================
# 학습 데이터 Fixtures
# ============================================================

@pytest.fixture
def sample_X_sequence() -> np.ndarray:
    """시퀀스 모델용 입력 데이터 (batch, seq_length, feature_dim)"""
    np.random.seed(42)
    return np.random.rand(10, 20, 74).astype(np.float32)


@pytest.fixture
def sample_X_flat() -> np.ndarray:
    """Flat 모델용 입력 데이터 (batch, feature_dim)"""
    np.random.seed(42)
    return np.random.rand(10, 74).astype(np.float32)


@pytest.fixture
def sample_y() -> np.ndarray:
    """타겟 데이터 (batch, 45) multi-hot"""
    np.random.seed(42)
    y = np.zeros((10, 45), dtype=np.float32)
    for i in range(10):
        indices = np.random.choice(45, 6, replace=False)
        y[i, indices] = 1.0
    return y


@pytest.fixture
def sample_single_X() -> np.ndarray:
    """단일 예측용 입력 데이터"""
    np.random.seed(42)
    return np.random.rand(1, 20, 74).astype(np.float32)


# ============================================================
# Model Mock Fixture
# ============================================================

class MockModel(BaseModel):
    """테스트용 Mock 모델"""

    def __init__(self, config=None):
        super().__init__(config or {})
        self.model_type = "mock"
        self._requires_sequence = True

    def train(self, X, y, validation_data=None, epochs=100, **kwargs):
        from core.types import TrainingHistory
        self.is_trained = True
        history = TrainingHistory(epochs=epochs)
        for i in range(epochs):
            history.add_epoch(train_loss=1.0 / (i + 1))
        return history

    def predict_proba(self, X):
        np.random.seed(42)
        probs = np.random.rand(45)
        probs = probs / probs.sum()
        return ProbabilityDistribution(probs, self.model_type)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        self.is_trained = True

    @property
    def requires_sequence(self) -> bool:
        return self._requires_sequence


@pytest.fixture
def mock_model() -> MockModel:
    """Mock 모델 fixture"""
    return MockModel()


@pytest.fixture
def trained_mock_model(mock_model, sample_X_sequence, sample_y) -> MockModel:
    """학습된 Mock 모델 fixture"""
    mock_model.train(sample_X_sequence, sample_y, epochs=10)
    return mock_model


# ============================================================
# 필터 테스트용 Fixtures
# ============================================================

@pytest.fixture
def valid_pattern_numbers() -> List[int]:
    """패턴 필터 통과하는 번호"""
    # 홀짝 3:3, 합계 약 130, AC값 양호
    return [5, 12, 23, 28, 37, 44]


@pytest.fixture
def invalid_pattern_numbers() -> List[int]:
    """패턴 필터 통과 못하는 번호 (올 짝수)"""
    return [2, 4, 6, 8, 10, 12]


# ============================================================
# 임시 파일 경로 Fixture
# ============================================================

@pytest.fixture
def temp_model_path(tmp_path):
    """임시 모델 저장 경로"""
    return str(tmp_path / "test_model.pt")


@pytest.fixture
def temp_pkl_path(tmp_path):
    """임시 pickle 저장 경로"""
    return str(tmp_path / "test_model.pkl")
