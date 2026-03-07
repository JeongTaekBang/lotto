"""모든 예측 모델의 베이스 클래스"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .types import Prediction, ProbabilityDistribution, TrainingHistory
from .utils import sample_numbers
from .constants import LOTTO_NUMBERS_COUNT


class BaseModel(ABC):
    """모든 예측 모델의 베이스 클래스"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_trained = False
        self.model_type: str = "base"
        self._device = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 100,
              **kwargs) -> TrainingHistory:
        """
        모델 학습

        Args:
            X: 입력 시퀀스 (batch, seq_length, feature_dim) 또는 (batch, feature_dim)
            y: 타겟 (batch, 45) multi-hot
            validation_data: 검증 데이터 (X_val, y_val)
            epochs: 학습 에포크 수

        Returns:
            학습 이력
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        """
        확률 분포 예측

        Args:
            X: 입력 시퀀스 (1, seq_length, feature_dim) 또는 (1, feature_dim)

        Returns:
            ProbabilityDistribution: 45개 번호의 확률 분포
        """
        pass

    def predict_numbers(self, X: np.ndarray, num_sets: int = 5,
                       temperature: float = 1.0) -> List[Prediction]:
        """
        번호 세트 예측 (기본 구현 제공)

        Args:
            X: 입력 시퀀스
            num_sets: 생성할 번호 세트 수
            temperature: 샘플링 온도 (높을수록 다양성 증가)

        Returns:
            예측 결과 리스트
        """
        probs = self.predict_proba(X)
        predictions = []

        for _ in range(num_sets):
            selected = sample_numbers(probs.probabilities, LOTTO_NUMBERS_COUNT, temperature)
            confidence = float(np.mean([probs.probabilities[n-1] for n in selected]))
            predictions.append(Prediction(
                numbers=sorted(selected),
                confidence=confidence,
                model_name=self.model_type
            ))

        return predictions

    def predict_top_k(self, X: np.ndarray, k: int = 6) -> List[int]:
        """상위 k개 번호 반환 (결정적)"""
        probs = self.predict_proba(X)
        return probs.top_k(k)

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 저장"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """모델 로드"""
        pass

    @property
    @abstractmethod
    def requires_sequence(self) -> bool:
        """
        시퀀스 입력 필요 여부
        - True: (batch, seq_length, feature_dim) 형태 필요 (LSTM, Transformer)
        - False: (batch, feature_dim) 형태 사용 (XGBoost, RandomForest)
        """
        pass

    @property
    def name(self) -> str:
        """모델 이름"""
        return self.model_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.model_type}, trained={self.is_trained})"
