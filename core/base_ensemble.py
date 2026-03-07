"""앙상블 전략 추상 클래스"""
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

from .base_model import BaseModel
from .types import Prediction, ProbabilityDistribution
from .utils import sample_numbers
from .constants import LOTTO_NUMBERS_COUNT


class BaseEnsemble(ABC):
    """앙상블 전략의 베이스 클래스"""

    def __init__(self, models: List[BaseModel] = None):
        self.models: List[BaseModel] = models or []
        self.weights: Dict[str, float] = {}
        self.name: str = "base_ensemble"

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        모델 추가

        Args:
            model: 추가할 모델
            weight: 모델 가중치 (기본 1.0)
        """
        self.models.append(model)
        self.weights[model.model_type] = weight

    def remove_model(self, model_name: str) -> bool:
        """
        모델 제거

        Args:
            model_name: 제거할 모델 이름

        Returns:
            제거 성공 여부
        """
        for i, model in enumerate(self.models):
            if model.model_type == model_name:
                self.models.pop(i)
                self.weights.pop(model_name, None)
                return True
        return False

    def set_weight(self, model_name: str, weight: float) -> None:
        """모델 가중치 설정"""
        self.weights[model_name] = weight

    @abstractmethod
    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """
        여러 모델의 예측을 결합

        Args:
            predictions: 각 모델의 확률 분포 리스트

        Returns:
            결합된 확률 분포
        """
        pass

    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        """
        앙상블 확률 예측

        Args:
            X: 입력 데이터

        Returns:
            결합된 확률 분포
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        model_predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            model_predictions.append(pred)

        return self.combine_predictions(model_predictions)

    def predict_numbers(self, X: np.ndarray, num_sets: int = 5,
                       temperature: float = 1.0) -> List[Prediction]:
        """
        앙상블 번호 예측

        Args:
            X: 입력 데이터
            num_sets: 생성할 세트 수
            temperature: 샘플링 온도

        Returns:
            예측 결과 리스트
        """
        combined_proba = self.predict_proba(X)
        predictions = []

        for _ in range(num_sets):
            selected = sample_numbers(combined_proba.probabilities, LOTTO_NUMBERS_COUNT, temperature)
            confidence = float(np.mean([combined_proba.probabilities[n-1] for n in selected]))
            predictions.append(Prediction(
                numbers=sorted(selected),
                confidence=confidence,
                model_name=self.name,
                metadata={'ensemble_type': self.__class__.__name__}
            ))

        return predictions

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        model_names = [m.model_type for m in self.models]
        return f"{self.__class__.__name__}(models={model_names})"
