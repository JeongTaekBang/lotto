"""규칙 기반 필터 추상 클래스"""
from abc import ABC, abstractmethod
from typing import List

from .types import Prediction


class BaseFilter(ABC):
    """규칙 기반 필터의 베이스 클래스"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.name: str = "base"

    @abstractmethod
    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        예측 결과 필터링

        Args:
            predictions: 필터링할 예측 리스트

        Returns:
            필터링된 예측 리스트
        """
        pass

    @abstractmethod
    def score(self, numbers: List[int]) -> float:
        """
        번호 조합 점수 (0.0 ~ 1.0)
        높을수록 좋은 조합

        Args:
            numbers: 6개 번호 리스트

        Returns:
            점수 (0.0 ~ 1.0)
        """
        pass

    def __call__(self, predictions: List[Prediction]) -> List[Prediction]:
        """필터 적용 (enabled 체크 포함)"""
        if not self.enabled:
            return predictions
        return self.filter(predictions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"
