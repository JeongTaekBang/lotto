"""복합 필터 - 여러 필터 조합"""
from typing import List, Dict, Optional
import numpy as np

from core.base_filter import BaseFilter
from core.types import Prediction


class CompositeFilter(BaseFilter):
    """
    복합 필터 - 여러 필터를 순차적으로 적용

    모든 필터를 통과해야 최종 결과에 포함
    """

    def __init__(self, filters: List[BaseFilter] = None, enabled: bool = True):
        super().__init__(enabled)
        self.name = "composite"
        self.filters: List[BaseFilter] = filters or []

    def add_filter(self, filter_: BaseFilter):
        """필터 추가"""
        self.filters.append(filter_)

    def remove_filter(self, filter_name: str) -> bool:
        """필터 제거"""
        for i, f in enumerate(self.filters):
            if f.name == filter_name:
                self.filters.pop(i)
                return True
        return False

    def score(self, numbers: List[int]) -> float:
        """모든 필터 점수의 가중 평균"""
        if not self.filters:
            return 1.0

        scores = [f.score(numbers) for f in self.filters if f.enabled]
        return float(np.mean(scores)) if scores else 1.0

    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """순차적 필터링"""
        result = predictions

        for f in self.filters:
            if f.enabled:
                result = f.filter(result)
                if not result:
                    # 모든 예측이 필터링되면 마지막 결과 반환
                    break

        return result if result else predictions[:1]

    def __len__(self) -> int:
        return len(self.filters)


class WeightedCompositeFilter(BaseFilter):
    """
    가중 복합 필터 - 점수 기반 필터링

    각 필터의 점수를 가중 합산하여 임계값 이상인 예측만 통과
    """

    def __init__(self,
                 filters: List[BaseFilter] = None,
                 weights: Dict[str, float] = None,
                 threshold: float = 0.7,
                 enabled: bool = True):
        super().__init__(enabled)
        self.name = "weighted_composite"
        self.filters: List[BaseFilter] = filters or []
        self.weights: Dict[str, float] = weights or {}
        self.threshold = threshold

    def add_filter(self, filter_: BaseFilter, weight: float = 1.0):
        """필터 추가 (가중치 포함)"""
        self.filters.append(filter_)
        self.weights[filter_.name] = weight

    def score(self, numbers: List[int]) -> float:
        """가중 점수 계산"""
        if not self.filters:
            return 1.0

        total_score = 0.0
        total_weight = 0.0

        for f in self.filters:
            if f.enabled:
                weight = self.weights.get(f.name, 1.0)
                total_score += f.score(numbers) * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 1.0

    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """임계값 이상인 예측만 통과"""
        filtered = [p for p in predictions if self.score(p.numbers) >= self.threshold]
        return filtered if filtered else predictions[:1]


class ScoreRankingFilter(BaseFilter):
    """
    점수 순위 필터 - 상위 N개만 선택

    모든 예측의 종합 점수를 계산하고 상위 N개만 반환
    """

    def __init__(self,
                 filters: List[BaseFilter] = None,
                 top_n: int = 10,
                 enabled: bool = True):
        super().__init__(enabled)
        self.name = "score_ranking"
        self.filters: List[BaseFilter] = filters or []
        self.top_n = top_n

    def add_filter(self, filter_: BaseFilter):
        """필터 추가"""
        self.filters.append(filter_)

    def score(self, numbers: List[int]) -> float:
        """종합 점수 계산"""
        if not self.filters:
            return 1.0

        scores = [f.score(numbers) for f in self.filters if f.enabled]
        return float(np.mean(scores)) if scores else 1.0

    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """점수 상위 N개 선택"""
        if not predictions:
            return []

        # 점수 계산 및 정렬
        scored_predictions = [
            (pred, self.score(pred.numbers))
            for pred in predictions
        ]
        scored_predictions.sort(key=lambda x: x[1], reverse=True)

        # 상위 N개 선택
        top_predictions = [pred for pred, _ in scored_predictions[:self.top_n]]

        return top_predictions if top_predictions else predictions[:1]
