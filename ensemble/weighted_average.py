"""가중 평균 앙상블"""
from typing import List
import numpy as np

from core.base_ensemble import BaseEnsemble
from core.types import ProbabilityDistribution


class WeightedAverageEnsemble(BaseEnsemble):
    """
    가중 평균 앙상블

    각 모델의 확률 분포를 가중 평균하여 최종 분포 생성
    """

    def __init__(self, models=None):
        super().__init__(models)
        self.name = "weighted_avg_ensemble"

    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """확률 분포의 가중 평균"""
        weighted_sum = np.zeros(45)
        total_weight = 0

        for pred in predictions:
            weight = self.weights.get(pred.model_name, 1.0)
            weighted_sum += pred.probabilities * weight
            total_weight += weight

        if total_weight > 0:
            probs = weighted_sum / total_weight
        else:
            probs = np.ones(45) / 45

        return ProbabilityDistribution(probs, self.name)


class SoftmaxAverageEnsemble(BaseEnsemble):
    """
    Softmax 가중 평균 앙상블

    각 모델의 성능 지표를 softmax로 변환하여 가중치로 사용
    """

    def __init__(self, models=None, temperature: float = 1.0):
        super().__init__(models)
        self.name = "softmax_avg_ensemble"
        self.temperature = temperature
        self.performance_scores = {}  # 모델별 성능 점수

    def set_performance(self, model_name: str, score: float):
        """모델 성능 점수 설정 (높을수록 좋음)"""
        self.performance_scores[model_name] = score

    def _compute_softmax_weights(self) -> dict:
        """성능 점수를 softmax 가중치로 변환"""
        if not self.performance_scores:
            return {m.model_type: 1.0 for m in self.models}

        scores = np.array([
            self.performance_scores.get(m.model_type, 0.0)
            for m in self.models
        ])

        # Softmax with temperature
        exp_scores = np.exp(scores / self.temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        return {
            m.model_type: float(w)
            for m, w in zip(self.models, softmax_weights)
        }

    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """Softmax 가중 평균"""
        softmax_weights = self._compute_softmax_weights()

        weighted_sum = np.zeros(45)
        total_weight = 0

        for pred in predictions:
            weight = softmax_weights.get(pred.model_name, 1.0)
            weighted_sum += pred.probabilities * weight
            total_weight += weight

        if total_weight > 0:
            probs = weighted_sum / total_weight
        else:
            probs = np.ones(45) / 45

        return ProbabilityDistribution(probs, self.name)
