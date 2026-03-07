"""다수결 투표 앙상블"""
from typing import List
import numpy as np

from core.base_ensemble import BaseEnsemble
from core.types import ProbabilityDistribution


class VotingEnsemble(BaseEnsemble):
    """
    다수결 투표 앙상블

    각 모델이 예측한 Top-K 번호에 투표하여 최종 확률 분포 생성
    """

    def __init__(self, models=None, top_k: int = 10):
        """
        Args:
            models: 앙상블에 포함할 모델 리스트
            top_k: 각 모델에서 투표할 상위 번호 개수
        """
        super().__init__(models)
        self.name = "voting_ensemble"
        self.top_k = top_k

    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """
        Top-K 투표 방식으로 예측 결합

        각 모델의 Top-K 번호에 가중치를 곱해 투표
        """
        vote_counts = np.zeros(45)

        for pred in predictions:
            # 상위 top_k개 번호
            top_indices = np.argsort(pred.probabilities)[-self.top_k:][::-1]

            for rank, idx in enumerate(top_indices):
                # 순위에 따른 가중치 (1위가 가장 높음)
                rank_weight = 1.0 - (rank / self.top_k) * 0.5

                # 모델 가중치
                model_weight = self.weights.get(pred.model_name, 1.0)

                vote_counts[idx] += rank_weight * model_weight

        # 정규화
        if vote_counts.sum() > 0:
            probs = vote_counts / vote_counts.sum()
        else:
            probs = np.ones(45) / 45

        return ProbabilityDistribution(probs, self.name)


class HardVotingEnsemble(BaseEnsemble):
    """
    하드 투표 앙상블

    각 모델의 Top-6 예측에 대해 단순 다수결 투표
    """

    def __init__(self, models=None):
        super().__init__(models)
        self.name = "hard_voting_ensemble"

    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """각 모델의 Top-6에 동일 가중치로 투표"""
        vote_counts = np.zeros(45)

        for pred in predictions:
            top6 = pred.top_k(6)
            model_weight = self.weights.get(pred.model_name, 1.0)

            for num in top6:
                vote_counts[num - 1] += model_weight

        # 정규화
        if vote_counts.sum() > 0:
            probs = vote_counts / vote_counts.sum()
        else:
            probs = np.ones(45) / 45

        return ProbabilityDistribution(probs, self.name)
