"""마르코프 체인 기반 예측 모델"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pickle

from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory


class MarkovChainModel(BaseModel):
    """
    마르코프 체인 기반 로또 예측 모델

    이전 회차에 특정 번호가 나왔을 때,
    다음 회차에 각 번호가 나올 조건부 확률을 계산
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "markov"

        # 마르코프 차수 (1차: 직전 회차만, 2차: 직전 2회차)
        self.order = self.config.get('order', 1)

        # 스무딩 파라미터 (0 확률 방지)
        self.smoothing = self.config.get('smoothing', 0.01)

        # 전이 행렬: transition[i, j] = P(j번호 당첨 | i번호가 이전 회차에 당첨)
        self.transition_matrix: Optional[np.ndarray] = None

        # 사전 확률 (각 번호의 전체 출현 빈도)
        self.prior: Optional[np.ndarray] = None

        # 공출현 행렬: 같은 회차에 함께 나온 번호 쌍
        self.co_occurrence: Optional[np.ndarray] = None

    def _extract_multihot(self, data: np.ndarray) -> np.ndarray:
        """확장 피처에서 multi-hot 부분(처음 45차원)만 추출"""
        if data.shape[-1] > 45:
            return data[..., :45]
        return data

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 1,
              verbose: bool = True,
              **kwargs) -> TrainingHistory:
        """
        전이 확률 학습

        Args:
            X: 시퀀스 데이터 (batch, seq_len, 45+) - multi-hot (확장 피처 포함 가능)
            y: 타겟 (batch, 45+) - multi-hot (확장 피처 포함 가능)
        """
        history = TrainingHistory(epochs=1)

        # 전이 행렬 초기화 (45 x 45)
        self.transition_matrix = np.zeros((45, 45)) + self.smoothing
        self.co_occurrence = np.zeros((45, 45)) + self.smoothing
        self.prior = np.zeros(45) + self.smoothing

        # 확장 피처 사용 시 multi-hot만 추출
        X = self._extract_multihot(X)
        y = self._extract_multihot(y)

        total_samples = len(X)

        for idx in range(total_samples):
            # 이전 회차들에서 나온 번호
            prev_seq = X[idx]  # (seq_len, 45)
            # 마지막 회차
            last_round = prev_seq[-1]
            prev_numbers = np.where(last_round > 0.5)[0]

            # 현재 회차 당첨 번호
            curr_numbers = np.where(y[idx] > 0.5)[0]

            # 사전 확률 업데이트
            for num in curr_numbers:
                self.prior[num] += 1

            # 전이 확률 업데이트
            for prev_num in prev_numbers:
                for curr_num in curr_numbers:
                    self.transition_matrix[prev_num, curr_num] += 1

            # 공출현 행렬 업데이트
            for i, num1 in enumerate(curr_numbers):
                for num2 in curr_numbers[i+1:]:
                    self.co_occurrence[num1, num2] += 1
                    self.co_occurrence[num2, num1] += 1

        # 정규화
        # 전이 행렬: 각 행의 합이 1이 되도록
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = self.transition_matrix / row_sums

        # 사전 확률 정규화
        self.prior = self.prior / self.prior.sum()

        # 공출현 행렬 정규화
        row_sums = self.co_occurrence.sum(axis=1, keepdims=True)
        self.co_occurrence = self.co_occurrence / row_sums

        if verbose:
            print(f"마르코프 체인 학습 완료 (샘플: {total_samples})")

        self.is_trained = True
        return history

    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        """
        확률 분포 예측

        Args:
            X: 입력 시퀀스 (1, seq_len, 45+) - 확장 피처 포함 가능
        """
        if self.transition_matrix is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 확장 피처 사용 시 multi-hot만 추출
        X = self._extract_multihot(X)

        # 마지막 회차 번호 추출
        last_round = X[0, -1]  # (45,)
        prev_numbers = np.where(last_round > 0.5)[0]

        # 전이 확률 기반 예측
        probs = np.zeros(45)

        if len(prev_numbers) > 0:
            for prev_num in prev_numbers:
                probs += self.transition_matrix[prev_num]
            probs /= len(prev_numbers)
        else:
            probs = self.prior.copy()

        # 공출현 확률 결합 (가중 평균)
        co_probs = np.zeros(45)
        if len(prev_numbers) > 0:
            for prev_num in prev_numbers:
                co_probs += self.co_occurrence[prev_num]
            co_probs /= len(prev_numbers)

        # 전이 확률 + 공출현 확률 + 사전 확률 결합
        alpha = self.config.get('transition_weight', 0.5)
        beta = self.config.get('co_occurrence_weight', 0.3)
        gamma = 1 - alpha - beta

        final_probs = alpha * probs + beta * co_probs + gamma * self.prior

        # 정규화
        final_probs = final_probs / final_probs.sum()

        return ProbabilityDistribution(final_probs, self.model_type)

    def get_transition_stats(self) -> Dict[str, Any]:
        """전이 행렬 통계"""
        if self.transition_matrix is None:
            return {}

        return {
            'mean_prob': self.transition_matrix.mean(),
            'max_prob': self.transition_matrix.max(),
            'min_prob': self.transition_matrix.min(),
            'top_transitions': self._get_top_transitions(10)
        }

    def _get_top_transitions(self, n: int) -> List[Tuple[int, int, float]]:
        """가장 높은 전이 확률 쌍"""
        flat_idx = np.argsort(self.transition_matrix.flatten())[-n:][::-1]
        result = []
        for idx in flat_idx:
            i, j = divmod(idx, 45)
            result.append((i + 1, j + 1, self.transition_matrix[i, j]))
        return result

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'transition_matrix': self.transition_matrix,
                'co_occurrence': self.co_occurrence,
                'prior': self.prior,
                'config': self.config,
                'model_type': self.model_type,
            }, f)

    def load(self, path: str) -> None:
        """모델 로드

        Raises:
            FileNotFoundError: 파일이 없을 때
            RuntimeError: 로드 실패 시
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.transition_matrix = data['transition_matrix']
                self.co_occurrence = data['co_occurrence']
                self.prior = data['prior']
                self.config = data.get('config', self.config)
            self.is_trained = True
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}") from e

    @property
    def requires_sequence(self) -> bool:
        return True  # 전이 계산에 시퀀스 필요
