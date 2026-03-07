"""MMR 기반 후보 선택 - 대량 생성 → 스코어링 → 다양성 리랭킹"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .types import Prediction, ProbabilityDistribution
from .base_filter import BaseFilter
from .utils import sample_numbers
from .constants import LOTTO_NUMBERS_COUNT, LOTTO_POOL_SIZE


@dataclass
class SelectionPreset:
    """추출 모드 프리셋"""
    temperature: float       # 샘플링 온도
    num_candidates: int      # 후보 생성 수
    diversity_lambda: float  # 1.0=품질만, 0.0=다양성만
    filter_weight: float     # 필터 점수 가중치


PRESETS = {
    'safe': SelectionPreset(
        temperature=0.7, num_candidates=500,
        diversity_lambda=0.4, filter_weight=1.5,
    ),
    'balanced': SelectionPreset(
        temperature=1.0, num_candidates=1000,
        diversity_lambda=0.6, filter_weight=1.0,
    ),
    'aggressive': SelectionPreset(
        temperature=1.5, num_candidates=1500,
        diversity_lambda=0.85, filter_weight=0.5,
    ),
}


class CandidateSelector:
    """
    후보 대량 생성 → 확률+필터 스코어링 → MMR 다양성 선택

    기존 독립 샘플링 대비 장점:
    1. 세트 간 중복 최소화 (MMR 다양성 항)
    2. 필터를 pass/fail이 아닌 연속 점수로 활용
    3. safe/balanced/aggressive 프리셋으로 전략 전환
    """

    def __init__(self,
                 filters: Optional[List[BaseFilter]] = None,
                 preset: str = 'balanced'):
        self.filters = filters or []
        if isinstance(preset, str):
            self.preset = PRESETS.get(preset, PRESETS['balanced'])
            self._preset_name = preset
        else:
            self.preset = preset
            self._preset_name = 'custom'

    def select(self,
               proba: ProbabilityDistribution,
               num_sets: int = 5,
               model_name: str = "ensemble") -> List[Prediction]:
        """
        후보 생성 → 스코어링 → MMR 선택

        Args:
            proba: 45개 번호 확률 분포
            num_sets: 최종 선택 세트 수
            model_name: 모델 이름 (출력용)

        Returns:
            다양하고 품질 높은 예측 리스트
        """
        p = self.preset

        # 1. 후보 대량 생성
        candidates, seen = self._generate_candidates(proba.probabilities, p)

        # 후보 부족 시 epsilon smoothing으로 보충
        if len(candidates) < num_sets:
            smoothed = self._epsilon_smooth(proba.probabilities)
            extra, _ = self._generate_candidates(smoothed, p, exclude=seen)
            candidates.extend(extra)

        if not candidates:
            top6 = sorted(proba.top_k(6))
            return [Prediction(numbers=top6, confidence=0.0, model_name=model_name)]

        # 2. 스코어링: log(확률) + 필터 점수 (원본 확률 기준)
        prob_scores = np.array([self._prob_score(nums, proba.probabilities) for nums in candidates])
        filter_scores = np.array([self._filter_score(nums) for nums in candidates])

        prob_norm = self._normalize(prob_scores)
        combined = prob_norm + p.filter_weight * filter_scores

        # 3. MMR 다양성 선택
        selected_indices = self._mmr_select(candidates, combined, num_sets, p.diversity_lambda)

        # 4. Prediction 객체 생성
        predictions = []
        for idx in selected_indices:
            nums = candidates[idx]
            confidence = float(np.mean([proba.probabilities[n - 1] for n in nums]))
            predictions.append(Prediction(
                numbers=nums,
                confidence=confidence,
                model_name=model_name,
                metadata={
                    'score': float(combined[idx]),
                    'prob_score': float(prob_scores[idx]),
                    'filter_score': float(filter_scores[idx]),
                    'mode': self._preset_name,
                }
            ))

        return predictions

    @staticmethod
    def _generate_candidates(probs: np.ndarray, preset: SelectionPreset,
                             exclude: set = None) -> tuple:
        """후보 세트 대량 생성 (중복 제거)"""
        candidates = []
        seen = exclude.copy() if exclude else set()
        attempts = 0
        max_attempts = preset.num_candidates * 3

        while len(candidates) < preset.num_candidates and attempts < max_attempts:
            nums = sorted(sample_numbers(probs, LOTTO_NUMBERS_COUNT, preset.temperature))
            key = tuple(nums)
            if key not in seen:
                seen.add(key)
                candidates.append(nums)
            attempts += 1

        return candidates, seen

    @staticmethod
    def _epsilon_smooth(probs: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """확률 분포에 epsilon smoothing 적용 (쏠림 완화)"""
        smoothed = probs * (1 - epsilon) + epsilon / LOTTO_POOL_SIZE
        return smoothed

    def _prob_score(self, numbers: List[int], probs: np.ndarray) -> float:
        """6개 번호의 joint log probability"""
        return float(sum(np.log(max(probs[n - 1], 1e-10)) for n in numbers))

    def _filter_score(self, numbers: List[int]) -> float:
        """필터 점수 평균 (0~1)"""
        if not self.filters:
            return 1.0
        scores = [f.score(numbers) for f in self.filters if f.enabled]
        return float(np.mean(scores)) if scores else 1.0

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """배열을 [0, 1]로 정규화"""
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _mmr_select(candidates: List[List[int]],
                    scores: np.ndarray,
                    k: int,
                    diversity_lambda: float) -> List[int]:
        """
        Maximal Marginal Relevance 선택

        mmr(i) = λ * score(i) - (1-λ) * max_sim(i, selected)
        sim = |교집합| / 6 (overlap ratio)
        """
        n = len(candidates)
        if n <= k:
            return list(range(n))

        norm_scores = CandidateSelector._normalize(scores)
        candidate_sets = [frozenset(c) for c in candidates]

        selected = []
        remaining = set(range(n))

        # 첫 번째: 최고 점수
        first = int(np.argmax(norm_scores))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            best_idx = -1
            best_mmr = -float('inf')

            for idx in remaining:
                max_sim = max(
                    len(candidate_sets[idx] & candidate_sets[s]) / 6.0
                    for s in selected
                )
                mmr = diversity_lambda * norm_scores[idx] - (1 - diversity_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected
