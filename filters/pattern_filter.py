"""패턴 기반 필터 (홀짝, 고저, 연번 등)"""
from typing import List, Tuple
import numpy as np

from core.base_filter import BaseFilter
from core.types import Prediction


class PatternFilter(BaseFilter):
    """
    패턴 기반 필터

    통계적으로 자주 나오는 패턴만 허용:
    - 홀짝 비율: 2:4, 3:3, 4:2
    - 고저 비율: 2:4, 3:3, 4:2
    - 합계 범위: 100~175
    - 연속번호: 최대 2쌍
    """

    def __init__(self,
                 odd_range: Tuple[int, int] = (2, 4),
                 low_range: Tuple[int, int] = (2, 4),
                 sum_range: Tuple[int, int] = (100, 175),
                 max_consecutive: int = 2,
                 enabled: bool = True):
        """
        Args:
            odd_range: 허용 홀수 개수 범위 (최소, 최대)
            low_range: 허용 저번호(1-22) 개수 범위
            sum_range: 허용 합계 범위
            max_consecutive: 최대 연속번호 쌍 수
        """
        super().__init__(enabled)
        self.name = "pattern"
        self.odd_range = odd_range
        self.low_range = low_range
        self.sum_range = sum_range
        self.max_consecutive = max_consecutive

    def _count_odd(self, numbers: List[int]) -> int:
        """홀수 개수"""
        return sum(1 for n in numbers if n % 2 == 1)

    def _count_low(self, numbers: List[int]) -> int:
        """저번호(1-22) 개수"""
        return sum(1 for n in numbers if n <= 22)

    def _count_consecutive(self, numbers: List[int]) -> int:
        """연속번호 쌍 개수"""
        sorted_nums = sorted(numbers)
        return sum(1 for i in range(5) if sorted_nums[i+1] - sorted_nums[i] == 1)

    def score(self, numbers: List[int]) -> float:
        """패턴 점수 (0.0 ~ 1.0)"""
        scores = []

        # 홀짝 점수
        odd_count = self._count_odd(numbers)
        if self.odd_range[0] <= odd_count <= self.odd_range[1]:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # 고저 점수
        low_count = self._count_low(numbers)
        if self.low_range[0] <= low_count <= self.low_range[1]:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # 합계 점수
        total = sum(numbers)
        if self.sum_range[0] <= total <= self.sum_range[1]:
            scores.append(1.0)
        else:
            # 범위에서 멀수록 점수 낮음
            diff = min(abs(total - self.sum_range[0]), abs(total - self.sum_range[1]))
            scores.append(max(0.0, 1.0 - diff / 50))

        # 연번 점수
        consecutive = self._count_consecutive(numbers)
        if consecutive <= self.max_consecutive:
            scores.append(1.0)
        else:
            scores.append(0.3)

        return float(np.mean(scores))

    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """점수가 일정 이상인 예측만 통과"""
        threshold = 0.7
        filtered = [p for p in predictions if self.score(p.numbers) >= threshold]
        return filtered if filtered else predictions[:1]

    def analyze(self, numbers: List[int]) -> dict:
        """번호 조합 분석"""
        return {
            'odd_count': self._count_odd(numbers),
            'even_count': 6 - self._count_odd(numbers),
            'low_count': self._count_low(numbers),
            'high_count': 6 - self._count_low(numbers),
            'sum': sum(numbers),
            'consecutive_pairs': self._count_consecutive(numbers),
            'score': self.score(numbers)
        }
