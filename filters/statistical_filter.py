"""통계 기반 필터 (AC값, 끝수 분포 등)"""
from typing import List, Set
import numpy as np

from core.base_filter import BaseFilter
from core.types import Prediction
from core.constants import PRIMES


class StatisticalFilter(BaseFilter):
    """
    통계 기반 필터

    - AC값: 번호 분포 복잡도 (최소 5 이상 권장)
    - 끝수 분포: 10개 끝수(0-9) 중 최소 3개 이상
    - 소수 개수: 1~4개 권장
    """

    def __init__(self,
                 min_ac: int = 5,
                 min_endings: int = 3,
                 prime_range: tuple = (1, 4),
                 enabled: bool = True):
        """
        Args:
            min_ac: 최소 AC값
            min_endings: 최소 끝수 종류 수
            prime_range: 허용 소수 개수 범위
        """
        super().__init__(enabled)
        self.name = "statistical"
        self.min_ac = min_ac
        self.min_endings = min_endings
        self.prime_range = prime_range

    def _calculate_ac(self, numbers: List[int]) -> int:
        """
        AC값 (Arithmetic Complexity) 계산
        번호 간 차이의 고유 개수 - 5
        """
        sorted_nums = sorted(numbers)
        differences = set()

        for i in range(6):
            for j in range(i + 1, 6):
                differences.add(sorted_nums[j] - sorted_nums[i])

        return len(differences) - 5

    def _count_endings(self, numbers: List[int]) -> int:
        """끝수(1의 자리) 종류 개수"""
        endings = set(n % 10 for n in numbers)
        return len(endings)

    def _count_primes(self, numbers: List[int]) -> int:
        """소수 개수"""
        return sum(1 for n in numbers if n in PRIMES)

    def score(self, numbers: List[int]) -> float:
        """통계 기반 점수 (0.0 ~ 1.0)"""
        scores = []

        # AC값 점수
        ac = self._calculate_ac(numbers)
        if ac >= self.min_ac:
            scores.append(1.0)
        else:
            scores.append(ac / self.min_ac)

        # 끝수 점수
        endings = self._count_endings(numbers)
        if endings >= self.min_endings:
            scores.append(1.0)
        else:
            scores.append(endings / self.min_endings)

        # 소수 점수
        primes = self._count_primes(numbers)
        if self.prime_range[0] <= primes <= self.prime_range[1]:
            scores.append(1.0)
        else:
            scores.append(0.5)

        return float(np.mean(scores))

    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """통계 조건을 만족하는 예측만 통과"""
        filtered = []

        for pred in predictions:
            ac = self._calculate_ac(pred.numbers)
            endings = self._count_endings(pred.numbers)
            primes = self._count_primes(pred.numbers)

            if (ac >= self.min_ac and
                endings >= self.min_endings and
                self.prime_range[0] <= primes <= self.prime_range[1]):
                filtered.append(pred)

        return filtered if filtered else predictions[:1]

    def analyze(self, numbers: List[int]) -> dict:
        """번호 조합 통계 분석"""
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(5)]

        return {
            'ac_value': self._calculate_ac(numbers),
            'unique_endings': self._count_endings(numbers),
            'prime_count': self._count_primes(numbers),
            'gaps': gaps,
            'max_gap': max(gaps),
            'min_gap': min(gaps),
            'score': self.score(numbers)
        }
