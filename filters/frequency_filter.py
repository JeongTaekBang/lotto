"""빈도 기반 필터"""
from typing import List, Dict, Optional
from collections import Counter
import numpy as np

from core.base_filter import BaseFilter
from core.base_datasource import BaseDataSource
from core.types import Prediction


class FrequencyFilter(BaseFilter):
    """
    빈도 기반 필터

    - 최근 N회차 출현 빈도를 기준으로 필터링
    - Cold 번호(너무 안 나온)가 많은 조합 제외
    - Hot 번호(너무 많이 나온)만 있는 조합 제외
    """

    def __init__(self,
                 datasource: Optional[BaseDataSource] = None,
                 recent_n: int = 100,
                 cold_threshold: float = 0.7,
                 hot_threshold: float = 1.5,
                 max_cold_numbers: int = 2,
                 max_hot_numbers: int = 4,
                 enabled: bool = True):
        """
        Args:
            datasource: 데이터 소스 (빈도 계산용)
            recent_n: 최근 N회차 기준
            cold_threshold: 기대값 대비 이 비율 미만이면 cold
            hot_threshold: 기대값 대비 이 비율 초과면 hot
            max_cold_numbers: 최대 허용 cold 번호 개수
            max_hot_numbers: 최대 허용 hot 번호 개수
        """
        super().__init__(enabled)
        self.name = "frequency"
        self.datasource = datasource
        self.recent_n = recent_n
        self.cold_threshold = cold_threshold
        self.hot_threshold = hot_threshold
        self.max_cold_numbers = max_cold_numbers
        self.max_hot_numbers = max_hot_numbers
        self._frequencies: Optional[Dict[int, float]] = None

    def set_datasource(self, datasource: BaseDataSource):
        """데이터 소스 설정"""
        self.datasource = datasource
        self._frequencies = None  # 캐시 무효화

    def _calculate_frequencies(self) -> Dict[int, float]:
        """최근 N회차 빈도 계산"""
        if self._frequencies is not None:
            return self._frequencies

        if self.datasource is None:
            # 데이터 소스 없으면 균등 분포 가정
            self._frequencies = {num: 1.0 for num in range(1, 46)}
            return self._frequencies

        records = self.datasource.get_latest(self.recent_n)
        counts = Counter()

        for rec in records:
            counts.update(rec.numbers)

        # 기대값 = recent_n * 6 / 45
        expected = self.recent_n * 6 / 45

        self._frequencies = {
            num: counts.get(num, 0) / expected
            for num in range(1, 46)
        }

        return self._frequencies

    def score(self, numbers: List[int]) -> float:
        """빈도 기반 점수 (0.0 ~ 1.0)"""
        freqs = self._calculate_frequencies()

        # 평균 빈도
        avg_freq = np.mean([freqs[n] for n in numbers])

        # 너무 낮거나 높으면 점수 감소
        if avg_freq < self.cold_threshold:
            return 0.3
        elif avg_freq > self.hot_threshold:
            return 0.7
        else:
            return 1.0

    def filter(self, predictions: List[Prediction]) -> List[Prediction]:
        """cold/hot 번호 기준 필터링"""
        freqs = self._calculate_frequencies()

        filtered = []
        for pred in predictions:
            cold_count = sum(1 for n in pred.numbers if freqs[n] < self.cold_threshold)
            hot_count = sum(1 for n in pred.numbers if freqs[n] > self.hot_threshold)

            if cold_count <= self.max_cold_numbers and hot_count <= self.max_hot_numbers:
                filtered.append(pred)

        return filtered if filtered else predictions[:1]

    def get_cold_numbers(self, top_n: int = 10) -> List[int]:
        """가장 안 나온 번호들"""
        freqs = self._calculate_frequencies()
        sorted_nums = sorted(freqs.items(), key=lambda x: x[1])
        return [num for num, _ in sorted_nums[:top_n]]

    def get_hot_numbers(self, top_n: int = 10) -> List[int]:
        """가장 많이 나온 번호들"""
        freqs = self._calculate_frequencies()
        sorted_nums = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:top_n]]
