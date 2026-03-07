"""피처 엔지니어링 모듈"""
from typing import List, Optional
import numpy as np

from .constants import (
    PRIMES,
    DEFAULT_PERIOD,
    FEATURE_DIM_BASIC,
    FEATURE_DIM_EXTENDED,
    FEATURE_DIM_XGBOOST,
    FeatureMode,
)
from .utils import numbers_to_multihot, calc_ac_value


class FeatureExtractor:
    """
    로또 번호를 ML 피처로 변환하는 클래스

    피처 모드:
    - basic: 45차원 (multi-hot only)
    - extended: 74차원 (전체 확장 피처)
    - xgboost: 66차원 (중복 제거된 확장 피처)
    """

    def __init__(self, feature_mode: str = FeatureMode.BASIC, use_bonus: bool = False):
        """
        Args:
            feature_mode: 피처 모드 ('basic', 'extended', 'xgboost')
            use_bonus: 보너스 번호 피처 사용 여부
        """
        self.feature_mode = feature_mode
        self.use_bonus = use_bonus
        self._feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self) -> int:
        """피처 차원 계산"""
        dims = {
            FeatureMode.BASIC: FEATURE_DIM_BASIC,
            FeatureMode.EXTENDED: FEATURE_DIM_EXTENDED,
            FeatureMode.XGBOOST: FEATURE_DIM_XGBOOST,
        }
        base_dim = dims.get(self.feature_mode, FEATURE_DIM_BASIC)
        return base_dim + (1 if self.use_bonus else 0)

    @property
    def feature_dim(self) -> int:
        """피처 차원"""
        return self._feature_dim

    def extract(self,
                numbers: List[int],
                bonus: Optional[int] = None,
                prev_numbers: Optional[List[int]] = None,
                round_num: Optional[int] = None) -> np.ndarray:
        """
        번호를 피처로 변환

        Args:
            numbers: 당첨 번호 6개
            bonus: 보너스 번호 (optional)
            prev_numbers: 이전 회차 번호 (optional)
            round_num: 회차 번호 (optional)

        Returns:
            피처 벡터
        """
        if self.feature_mode == FeatureMode.BASIC:
            return self._extract_basic(numbers, bonus)
        elif self.feature_mode == FeatureMode.XGBOOST:
            return self._extract_xgboost(numbers, bonus, prev_numbers, round_num)
        else:
            return self._extract_extended(numbers, bonus, prev_numbers, round_num)

    def _extract_basic(self, numbers: List[int], bonus: Optional[int] = None) -> np.ndarray:
        """기본 피처 추출 (45차원)"""
        multihot = numbers_to_multihot(numbers)

        if self.use_bonus and bonus is not None:
            bonus_norm = (bonus - 1) / 44.0
            return np.concatenate([multihot, [bonus_norm]])

        return multihot

    def _extract_core_features(self,
                               numbers: List[int],
                               prev_numbers: Optional[List[int]] = None,
                               round_num: Optional[int] = None) -> dict:
        """핵심 피처 추출 (공통)"""
        sorted_nums = sorted(numbers)

        # gaps (5): 인접 번호 간격 패턴
        gaps = [(sorted_nums[i + 1] - sorted_nums[i]) / 44.0 for i in range(5)]

        # consecutive (1): 연번 개수
        consecutive = sum(1 for i in range(5) if sorted_nums[i + 1] - sorted_nums[i] == 1) / 5.0

        # ac_value (1): 산술 복잡도
        ac_value = calc_ac_value(numbers) / 10.0

        # prime_count (1): 소수 개수
        prime_count = sum(1 for n in numbers if n in PRIMES) / 6.0

        # ending_dist (10): 끝수 분포
        ending_bins = [0] * 10
        for n in numbers:
            ending_bins[n % 10] += 1
        ending_dist = [b / 6.0 for b in ending_bins]

        # carryover (1): 이월 번호 비율
        if prev_numbers is not None:
            carryover = len(set(prev_numbers) & set(numbers)) / 6.0
        else:
            carryover = 0.0

        # round_sin/cos (2): 회차 주기성
        if round_num is not None:
            round_sin = np.sin(2 * np.pi * round_num / DEFAULT_PERIOD)
            round_cos = np.cos(2 * np.pi * round_num / DEFAULT_PERIOD)
        else:
            round_sin = 0.0
            round_cos = 0.0

        return {
            'gaps': gaps,
            'consecutive': consecutive,
            'ac_value': ac_value,
            'prime_count': prime_count,
            'ending_dist': ending_dist,
            'carryover': carryover,
            'round_sin': round_sin,
            'round_cos': round_cos,
        }

    def _extract_xgboost(self,
                         numbers: List[int],
                         bonus: Optional[int] = None,
                         prev_numbers: Optional[List[int]] = None,
                         round_num: Optional[int] = None) -> np.ndarray:
        """XGBoost 전용 피처 추출 (66차원)"""
        multihot = numbers_to_multihot(numbers)
        core = self._extract_core_features(numbers, prev_numbers, round_num)

        features = np.concatenate([
            multihot,                       # 45
            core['gaps'],                   # 5
            [core['consecutive']],          # 1
            [core['ac_value']],             # 1
            [core['prime_count']],          # 1
            core['ending_dist'],            # 10
            [core['carryover']],            # 1
            [core['round_sin']],            # 1
            [core['round_cos']],            # 1
        ])  # Total: 66

        if self.use_bonus and bonus is not None:
            bonus_norm = (bonus - 1) / 44.0
            features = np.concatenate([features, [bonus_norm]])

        return features

    def _extract_extended(self,
                          numbers: List[int],
                          bonus: Optional[int] = None,
                          prev_numbers: Optional[List[int]] = None,
                          round_num: Optional[int] = None) -> np.ndarray:
        """확장 피처 추출 (74차원)"""
        multihot = numbers_to_multihot(numbers)
        core = self._extract_core_features(numbers, prev_numbers, round_num)

        # 추가 피처 (XGBoost mean과 유사하지만 DL 모델에 유용)
        odd_count = sum(1 for n in numbers if n % 2 == 1) / 6.0
        high_count = sum(1 for n in numbers if n >= 23) / 6.0
        total_sum = sum(numbers)
        sum_norm = (total_sum - 21) / (255 - 21)

        # 구간 분포 (10단위)
        decade_bins = [0] * 5
        for n in numbers:
            if n <= 9:
                decade_bins[0] += 1
            elif n <= 19:
                decade_bins[1] += 1
            elif n <= 29:
                decade_bins[2] += 1
            elif n <= 39:
                decade_bins[3] += 1
            else:
                decade_bins[4] += 1
        decade_dist = [b / 6.0 for b in decade_bins]

        features = np.concatenate([
            multihot,                       # 45
            [odd_count],                    # 1
            [high_count],                   # 1
            [sum_norm],                     # 1
            core['gaps'],                   # 5
            decade_dist,                    # 5
            [core['consecutive']],          # 1
            [core['ac_value']],             # 1
            [core['prime_count']],          # 1
            core['ending_dist'],            # 10
            [core['carryover']],            # 1
            [core['round_sin']],            # 1
            [core['round_cos']],            # 1
        ])  # Total: 74

        if self.use_bonus and bonus is not None:
            bonus_norm = (bonus - 1) / 44.0
            features = np.concatenate([features, [bonus_norm]])

        return features

    def get_feature_names(self) -> List[str]:
        """피처 이름 리스트 반환"""
        names = [f"num_{i+1}" for i in range(45)]

        if self.feature_mode == FeatureMode.BASIC:
            pass
        elif self.feature_mode == FeatureMode.XGBOOST:
            names += [f"gap_{i+1}" for i in range(5)]
            names += ["consecutive", "ac_value", "prime_count"]
            names += [f"ending_{i}" for i in range(10)]
            names += ["carryover", "round_sin", "round_cos"]
        else:  # extended
            names += ["odd_count", "high_count", "sum_norm"]
            names += [f"gap_{i+1}" for i in range(5)]
            names += [f"decade_{i}" for i in range(5)]
            names += ["consecutive", "ac_value", "prime_count"]
            names += [f"ending_{i}" for i in range(10)]
            names += ["carryover", "round_sin", "round_cos"]

        if self.use_bonus:
            names.append("bonus")

        return names
