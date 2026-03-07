"""core/feature_extractor.py 단위 테스트"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

from core.feature_extractor import FeatureExtractor
from core.constants import (
    FEATURE_DIM_BASIC,
    FEATURE_DIM_EXTENDED,
    FEATURE_DIM_XGBOOST,
    FeatureMode,
)


class TestFeatureExtractorBasic(unittest.TestCase):
    """Basic 모드 테스트"""

    def setUp(self):
        self.extractor = FeatureExtractor(FeatureMode.BASIC)
        self.sample_numbers = [1, 10, 20, 30, 40, 45]

    def test_feature_dim(self):
        """기본 피처 차원"""
        self.assertEqual(self.extractor.feature_dim, FEATURE_DIM_BASIC)

    def test_extract_basic(self):
        """기본 피처 추출"""
        features = self.extractor.extract(self.sample_numbers)

        self.assertEqual(len(features), 45)
        self.assertEqual(features.sum(), 6)  # 6개 번호

        # 올바른 위치에 1이 있는지 확인
        for num in self.sample_numbers:
            self.assertEqual(features[num - 1], 1)


class TestFeatureExtractorExtended(unittest.TestCase):
    """Extended 모드 테스트"""

    def setUp(self):
        self.extractor = FeatureExtractor(FeatureMode.EXTENDED)
        self.sample_numbers = [1, 10, 20, 30, 40, 45]

    def test_feature_dim(self):
        """확장 피처 차원"""
        self.assertEqual(self.extractor.feature_dim, FEATURE_DIM_EXTENDED)

    def test_extract_extended(self):
        """확장 피처 추출"""
        features = self.extractor.extract(self.sample_numbers)
        self.assertEqual(len(features), 74)

    def test_with_prev_numbers(self):
        """이전 번호 포함 피처"""
        prev = [1, 2, 3, 4, 5, 6]
        features = self.extractor.extract(
            self.sample_numbers,
            prev_numbers=prev
        )

        # carryover 피처가 0이 아님 (1번이 공통)
        self.assertEqual(len(features), 74)

    def test_with_round_num(self):
        """회차 번호 포함 피처"""
        features = self.extractor.extract(
            self.sample_numbers,
            round_num=100
        )

        # round_sin, round_cos가 0이 아님
        self.assertEqual(len(features), 74)


class TestFeatureExtractorXGBoost(unittest.TestCase):
    """XGBoost 모드 테스트"""

    def setUp(self):
        self.extractor = FeatureExtractor(FeatureMode.XGBOOST)
        self.sample_numbers = [1, 10, 20, 30, 40, 45]

    def test_feature_dim(self):
        """XGBoost 피처 차원"""
        self.assertEqual(self.extractor.feature_dim, FEATURE_DIM_XGBOOST)

    def test_extract_xgboost(self):
        """XGBoost 피처 추출"""
        features = self.extractor.extract(self.sample_numbers)
        self.assertEqual(len(features), 66)


class TestFeatureExtractorWithBonus(unittest.TestCase):
    """보너스 번호 포함 테스트"""

    def test_basic_with_bonus(self):
        """Basic + 보너스"""
        extractor = FeatureExtractor(FeatureMode.BASIC, use_bonus=True)
        self.assertEqual(extractor.feature_dim, FEATURE_DIM_BASIC + 1)

        features = extractor.extract([1, 2, 3, 4, 5, 6], bonus=7)
        self.assertEqual(len(features), 46)

    def test_extended_with_bonus(self):
        """Extended + 보너스"""
        extractor = FeatureExtractor(FeatureMode.EXTENDED, use_bonus=True)
        self.assertEqual(extractor.feature_dim, FEATURE_DIM_EXTENDED + 1)


class TestFeatureExtractorFeatureNames(unittest.TestCase):
    """피처 이름 테스트"""

    def test_basic_names(self):
        """Basic 피처 이름"""
        extractor = FeatureExtractor(FeatureMode.BASIC)
        names = extractor.get_feature_names()
        self.assertEqual(len(names), 45)

    def test_extended_names(self):
        """Extended 피처 이름"""
        extractor = FeatureExtractor(FeatureMode.EXTENDED)
        names = extractor.get_feature_names()
        self.assertEqual(len(names), 74)

    def test_xgboost_names(self):
        """XGBoost 피처 이름"""
        extractor = FeatureExtractor(FeatureMode.XGBOOST)
        names = extractor.get_feature_names()
        self.assertEqual(len(names), 66)


class TestFeatureExtractorConsistency(unittest.TestCase):
    """일관성 테스트"""

    def test_same_input_same_output(self):
        """같은 입력 → 같은 출력"""
        extractor = FeatureExtractor(FeatureMode.EXTENDED)
        numbers = [5, 12, 23, 34, 41, 45]

        features1 = extractor.extract(numbers, round_num=100)
        features2 = extractor.extract(numbers, round_num=100)

        np.testing.assert_array_equal(features1, features2)

    def test_normalized_values(self):
        """정규화된 값 범위"""
        extractor = FeatureExtractor(FeatureMode.EXTENDED)
        numbers = [1, 15, 23, 30, 38, 45]

        features = extractor.extract(numbers, round_num=100)

        # 대부분의 피처는 0-1 범위 (sin/cos 제외)
        # multi-hot은 0 또는 1
        for i in range(45):
            self.assertIn(features[i], [0, 1])


if __name__ == '__main__':
    unittest.main()
