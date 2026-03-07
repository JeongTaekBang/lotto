"""core/constants.py 단위 테스트"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from core.constants import (
    LOTTO_MIN_NUMBER,
    LOTTO_MAX_NUMBER,
    LOTTO_NUMBERS_COUNT,
    LOTTO_POOL_SIZE,
    FEATURE_DIM_BASIC,
    FEATURE_DIM_EXTENDED,
    FEATURE_DIM_XGBOOST,
    DEFAULT_SEQ_LENGTH,
    PRIZE_TABLE,
    PRIMES,
    MODEL_EXTENSIONS,
    ENSEMBLE_DEFAULT_MODELS,
    FeatureMode,
    get_feature_dim,
)


class TestLottoConstants(unittest.TestCase):
    """로또 게임 상수 테스트"""

    def test_number_range(self):
        """번호 범위"""
        self.assertEqual(LOTTO_MIN_NUMBER, 1)
        self.assertEqual(LOTTO_MAX_NUMBER, 45)

    def test_numbers_count(self):
        """선택 번호 개수"""
        self.assertEqual(LOTTO_NUMBERS_COUNT, 6)

    def test_pool_size(self):
        """번호 풀 크기"""
        self.assertEqual(LOTTO_POOL_SIZE, 45)


class TestFeatureDimensions(unittest.TestCase):
    """피처 차원 상수 테스트"""

    def test_basic_dim(self):
        """기본 피처 차원"""
        self.assertEqual(FEATURE_DIM_BASIC, 45)

    def test_extended_dim(self):
        """확장 피처 차원"""
        self.assertEqual(FEATURE_DIM_EXTENDED, 74)

    def test_xgboost_dim(self):
        """XGBoost 피처 차원"""
        self.assertEqual(FEATURE_DIM_XGBOOST, 66)

    def test_dimensions_relationship(self):
        """차원 간 관계"""
        self.assertLess(FEATURE_DIM_BASIC, FEATURE_DIM_XGBOOST)
        self.assertLess(FEATURE_DIM_XGBOOST, FEATURE_DIM_EXTENDED)


class TestPrizeTable(unittest.TestCase):
    """당첨금 테이블 테스트"""

    def test_prize_keys(self):
        """당첨금 키 (적중 개수)"""
        self.assertIn(3, PRIZE_TABLE)
        self.assertIn(4, PRIZE_TABLE)
        self.assertIn(5, PRIZE_TABLE)
        self.assertIn(6, PRIZE_TABLE)

    def test_prize_values_increasing(self):
        """당첨금 증가 확인"""
        self.assertLess(PRIZE_TABLE[3], PRIZE_TABLE[4])
        self.assertLess(PRIZE_TABLE[4], PRIZE_TABLE[5])
        self.assertLess(PRIZE_TABLE[5], PRIZE_TABLE[6])


class TestPrimes(unittest.TestCase):
    """소수 상수 테스트"""

    def test_primes_in_range(self):
        """소수가 1-45 범위 내"""
        for p in PRIMES:
            self.assertGreaterEqual(p, 1)
            self.assertLessEqual(p, 45)

    def test_known_primes(self):
        """알려진 소수 포함"""
        known = {2, 3, 5, 7, 11, 13}
        for p in known:
            self.assertIn(p, PRIMES)

    def test_no_composites(self):
        """합성수 미포함"""
        composites = {4, 6, 8, 9, 10, 12}
        for c in composites:
            self.assertNotIn(c, PRIMES)


class TestModelExtensions(unittest.TestCase):
    """모델 확장자 테스트"""

    def test_pytorch_models(self):
        """PyTorch 모델 확장자"""
        self.assertEqual(MODEL_EXTENSIONS['transformer'], '.pt')
        self.assertEqual(MODEL_EXTENSIONS['lstm'], '.pt')
        self.assertEqual(MODEL_EXTENSIONS['gru'], '.pt')

    def test_sklearn_models(self):
        """scikit-learn 모델 확장자"""
        self.assertEqual(MODEL_EXTENSIONS['xgboost'], '.pkl')
        self.assertEqual(MODEL_EXTENSIONS['random_forest'], '.pkl')
        self.assertEqual(MODEL_EXTENSIONS['markov'], '.pkl')


class TestEnsembleDefaultModels(unittest.TestCase):
    """앙상블 기본 모델 테스트"""

    def test_xgboost_excluded(self):
        """XGBoost 제외됨"""
        self.assertNotIn('xgboost', ENSEMBLE_DEFAULT_MODELS)

    def test_main_models_included(self):
        """주요 모델 포함"""
        expected = ['gru', 'transformer', 'random_forest', 'markov', 'lstm']
        for model in expected:
            self.assertIn(model, ENSEMBLE_DEFAULT_MODELS)


class TestFeatureMode(unittest.TestCase):
    """FeatureMode 상수 테스트"""

    def test_mode_values(self):
        """모드 값"""
        self.assertEqual(FeatureMode.BASIC, 'basic')
        self.assertEqual(FeatureMode.EXTENDED, 'extended')
        self.assertEqual(FeatureMode.XGBOOST, 'xgboost')


class TestGetFeatureDim(unittest.TestCase):
    """get_feature_dim 함수 테스트"""

    def test_basic_mode(self):
        """Basic 모드 차원"""
        self.assertEqual(get_feature_dim('basic'), 45)

    def test_extended_mode(self):
        """Extended 모드 차원"""
        self.assertEqual(get_feature_dim('extended'), 74)

    def test_xgboost_mode(self):
        """XGBoost 모드 차원"""
        self.assertEqual(get_feature_dim('xgboost'), 66)

    def test_with_bonus(self):
        """보너스 포함 차원"""
        self.assertEqual(get_feature_dim('basic', use_bonus=True), 46)
        self.assertEqual(get_feature_dim('extended', use_bonus=True), 75)

    def test_unknown_mode(self):
        """알 수 없는 모드 (기본값)"""
        self.assertEqual(get_feature_dim('unknown'), 45)


if __name__ == '__main__':
    unittest.main()
