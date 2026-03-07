"""core/utils.py 단위 테스트"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

from core.utils import (
    sample_numbers,
    flatten_sequence,
    numbers_to_multihot,
    multihot_to_numbers,
    calc_ac_value,
    normalize_value,
)


class TestSampleNumbers(unittest.TestCase):
    """sample_numbers 함수 테스트"""

    def test_returns_correct_count(self):
        """k개의 번호를 반환하는지 확인"""
        probs = np.random.random(45)
        result = sample_numbers(probs, k=6)
        self.assertEqual(len(result), 6)

    def test_no_duplicates(self):
        """중복 번호가 없는지 확인"""
        probs = np.random.random(45)
        result = sample_numbers(probs, k=6)
        self.assertEqual(len(result), len(set(result)))

    def test_valid_range(self):
        """1-45 범위 내의 번호인지 확인"""
        probs = np.random.random(45)
        result = sample_numbers(probs, k=6)
        for num in result:
            self.assertGreaterEqual(num, 1)
            self.assertLessEqual(num, 45)

    def test_temperature_effect(self):
        """온도 파라미터가 작동하는지 확인"""
        probs = np.zeros(45)
        probs[0] = 1.0  # 1번에 모든 확률 집중

        # 낮은 온도: 더 결정적
        np.random.seed(42)
        low_temp = sample_numbers(probs.copy(), k=6, temperature=0.1)

        # 높은 온도: 더 무작위
        np.random.seed(42)
        high_temp = sample_numbers(probs.copy(), k=6, temperature=2.0)

        # 낮은 온도에서 1번이 선택될 확률이 높음
        self.assertIn(1, low_temp)


class TestFlattenSequence(unittest.TestCase):
    """flatten_sequence 함수 테스트"""

    def test_2d_input_unchanged(self):
        """2D 입력은 그대로 반환"""
        X = np.random.random((10, 45))
        result = flatten_sequence(X)
        np.testing.assert_array_equal(result, X)

    def test_3d_to_2d(self):
        """3D → 2D 변환"""
        X = np.random.random((10, 20, 45))  # (batch, seq, features)
        result = flatten_sequence(X)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 10)  # batch 유지
        self.assertEqual(result.shape[1], 45 * 6)  # 6개 통계

    def test_correct_statistics(self):
        """올바른 통계값 추출"""
        seq = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        X = np.array([seq])  # (1, 3, 3)
        result = flatten_sequence(X)

        # mean, std, last, max, min, trend
        expected_mean = seq.mean(axis=0)
        expected_last = seq[-1]
        expected_max = seq.max(axis=0)
        expected_min = seq.min(axis=0)
        expected_trend = seq[-1] - seq[0]

        np.testing.assert_array_almost_equal(result[0, :3], expected_mean)
        np.testing.assert_array_almost_equal(result[0, 6:9], expected_last)


class TestNumbersToMultihot(unittest.TestCase):
    """numbers_to_multihot 함수 테스트"""

    def test_correct_encoding(self):
        """올바른 multi-hot 인코딩"""
        numbers = [1, 10, 45]
        result = numbers_to_multihot(numbers)

        self.assertEqual(result[0], 1)   # 1번
        self.assertEqual(result[9], 1)   # 10번
        self.assertEqual(result[44], 1)  # 45번
        self.assertEqual(result.sum(), 3)

    def test_out_of_range_ignored(self):
        """범위 밖 번호 무시"""
        numbers = [0, 46, 10]
        result = numbers_to_multihot(numbers)
        self.assertEqual(result.sum(), 1)  # 10번만 인코딩


class TestMultihotToNumbers(unittest.TestCase):
    """multihot_to_numbers 함수 테스트"""

    def test_correct_decoding(self):
        """올바른 디코딩"""
        multihot = np.zeros(45)
        multihot[0] = 1.0
        multihot[9] = 1.0
        multihot[44] = 1.0

        result = multihot_to_numbers(multihot)
        self.assertEqual(sorted(result), [1, 10, 45])

    def test_roundtrip(self):
        """인코딩/디코딩 왕복"""
        original = [5, 12, 23, 34, 42, 45]
        encoded = numbers_to_multihot(original)
        decoded = multihot_to_numbers(encoded)
        self.assertEqual(sorted(decoded), sorted(original))


class TestCalcAcValue(unittest.TestCase):
    """calc_ac_value 함수 테스트"""

    def test_consecutive_numbers(self):
        """연속 번호의 AC값 (최소)"""
        numbers = [1, 2, 3, 4, 5, 6]
        result = calc_ac_value(numbers)
        self.assertEqual(result, 0)  # 최소 AC값

    def test_spread_numbers(self):
        """분산된 번호의 AC값 (최대)"""
        numbers = [1, 10, 20, 30, 40, 45]
        result = calc_ac_value(numbers)
        self.assertGreater(result, 5)  # 높은 AC값


class TestNormalizeValue(unittest.TestCase):
    """normalize_value 함수 테스트"""

    def test_normalization(self):
        """정규화 동작"""
        self.assertEqual(normalize_value(50, 0, 100), 0.5)
        self.assertEqual(normalize_value(0, 0, 100), 0.0)
        self.assertEqual(normalize_value(100, 0, 100), 1.0)

    def test_same_min_max(self):
        """min == max인 경우"""
        self.assertEqual(normalize_value(5, 5, 5), 0.0)


if __name__ == '__main__':
    unittest.main()
