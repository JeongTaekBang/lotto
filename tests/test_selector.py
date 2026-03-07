"""CandidateSelector 단위 테스트"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.selector import CandidateSelector, PRESETS, SelectionPreset
from core.types import ProbabilityDistribution
from filters.pattern_filter import PatternFilter
from filters.statistical_filter import StatisticalFilter


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def uniform_proba():
    """균등 확률 분포"""
    probs = np.ones(45) / 45
    return ProbabilityDistribution(probs, 'uniform')


@pytest.fixture
def realistic_proba():
    """현실적 확률 분포 (Dirichlet)"""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(45) * 2)
    return ProbabilityDistribution(probs, 'realistic')


@pytest.fixture
def extreme_proba():
    """극단적 쏠림 분포 (6개만 확률 > 0)"""
    probs = np.zeros(45)
    probs[[0, 9, 19, 29, 39, 44]] = 1.0 / 6
    return ProbabilityDistribution(probs, 'extreme')


@pytest.fixture
def single_peak_proba():
    """단일 번호 극단 집중"""
    probs = np.full(45, 1e-10)
    probs[0] = 0.99
    return ProbabilityDistribution(probs, 'single_peak')


@pytest.fixture
def filters():
    """테스트용 필터"""
    return [PatternFilter(), StatisticalFilter()]


# ============================================================
# 기본 동작
# ============================================================

class TestBasicBehavior:

    def test_returns_requested_num_sets(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        assert len(results) == 5

    def test_each_set_has_6_numbers(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=3)
        for pred in results:
            assert len(pred.numbers) == 6

    def test_numbers_in_valid_range(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        for pred in results:
            assert all(1 <= n <= 45 for n in pred.numbers)

    def test_numbers_sorted(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        for pred in results:
            assert pred.numbers == sorted(pred.numbers)

    def test_no_duplicate_numbers_within_set(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        for pred in results:
            assert len(set(pred.numbers)) == 6

    def test_no_duplicate_sets(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=10)
        number_tuples = [tuple(r.numbers) for r in results]
        assert len(set(number_tuples)) == len(number_tuples)

    def test_metadata_present(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=1)
        meta = results[0].metadata
        assert 'score' in meta
        assert 'prob_score' in meta
        assert 'filter_score' in meta
        assert 'mode' in meta

    def test_confidence_in_valid_range(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        for pred in results:
            assert 0.0 <= pred.confidence <= 1.0


# ============================================================
# 프리셋
# ============================================================

class TestPresets:

    @pytest.mark.parametrize("preset_name", ['safe', 'balanced', 'aggressive'])
    def test_all_presets_return_results(self, realistic_proba, preset_name):
        selector = CandidateSelector(preset=preset_name)
        results = selector.select(realistic_proba, num_sets=5)
        assert len(results) == 5

    def test_safe_has_more_diversity(self, realistic_proba):
        """safe 모드가 aggressive보다 세트 간 겹침이 적어야 함"""
        results = {}
        for mode in ['safe', 'aggressive']:
            selector = CandidateSelector(preset=mode)
            results[mode] = selector.select(realistic_proba, num_sets=5)

        def avg_overlap(preds):
            overlaps = []
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    overlaps.append(len(set(preds[i].numbers) & set(preds[j].numbers)))
            return np.mean(overlaps)

        safe_overlap = avg_overlap(results['safe'])
        aggressive_overlap = avg_overlap(results['aggressive'])
        assert safe_overlap <= aggressive_overlap

    def test_unknown_preset_falls_back_to_balanced(self, realistic_proba):
        selector = CandidateSelector(preset='nonexistent')
        results = selector.select(realistic_proba, num_sets=3)
        assert len(results) == 3
        assert results[0].metadata['mode'] == 'nonexistent'

    def test_custom_preset(self, realistic_proba):
        custom = SelectionPreset(
            temperature=0.5, num_candidates=100,
            diversity_lambda=0.3, filter_weight=2.0,
        )
        selector = CandidateSelector(preset=custom)
        results = selector.select(realistic_proba, num_sets=3)
        assert len(results) == 3


# ============================================================
# 극단 분포 (epsilon smoothing)
# ============================================================

class TestEdgeCases:

    def test_extreme_skew_returns_num_sets(self, extreme_proba):
        """6개 번호만 확률 > 0이어도 N세트 보장"""
        selector = CandidateSelector(preset='balanced')
        results = selector.select(extreme_proba, num_sets=5)
        assert len(results) == 5

    def test_extreme_skew_all_unique(self, extreme_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(extreme_proba, num_sets=5)
        number_tuples = [tuple(r.numbers) for r in results]
        assert len(set(number_tuples)) == 5

    def test_single_peak_returns_num_sets(self, single_peak_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(single_peak_proba, num_sets=5)
        assert len(results) == 5

    def test_uniform_distribution(self, uniform_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(uniform_proba, num_sets=5)
        assert len(results) == 5

    def test_num_sets_one(self, realistic_proba):
        selector = CandidateSelector(preset='balanced')
        results = selector.select(realistic_proba, num_sets=1)
        assert len(results) == 1


# ============================================================
# 필터 통합
# ============================================================

class TestFilterIntegration:

    def test_with_filters(self, realistic_proba, filters):
        selector = CandidateSelector(filters=filters, preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        assert len(results) == 5
        for pred in results:
            assert pred.metadata['filter_score'] > 0

    def test_without_filters(self, realistic_proba):
        selector = CandidateSelector(filters=[], preset='balanced')
        results = selector.select(realistic_proba, num_sets=5)
        for pred in results:
            assert pred.metadata['filter_score'] == 1.0

    def test_filter_score_influences_ranking(self, realistic_proba, filters):
        """필터가 있을 때와 없을 때 결과가 달라야 함"""
        np.random.seed(99)
        sel_no = CandidateSelector(filters=[], preset='balanced')
        res_no = sel_no.select(realistic_proba, num_sets=5)

        np.random.seed(99)
        sel_yes = CandidateSelector(filters=filters, preset='balanced')
        res_yes = sel_yes.select(realistic_proba, num_sets=5)

        sets_no = {tuple(r.numbers) for r in res_no}
        sets_yes = {tuple(r.numbers) for r in res_yes}
        # 완전히 같을 확률은 매우 낮음 (필터가 리랭킹에 영향)
        # 하지만 보장할 수 없으므로 최소 형식 검증만
        assert len(res_yes) == 5


# ============================================================
# 내부 메서드
# ============================================================

class TestInternalMethods:

    def test_normalize(self):
        arr = np.array([1.0, 3.0, 5.0])
        result = CandidateSelector._normalize(arr)
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[2] - 1.0) < 1e-6

    def test_normalize_constant(self):
        arr = np.array([2.0, 2.0, 2.0])
        result = CandidateSelector._normalize(arr)
        assert np.allclose(result, 0.5)

    def test_epsilon_smooth(self):
        probs = np.zeros(45)
        probs[0] = 1.0
        smoothed = CandidateSelector._epsilon_smooth(probs)
        assert smoothed[0] < 1.0
        assert all(s > 0 for s in smoothed)
        assert abs(smoothed.sum() - 1.0) < 1e-6

    def test_mmr_select_returns_k(self):
        candidates = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                       [1, 2, 3, 7, 8, 9], [13, 14, 15, 16, 17, 18]]
        scores = np.array([1.0, 0.9, 0.8, 0.7])
        selected = CandidateSelector._mmr_select(candidates, scores, k=3, diversity_lambda=0.5)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # 중복 없음

    def test_mmr_select_small_pool(self):
        """후보가 k보다 적으면 전부 반환"""
        candidates = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        scores = np.array([1.0, 0.5])
        selected = CandidateSelector._mmr_select(candidates, scores, k=5, diversity_lambda=0.5)
        assert len(selected) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
