"""필터 테스트"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.pattern_filter import PatternFilter
from filters.statistical_filter import StatisticalFilter
from filters.frequency_filter import FrequencyFilter
from filters.composite_filter import CompositeFilter, WeightedCompositeFilter, ScoreRankingFilter
from core.types import Prediction


class TestPatternFilter:
    """PatternFilter 테스트"""

    @pytest.fixture
    def filter_(self):
        """PatternFilter 생성"""
        return PatternFilter()

    def test_count_odd(self, filter_):
        """홀수 개수 계산"""
        # 3 홀수: 1, 11, 21
        assert filter_._count_odd([1, 2, 11, 12, 21, 22]) == 3
        # 모두 홀수
        assert filter_._count_odd([1, 3, 5, 7, 9, 11]) == 6
        # 모두 짝수
        assert filter_._count_odd([2, 4, 6, 8, 10, 12]) == 0

    def test_count_low(self, filter_):
        """저번호 개수 계산"""
        # 저번호 (1-22): 1, 2, 11, 12, 21, 22 중 22까지 = 6개
        assert filter_._count_low([1, 2, 11, 12, 21, 22]) == 6
        # 고번호만: 23-45
        assert filter_._count_low([23, 24, 25, 40, 41, 45]) == 0

    def test_count_consecutive(self, filter_):
        """연속번호 쌍 개수"""
        # 1-2, 2-3, 3-4 = 3쌍
        assert filter_._count_consecutive([1, 2, 3, 4, 10, 20]) == 3
        # 연속 없음
        assert filter_._count_consecutive([1, 5, 10, 20, 30, 40]) == 0
        # 1-2, 2-3 = 2쌍
        assert filter_._count_consecutive([1, 2, 3, 10, 20, 30]) == 2

    def test_score_valid_pattern(self, filter_, valid_pattern_numbers):
        """유효한 패턴 점수"""
        score = filter_.score(valid_pattern_numbers)
        assert 0.7 <= score <= 1.0

    def test_score_invalid_pattern(self, filter_, invalid_pattern_numbers):
        """무효한 패턴 점수 (올 짝수)"""
        score = filter_.score(invalid_pattern_numbers)
        assert score < 0.8  # 홀수 0개 = 낮은 점수

    def test_filter_predictions(self, filter_, sample_predictions):
        """예측 필터링"""
        filtered = filter_.filter(sample_predictions)
        assert len(filtered) >= 1  # 최소 1개 반환

    def test_analyze(self, filter_):
        """번호 분석"""
        analysis = filter_.analyze([1, 2, 23, 24, 43, 44])

        assert 'odd_count' in analysis
        assert 'sum' in analysis
        assert 'score' in analysis

    def test_filter_disabled(self, filter_, sample_predictions):
        """비활성화된 필터"""
        filter_.enabled = False
        filtered = filter_.filter(sample_predictions)
        assert len(filtered) == len(sample_predictions)


class TestStatisticalFilter:
    """StatisticalFilter 테스트"""

    @pytest.fixture
    def filter_(self):
        """StatisticalFilter 생성"""
        return StatisticalFilter()

    def test_calculate_ac(self, filter_):
        """AC값 계산"""
        # AC값 = 고유 차이 개수 - 5
        # [1, 10, 20, 30, 40, 45]: 다양한 차이
        ac = filter_._calculate_ac([1, 10, 20, 30, 40, 45])
        assert 5 <= ac <= 10

    def test_score(self, filter_):
        """통계 점수"""
        score = filter_.score([5, 12, 23, 28, 37, 44])
        assert 0 <= score <= 1

    def test_filter_predictions(self, filter_, sample_predictions):
        """예측 필터링"""
        filtered = filter_.filter(sample_predictions)
        assert len(filtered) >= 1


class TestFrequencyFilter:
    """FrequencyFilter 테스트"""

    @pytest.fixture
    def filter_(self):
        """FrequencyFilter 생성"""
        # FrequencyFilter는 기본 설정으로 생성
        return FrequencyFilter()

    def test_score(self, filter_):
        """빈도 점수"""
        # 기본 점수 계산 테스트
        score = filter_.score([5, 12, 23, 28, 37, 44])
        assert 0 <= score <= 1


class TestCompositeFilter:
    """CompositeFilter 테스트"""

    @pytest.fixture
    def composite(self):
        """CompositeFilter 생성"""
        composite = CompositeFilter()
        composite.add_filter(PatternFilter())
        composite.add_filter(StatisticalFilter())
        return composite

    def test_add_filter(self):
        """필터 추가"""
        composite = CompositeFilter()
        assert len(composite) == 0

        composite.add_filter(PatternFilter())
        assert len(composite) == 1

    def test_remove_filter(self, composite):
        """필터 제거"""
        result = composite.remove_filter('pattern')
        assert result is True
        assert len(composite) == 1

    def test_score(self, composite):
        """복합 점수 (평균)"""
        score = composite.score([5, 12, 23, 28, 37, 44])
        assert 0 <= score <= 1

    def test_filter_sequential(self, composite, sample_predictions):
        """순차 필터링"""
        filtered = composite.filter(sample_predictions)
        assert len(filtered) >= 1


class TestWeightedCompositeFilter:
    """WeightedCompositeFilter 테스트"""

    @pytest.fixture
    def weighted(self):
        """WeightedCompositeFilter 생성"""
        weighted = WeightedCompositeFilter(threshold=0.6)
        weighted.add_filter(PatternFilter(), weight=2.0)
        weighted.add_filter(StatisticalFilter(), weight=1.0)
        return weighted

    def test_weighted_score(self, weighted):
        """가중 점수"""
        score = weighted.score([5, 12, 23, 28, 37, 44])
        assert 0 <= score <= 1

    def test_filter_with_threshold(self, weighted, sample_predictions):
        """임계값 필터링"""
        filtered = weighted.filter(sample_predictions)
        assert len(filtered) >= 1


class TestScoreRankingFilter:
    """ScoreRankingFilter 테스트"""

    @pytest.fixture
    def ranking(self):
        """ScoreRankingFilter 생성"""
        ranking = ScoreRankingFilter(top_n=2)
        ranking.add_filter(PatternFilter())
        return ranking

    def test_filter_top_n(self, ranking):
        """상위 N개 선택"""
        predictions = [
            Prediction(numbers=[1, 2, 3, 4, 5, 6], confidence=0.9, model_name="test"),
            Prediction(numbers=[5, 12, 23, 28, 37, 44], confidence=0.8, model_name="test"),
            Prediction(numbers=[2, 4, 6, 8, 10, 12], confidence=0.7, model_name="test"),
        ]

        filtered = ranking.filter(predictions)
        assert len(filtered) == 2

    def test_filter_empty(self, ranking):
        """빈 입력"""
        filtered = ranking.filter([])
        assert filtered == []


class TestFilterEnabled:
    """필터 활성화/비활성화 테스트"""

    def test_disabled_filter_passes_all(self, sample_predictions):
        """비활성화된 필터는 모든 예측 통과"""
        filter_ = PatternFilter(enabled=False)
        filtered = filter_.filter(sample_predictions)

        # BaseFilter에서 disabled면 전체 반환
        assert len(filtered) == len(sample_predictions)

    def test_composite_with_disabled_filter(self, sample_predictions):
        """복합 필터 내 비활성화된 필터"""
        composite = CompositeFilter()
        pattern = PatternFilter(enabled=True)
        stat = StatisticalFilter(enabled=False)

        composite.add_filter(pattern)
        composite.add_filter(stat)

        # score 계산 시 enabled된 것만 사용
        score = composite.score([5, 12, 23, 28, 37, 44])
        assert 0 <= score <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
