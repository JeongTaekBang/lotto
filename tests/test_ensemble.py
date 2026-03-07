"""앙상블 매니저 테스트"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.manager import EnsembleManager
from ensemble.weighted_average import WeightedAverageEnsemble
from ensemble.voting import VotingEnsemble
from core.types import ProbabilityDistribution, Prediction
from filters.pattern_filter import PatternFilter


class TestEnsembleManager:
    """EnsembleManager 기본 기능 테스트"""

    @pytest.fixture
    def manager(self):
        """빈 EnsembleManager 생성"""
        return EnsembleManager()

    @pytest.fixture
    def manager_with_models(self, mock_model):
        """모델이 등록된 EnsembleManager"""
        manager = EnsembleManager()
        manager.register_model('mock1', mock_model)

        # 두 번째 mock 모델
        from tests.conftest import MockModel
        mock2 = MockModel()
        mock2.model_type = "mock2"
        manager.register_model('mock2', mock2)

        return manager

    def test_register_model(self, manager, mock_model):
        """모델 등록"""
        manager.register_model('test', mock_model)

        assert 'test' in manager.models
        assert len(manager.models) == 1

    def test_unregister_model(self, manager, mock_model):
        """모델 등록 해제"""
        manager.register_model('test', mock_model)
        result = manager.unregister_model('test')

        assert result is True
        assert 'test' not in manager.models

    def test_unregister_nonexistent_model(self, manager):
        """미등록 모델 해제 시도"""
        result = manager.unregister_model('nonexistent')
        assert result is False

    def test_predict_without_models_raises_error(self, manager, sample_single_X):
        """모델 없이 예측 시 오류"""
        with pytest.raises(ValueError) as excinfo:
            manager.predict(sample_single_X)
        assert '등록된 모델이 없습니다' in str(excinfo.value)

    def test_predict_single_model(self, manager, mock_model, sample_single_X):
        """단일 모델 예측"""
        manager.register_model('test', mock_model)
        predictions = manager.predict(sample_single_X, num_sets=3, apply_filters=False)

        assert len(predictions) == 3
        for pred in predictions:
            assert len(pred.numbers) == 6


class TestEnsembleStrategies:
    """앙상블 전략 테스트"""

    @pytest.fixture
    def manager_with_strategy(self, mock_model):
        """전략이 설정된 EnsembleManager"""
        manager = EnsembleManager()

        # 두 개의 mock 모델
        from tests.conftest import MockModel
        mock1 = MockModel()
        mock1.model_type = "mock1"
        mock2 = MockModel()
        mock2.model_type = "mock2"

        manager.register_model('mock1', mock1)
        manager.register_model('mock2', mock2)

        return manager

    def test_set_weighted_average_strategy(self, manager_with_strategy):
        """가중 평균 전략 설정"""
        strategy = WeightedAverageEnsemble()
        manager_with_strategy.set_strategy(strategy)

        assert manager_with_strategy.ensemble_strategy is not None
        assert len(manager_with_strategy.ensemble_strategy.models) == 2

    def test_set_voting_strategy(self, manager_with_strategy):
        """투표 전략 설정"""
        strategy = VotingEnsemble()
        manager_with_strategy.set_strategy(strategy)

        assert manager_with_strategy.ensemble_strategy is not None

    def test_predict_with_strategy(self, manager_with_strategy, sample_single_X):
        """전략 적용 예측"""
        strategy = WeightedAverageEnsemble()
        manager_with_strategy.set_strategy(strategy)

        predictions = manager_with_strategy.predict(
            sample_single_X, num_sets=5, apply_filters=False
        )

        assert len(predictions) == 5

    def test_predict_proba_with_strategy(self, manager_with_strategy, sample_single_X):
        """전략 적용 확률 분포"""
        strategy = WeightedAverageEnsemble()
        manager_with_strategy.set_strategy(strategy)

        proba = manager_with_strategy.predict_proba(sample_single_X)

        assert isinstance(proba, ProbabilityDistribution)
        assert proba.probabilities.shape == (45,)


class TestAutoWeight:
    """자동 가중치 계산 테스트"""

    @pytest.fixture
    def manager_for_weight(self):
        """가중치 테스트용 매니저"""
        from tests.conftest import MockModel
        manager = EnsembleManager()

        mock1 = MockModel()
        mock1.model_type = "mock1"
        mock2 = MockModel()
        mock2.model_type = "mock2"

        manager.register_model('mock1', mock1)
        manager.register_model('mock2', mock2)

        strategy = WeightedAverageEnsemble()
        manager.set_strategy(strategy)

        return manager

    def test_auto_weight_softmax(self, manager_for_weight, sample_X_sequence, sample_y):
        """Softmax 방식 자동 가중치"""
        weights = manager_for_weight.auto_weight_by_performance(
            sample_X_sequence, sample_y,
            method='softmax',
            verbose=False
        )

        assert isinstance(weights, dict)
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.01  # 합이 1에 가까움

    def test_auto_weight_linear(self, manager_for_weight, sample_X_sequence, sample_y):
        """Linear 방식 자동 가중치"""
        weights = manager_for_weight.auto_weight_by_performance(
            sample_X_sequence, sample_y,
            method='linear',
            verbose=False
        )

        assert isinstance(weights, dict)
        assert all(w >= 0 for w in weights.values())

    def test_auto_weight_rank(self, manager_for_weight, sample_X_sequence, sample_y):
        """Rank 방식 자동 가중치"""
        weights = manager_for_weight.auto_weight_by_performance(
            sample_X_sequence, sample_y,
            method='rank',
            verbose=False
        )

        assert isinstance(weights, dict)


class TestFilterPipeline:
    """필터 파이프라인 테스트"""

    @pytest.fixture
    def manager_with_filter(self, mock_model):
        """필터가 설정된 매니저"""
        manager = EnsembleManager()
        manager.register_model('test', mock_model)
        return manager

    def test_add_filter(self, manager_with_filter):
        """필터 추가"""
        filter_ = PatternFilter()
        manager_with_filter.add_filter(filter_)

        assert len(manager_with_filter.filters) == 1

    def test_remove_filter(self, manager_with_filter):
        """필터 제거"""
        filter_ = PatternFilter()
        manager_with_filter.add_filter(filter_)
        result = manager_with_filter.remove_filter('pattern')

        assert result is True
        assert len(manager_with_filter.filters) == 0

    def test_clear_filters(self, manager_with_filter):
        """모든 필터 제거"""
        filter_ = PatternFilter()
        manager_with_filter.add_filter(filter_)
        manager_with_filter.clear_filters()

        assert len(manager_with_filter.filters) == 0

    def test_predict_with_filter(self, manager_with_filter, sample_single_X):
        """필터 적용 예측"""
        filter_ = PatternFilter()
        manager_with_filter.add_filter(filter_)

        predictions = manager_with_filter.predict(
            sample_single_X, num_sets=5, apply_filters=True
        )

        # 필터를 통과한 결과 반환
        assert len(predictions) <= 5


class TestPredictIndividual:
    """개별 모델 예측 테스트"""

    def test_predict_individual(self, mock_model, sample_single_X):
        """각 모델의 개별 예측"""
        from tests.conftest import MockModel
        manager = EnsembleManager()

        mock1 = MockModel()
        mock1.model_type = "model_a"
        mock2 = MockModel()
        mock2.model_type = "model_b"

        manager.register_model('model_a', mock1)
        manager.register_model('model_b', mock2)

        results = manager.predict_individual(sample_single_X)

        assert 'model_a' in results
        assert 'model_b' in results
        assert isinstance(results['model_a'], ProbabilityDistribution)


class TestModelRankings:
    """모델 순위 테스트"""

    def test_get_model_rankings_empty(self):
        """성능 데이터 없을 때"""
        manager = EnsembleManager()
        rankings = manager.get_model_rankings()

        assert rankings == []

    def test_get_model_rankings_after_eval(self, sample_X_sequence, sample_y):
        """평가 후 순위"""
        from tests.conftest import MockModel
        manager = EnsembleManager()

        mock1 = MockModel()
        mock1.model_type = "model_a"
        mock2 = MockModel()
        mock2.model_type = "model_b"

        manager.register_model('model_a', mock1)
        manager.register_model('model_b', mock2)

        # 평가 수행
        manager.evaluate_all(sample_X_sequence, sample_y, verbose=False)

        rankings = manager.get_model_rankings()
        assert len(rankings) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
