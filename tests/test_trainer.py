"""트레이너 테스트"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import UnifiedTrainer
from core.constants import (
    FEATURE_DIM_BASIC,
    FEATURE_DIM_EXTENDED,
    FEATURE_DIM_XGBOOST,
    FeatureMode,
)


class TestUnifiedTrainerInit:
    """UnifiedTrainer 초기화 테스트"""

    def test_init_basic_mode(self, mock_datasource):
        """Basic 모드 초기화"""
        trainer = UnifiedTrainer(
            mock_datasource,
            feature_mode=FeatureMode.BASIC
        )

        assert trainer.feature_dim == FEATURE_DIM_BASIC
        assert trainer.feature_mode == FeatureMode.BASIC

    def test_init_extended_mode(self, mock_datasource):
        """Extended 모드 초기화"""
        trainer = UnifiedTrainer(
            mock_datasource,
            feature_mode=FeatureMode.EXTENDED
        )

        assert trainer.feature_dim == FEATURE_DIM_EXTENDED
        assert trainer.feature_mode == FeatureMode.EXTENDED

    def test_init_xgboost_mode(self, mock_datasource):
        """XGBoost 모드 초기화"""
        trainer = UnifiedTrainer(
            mock_datasource,
            feature_mode=FeatureMode.XGBOOST
        )

        assert trainer.feature_dim == FEATURE_DIM_XGBOOST
        assert trainer.feature_mode == FeatureMode.XGBOOST

    def test_init_with_bonus(self, mock_datasource):
        """보너스 번호 피처 포함"""
        trainer = UnifiedTrainer(
            mock_datasource,
            feature_mode=FeatureMode.BASIC,
            use_bonus=True
        )

        # 보너스 포함 시 +1 차원
        assert trainer.feature_dim == FEATURE_DIM_BASIC + 1

    def test_init_legacy_extended_features(self, mock_datasource):
        """레거시 use_extended_features 파라미터"""
        trainer = UnifiedTrainer(
            mock_datasource,
            use_extended_features=True
        )

        assert trainer.feature_mode == FeatureMode.EXTENDED

    def test_custom_seq_length(self, mock_datasource):
        """커스텀 시퀀스 길이"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=10
        )

        assert trainer.seq_length == 10


class TestPrepareSequences:
    """시퀀스 생성 테스트"""

    def test_prepare_sequences_basic(self, mock_datasource):
        """Basic 모드 시퀀스 생성"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5,
            feature_mode=FeatureMode.BASIC
        )

        X, y = trainer.prepare_sequences()

        # mock_datasource에 30개 레코드, seq_length=5
        # 5번째부터 타겟 가능 → 25개 샘플
        assert X.shape[0] == 25
        assert X.shape[1] == 5  # seq_length
        assert X.shape[2] == FEATURE_DIM_BASIC
        assert y.shape == (25, 45)  # multi-hot

    def test_prepare_sequences_extended(self, mock_datasource):
        """Extended 모드 시퀀스 생성"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5,
            feature_mode=FeatureMode.EXTENDED
        )

        X, y = trainer.prepare_sequences()

        assert X.shape[2] == FEATURE_DIM_EXTENDED

    def test_prepare_sequences_cached(self, mock_datasource):
        """시퀀스 캐싱"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        X1, y1 = trainer.prepare_sequences()
        X2, y2 = trainer.prepare_sequences()

        # 같은 객체 반환 (캐시)
        assert X1 is X2
        assert y1 is y2


class TestTrainTestSplit:
    """학습/테스트 분할 테스트"""

    def test_split_ratio(self, mock_datasource):
        """분할 비율"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        X_train, y_train, X_test, y_test = trainer.get_train_test_split(test_ratio=0.2)

        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.1  # 약 20%

    def test_split_shapes(self, mock_datasource):
        """분할 데이터 형태"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        X_train, y_train, X_test, y_test = trainer.get_train_test_split()

        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5
        assert y_train.shape[1] == 45
        assert y_test.shape[1] == 45


class TestGetLatestSequence:
    """최신 시퀀스 테스트"""

    def test_latest_sequence_shape(self, mock_datasource):
        """최신 시퀀스 형태"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5,
            feature_mode=FeatureMode.BASIC
        )

        latest = trainer.get_latest_sequence()

        assert latest.shape == (1, 5, FEATURE_DIM_BASIC)

    def test_latest_sequence_extended(self, mock_datasource):
        """확장 피처 최신 시퀀스"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5,
            feature_mode=FeatureMode.EXTENDED
        )

        latest = trainer.get_latest_sequence()

        assert latest.shape == (1, 5, FEATURE_DIM_EXTENDED)


class TestFeatureConversion:
    """피처 변환 테스트"""

    def test_numbers_to_multihot(self, mock_datasource):
        """multi-hot 변환"""
        trainer = UnifiedTrainer(mock_datasource)

        multihot = trainer.numbers_to_multihot([1, 10, 20, 30, 40, 45])

        assert multihot.shape == (45,)
        assert multihot.sum() == 6
        assert multihot[0] == 1  # 1번
        assert multihot[9] == 1  # 10번
        assert multihot[44] == 1  # 45번

    def test_numbers_to_features_basic(self, mock_datasource):
        """Basic 피처 변환"""
        trainer = UnifiedTrainer(
            mock_datasource,
            feature_mode=FeatureMode.BASIC
        )

        features = trainer.numbers_to_features([1, 10, 20, 30, 40, 45])

        assert features.shape == (FEATURE_DIM_BASIC,)

    def test_numbers_to_features_extended(self, mock_datasource):
        """Extended 피처 변환"""
        trainer = UnifiedTrainer(
            mock_datasource,
            feature_mode=FeatureMode.EXTENDED
        )

        features = trainer.numbers_to_features(
            [1, 10, 20, 30, 40, 45],
            prev_numbers=[2, 11, 21, 31, 41, 44],
            round_num=100
        )

        assert features.shape == (FEATURE_DIM_EXTENDED,)


class TestTrainModel:
    """모델 학습 테스트"""

    def test_train_model(self, mock_datasource, mock_model):
        """모델 학습"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        history = trainer.train_model(
            mock_model,
            epochs=2,
            test_ratio=0.2,
            verbose=False
        )

        assert mock_model.is_trained is True
        assert history.epochs == 2

    def test_create_and_train(self, mock_datasource):
        """모델 생성 및 학습"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        # MockModel 사용을 위해 팩토리에 등록
        from models.factory import ModelFactory
        from tests.conftest import MockModel
        ModelFactory.register('test_mock', MockModel)

        try:
            model, history = trainer.create_and_train(
                'test_mock',
                epochs=2,
                verbose=False
            )

            assert model.is_trained is True
            assert history.epochs == 2
        finally:
            # 정리
            if 'test_mock' in ModelFactory._registry:
                del ModelFactory._registry['test_mock']


class TestDataTypes:
    """데이터 타입 테스트"""

    def test_sequence_dtype(self, mock_datasource):
        """시퀀스 데이터 타입"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        X, y = trainer.prepare_sequences()

        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_latest_sequence_dtype(self, mock_datasource):
        """최신 시퀀스 데이터 타입"""
        trainer = UnifiedTrainer(
            mock_datasource,
            seq_length=5
        )

        latest = trainer.get_latest_sequence()

        assert latest.dtype == np.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
