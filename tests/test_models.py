"""모델 및 팩토리 테스트"""
import os
import sys
import pytest
import numpy as np
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import ModelFactory
from core.types import ProbabilityDistribution, TrainingHistory


class TestModelFactory:
    """ModelFactory 테스트"""

    def test_list_models(self):
        """등록된 모델 목록 확인"""
        models = ModelFactory.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1  # 최소 1개 모델 등록

    def test_create_registered_model(self):
        """등록된 모델 생성"""
        models = ModelFactory.list_models()
        if models:
            model = ModelFactory.create(models[0], {'input_dim': 74})
            assert model is not None
            assert hasattr(model, 'model_type')

    def test_create_unknown_model_raises_error(self):
        """미등록 모델 생성 시 오류"""
        with pytest.raises(ValueError) as excinfo:
            ModelFactory.create('unknown_model_xyz')
        assert 'Unknown model' in str(excinfo.value)

    def test_get_class(self):
        """모델 클래스 조회"""
        models = ModelFactory.list_models()
        if models:
            cls = ModelFactory.get_class(models[0])
            assert cls is not None


class TestGRUModel:
    """GRU 모델 테스트"""

    @pytest.fixture
    def gru_model(self):
        """GRU 모델 생성"""
        try:
            return ModelFactory.create('gru', {
                'input_dim': 74,
                'd_model': 32,
                'num_layers': 1,
                'dropout': 0.1
            })
        except ValueError:
            pytest.skip("GRU model not registered")

    def test_requires_sequence(self, gru_model):
        """시퀀스 입력 필요 여부"""
        assert gru_model.requires_sequence is True

    def test_predict_proba(self, gru_model, sample_single_X):
        """확률 분포 예측"""
        proba = gru_model.predict_proba(sample_single_X)

        assert isinstance(proba, ProbabilityDistribution)
        assert proba.probabilities.shape == (45,)
        assert np.all(proba.probabilities >= 0)
        assert np.all(proba.probabilities <= 1)

    def test_predict_numbers(self, gru_model, sample_single_X):
        """번호 예측"""
        predictions = gru_model.predict_numbers(sample_single_X, num_sets=3)

        assert len(predictions) == 3
        for pred in predictions:
            assert len(pred.numbers) == 6
            assert all(1 <= n <= 45 for n in pred.numbers)
            assert len(set(pred.numbers)) == 6  # 중복 없음

    def test_save_load(self, gru_model, sample_X_sequence, sample_y, tmp_path):
        """모델 저장 및 로드"""
        # 간단한 학습
        gru_model.train(sample_X_sequence, sample_y, epochs=2)

        # 저장
        model_path = str(tmp_path / "gru_test.pt")
        gru_model.save(model_path)
        assert os.path.exists(model_path)

        # 새 모델에 로드 - 같은 설정으로 생성해야 함
        new_model = ModelFactory.create('gru', {
            'input_dim': 74,
            'd_model': 32,
            'num_layers': 1,
            'dropout': 0.1
        })
        new_model.load(model_path)
        assert new_model.is_trained is True


class TestTransformerModel:
    """Transformer 모델 테스트"""

    @pytest.fixture
    def transformer_model(self):
        """Transformer 모델 생성"""
        try:
            return ModelFactory.create('transformer', {
                'input_dim': 74,
                'd_model': 32,
                'nhead': 2,
                'num_layers': 1
            })
        except ValueError:
            pytest.skip("Transformer model not registered")

    def test_requires_sequence(self, transformer_model):
        """시퀀스 입력 필요 여부"""
        assert transformer_model.requires_sequence is True

    def test_predict_proba(self, transformer_model, sample_single_X):
        """확률 분포 예측"""
        proba = transformer_model.predict_proba(sample_single_X)

        assert isinstance(proba, ProbabilityDistribution)
        assert proba.probabilities.shape == (45,)

    def test_load_rebuilds_from_saved_config(self, sample_X_sequence, sample_y, tmp_path):
        """다른 config로 생성한 모델이 load 시 저장된 config로 재구성"""
        try:
            # seq_length=10으로 학습 및 저장
            model = ModelFactory.create('transformer', {
                'input_dim': 74, 'd_model': 32, 'nhead': 2,
                'num_layers': 1, 'seq_length': 10,
            })
        except ValueError:
            pytest.skip("Transformer model not registered")

        X_short = sample_X_sequence[:, :10, :]  # seq_length=10
        model.train(X_short, sample_y, epochs=2, verbose=False)

        path = str(tmp_path / "transformer_mismatch.pt")
        model.save(path)

        # 기본 config (seq_length=20)로 생성 후 로드 → 자동 재구성
        model2 = ModelFactory.create('transformer', {
            'input_dim': 45, 'd_model': 32, 'nhead': 2, 'num_layers': 1,
        })
        model2.load(path)

        assert model2.seq_length == 10
        assert model2.input_dim == 74

        import numpy as np
        test_X = np.random.rand(1, 10, 74).astype(np.float32)
        p1 = model.predict_proba(test_X).probabilities
        p2 = model2.predict_proba(test_X).probabilities
        np.testing.assert_allclose(p1, p2, atol=1e-6)


class TestLSTMModel:
    """LSTM 모델 테스트"""

    @pytest.fixture
    def lstm_model(self):
        """LSTM 모델 생성"""
        try:
            return ModelFactory.create('lstm', {
                'input_dim': 74,
                'd_model': 32,
                'num_layers': 1
            })
        except ValueError:
            pytest.skip("LSTM model not registered")

    def test_requires_sequence(self, lstm_model):
        """시퀀스 입력 필요 여부"""
        assert lstm_model.requires_sequence is True


class TestRandomForestModel:
    """RandomForest 모델 테스트"""

    @pytest.fixture
    def rf_model(self):
        """RandomForest 모델 생성"""
        try:
            return ModelFactory.create('random_forest', {
                'n_estimators': 10,
                'max_depth': 5
            })
        except ValueError:
            pytest.skip("RandomForest model not registered")

    def test_requires_sequence(self, rf_model):
        """시퀀스 입력 불필요"""
        assert rf_model.requires_sequence is False

    def test_train_and_predict(self, rf_model, sample_X_flat, sample_y):
        """학습 및 예측"""
        history = rf_model.train(sample_X_flat, sample_y, epochs=1)
        assert rf_model.is_trained is True

        # 예측
        proba = rf_model.predict_proba(sample_X_flat[:1])
        assert proba.probabilities.shape == (45,)

    def test_save_load(self, rf_model, sample_X_flat, sample_y, tmp_path):
        """모델 저장 및 로드"""
        rf_model.train(sample_X_flat, sample_y, epochs=1)

        model_path = str(tmp_path / "rf_test.pkl")
        rf_model.save(model_path)
        assert os.path.exists(model_path)

        new_model = ModelFactory.create('random_forest')
        new_model.load(model_path)
        assert new_model.is_trained is True


class TestMarkovModel:
    """Markov 모델 테스트"""

    @pytest.fixture
    def markov_model(self):
        """Markov 모델 생성"""
        try:
            return ModelFactory.create('markov', {})
        except ValueError:
            pytest.skip("Markov model not registered")

    def test_requires_sequence(self, markov_model):
        """시퀀스 입력 필요 (전이 계산)"""
        assert markov_model.requires_sequence is True


class TestXGBoostModel:
    """XGBoost 모델 테스트"""

    @pytest.fixture
    def xgb_model(self):
        """XGBoost 모델 생성"""
        try:
            return ModelFactory.create('xgboost', {
                'n_estimators': 10,
                'max_depth': 3
            })
        except (ValueError, ImportError):
            pytest.skip("XGBoost not available")

    def test_requires_sequence(self, xgb_model):
        """시퀀스 입력 불필요"""
        assert xgb_model.requires_sequence is False

    def test_train_and_predict(self, xgb_model, sample_X_flat, sample_y):
        """학습 및 예측"""
        history = xgb_model.train(sample_X_flat, sample_y, epochs=1)
        assert xgb_model.is_trained is True

        proba = xgb_model.predict_proba(sample_X_flat[:1])
        assert proba.probabilities.shape == (45,)


class TestProbabilityDistribution:
    """ProbabilityDistribution 테스트"""

    def test_top_k(self, sample_proba):
        """상위 k개 번호 추출"""
        top6 = sample_proba.top_k(6)

        assert len(top6) == 6
        assert all(1 <= n <= 45 for n in top6)
        assert len(set(top6)) == 6  # 중복 없음

    def test_get_prob(self, sample_proba):
        """특정 번호 확률 조회"""
        prob = sample_proba.get_prob(1)

        assert 0 <= prob <= 1

    def test_normalize(self, sample_proba):
        """정규화"""
        normalized = sample_proba.normalize()

        assert abs(normalized.probabilities.sum() - 1.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
