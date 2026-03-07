"""CNN Grid 모델 단위 테스트"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.cnn_grid import CNNGridModel, LottoCNNGridNet, _to_grid


# ============================================================
# 그리드 변환
# ============================================================

class TestToGrid:

    def test_basic_shape(self):
        """기본 45차원 → (batch, seq, 7, 7)"""
        X = np.random.rand(4, 20, 45).astype(np.float32)
        grids, aux = _to_grid(X, feature_dim=45)
        assert grids.shape == (4, 20, 7, 7)
        assert aux is None

    def test_extended_shape(self):
        """확장 74차원 → grids + aux"""
        X = np.random.rand(4, 20, 74).astype(np.float32)
        grids, aux = _to_grid(X, feature_dim=74)
        assert grids.shape == (4, 20, 7, 7)
        assert aux.shape == (4, 29)

    def test_different_seq_length(self):
        """seq_length=10도 정상 동작"""
        X = np.random.rand(2, 10, 45).astype(np.float32)
        grids, aux = _to_grid(X, feature_dim=45)
        assert grids.shape == (2, 10, 7, 7)

    def test_grid_values_binary(self):
        """multi-hot 입력이면 그리드도 0/1"""
        X = np.zeros((1, 5, 45), dtype=np.float32)
        X[0, 0, [0, 9, 19, 29, 39, 44]] = 1.0
        grids, _ = _to_grid(X, feature_dim=45)
        # 45개 + 4개 패딩 = 49개 중 6개만 1
        assert grids[0, 0].sum() == 6.0
        # 패딩 영역 (마지막 행 뒤 4칸) 은 0
        flat = grids[0, 0].flatten()
        assert flat[45:].sum() == 0.0


# ============================================================
# 네트워크 forward
# ============================================================

class TestLottoCNNGridNet:

    def test_forward_basic(self):
        net = LottoCNNGridNet(seq_length=20, aux_dim=0)
        grid = torch.randn(2, 20, 7, 7)
        out = net(grid)
        assert out.shape == (2, 45)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_with_aux(self):
        net = LottoCNNGridNet(seq_length=20, aux_dim=29)
        grid = torch.randn(2, 20, 7, 7)
        aux = torch.randn(2, 29)
        out = net(grid, aux)
        assert out.shape == (2, 45)

    def test_different_seq_length(self):
        net = LottoCNNGridNet(seq_length=10, aux_dim=0)
        grid = torch.randn(3, 10, 7, 7)
        out = net(grid)
        assert out.shape == (3, 45)


# ============================================================
# CNNGridModel wrapper
# ============================================================

class TestCNNGridModel:

    @pytest.fixture
    def model_basic(self):
        return CNNGridModel({'input_dim': 45, 'seq_length': 20})

    @pytest.fixture
    def model_extended(self):
        return CNNGridModel({'input_dim': 74, 'seq_length': 20})

    def test_predict_proba_shape(self, model_basic):
        X = np.random.rand(1, 20, 45).astype(np.float32)
        proba = model_basic.predict_proba(X)
        assert proba.probabilities.shape == (45,)
        assert all(0 <= p <= 1 for p in proba.probabilities)

    def test_predict_proba_extended(self, model_extended):
        X = np.random.rand(1, 20, 74).astype(np.float32)
        proba = model_extended.predict_proba(X)
        assert proba.probabilities.shape == (45,)

    def test_requires_sequence(self, model_basic):
        assert model_basic.requires_sequence is True

    def test_model_type(self, model_basic):
        assert model_basic.model_type == "cnn_grid"


# ============================================================
# Train / Save / Load 라운드트립
# ============================================================

class TestTrainSaveLoad:

    def test_train_smoke(self):
        """학습이 에러 없이 완료되는지 확인"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        X = np.random.rand(8, 10, 45).astype(np.float32)
        y = np.zeros((8, 45), dtype=np.float32)
        for i in range(8):
            idx = np.random.choice(45, 6, replace=False)
            y[i, idx] = 1.0

        history = model.train(X, y, epochs=3, verbose=False)
        assert model.is_trained
        assert len(history.train_losses) == 3

    def test_save_load_roundtrip(self, tmp_path):
        """저장 → 로드 후 동일 예측"""
        model = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        X = np.random.rand(8, 10, 74).astype(np.float32)
        y = np.zeros((8, 45), dtype=np.float32)
        for i in range(8):
            idx = np.random.choice(45, 6, replace=False)
            y[i, idx] = 1.0

        model.train(X, y, epochs=3, verbose=False)

        path = str(tmp_path / "cnn_grid_test.pt")
        model.save(path)

        # 새 모델로 로드 (같은 config)
        model2 = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        model2.load(path)

        test_X = np.random.rand(1, 10, 74).astype(np.float32)
        p1 = model.predict_proba(test_X).probabilities
        p2 = model2.predict_proba(test_X).probabilities
        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_load_rebuilds_from_saved_config(self, tmp_path):
        """다른 config로 생성한 모델이 load 시 저장된 config로 재구성"""
        # seq_length=10으로 학습 및 저장
        model = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        X = np.random.rand(8, 10, 74).astype(np.float32)
        y = np.zeros((8, 45), dtype=np.float32)
        for i in range(8):
            idx = np.random.choice(45, 6, replace=False)
            y[i, idx] = 1.0

        model.train(X, y, epochs=3, verbose=False)

        path = str(tmp_path / "cnn_grid_mismatch.pt")
        model.save(path)

        # 기본 config (seq_length=20)로 생성 후 로드 → 자동 재구성
        model2 = CNNGridModel({'input_dim': 45})
        model2.load(path)

        assert model2.seq_length == 10
        assert model2.input_dim == 74

        test_X = np.random.rand(1, 10, 74).astype(np.float32)
        p1 = model.predict_proba(test_X).probabilities
        p2 = model2.predict_proba(test_X).probabilities
        np.testing.assert_allclose(p1, p2, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
