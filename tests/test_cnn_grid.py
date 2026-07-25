"""CNN Grid 모델 단위 테스트 (V1 + V2)"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.cnn_grid import (
    CNNGridModel, LottoCNNGridNet, LottoCNNGridNetV1, _to_grid,
    FocalLoss, ChannelAttention, SpatialAttention, CBAM, TemporalAttention,
)


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
# V2 컴포넌트 테스트
# ============================================================

class TestFocalLoss:

    def test_output_scalar(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.sigmoid(torch.randn(4, 45))
        target = torch.zeros(4, 45)
        for i in range(4):
            idx = torch.randperm(45)[:6]
            target[i, idx] = 1.0
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        """완벽한 예측 → 매우 낮은 loss"""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        target = torch.zeros(2, 45)
        target[0, [0, 1, 2, 3, 4, 5]] = 1.0
        target[1, [10, 11, 12, 13, 14, 15]] = 1.0
        # near-perfect prediction
        pred = target * 0.99 + (1 - target) * 0.01
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        loss_fn = FocalLoss()
        logits = torch.randn(2, 45, requires_grad=True)
        pred = torch.sigmoid(logits)
        target = torch.zeros(2, 45)
        target[0, :6] = 1.0
        target[1, 6:12] = 1.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert logits.grad is not None

    def test_default_alpha_weights_positive_higher(self):
        """기본 alpha=0.75 → positive miss 1개의 loss > negative miss 1개의 loss"""
        assert FocalLoss().alpha == 0.75
        loss_fn = FocalLoss(alpha=0.75, gamma=0.0)  # gamma=0으로 focal 효과 제거
        # 동일 pred=0.5에서 positive miss vs negative miss 각각 1개씩 비교
        pred = torch.full((1, 1), 0.5)
        pos_loss = loss_fn(pred, torch.ones(1, 1))   # positive miss: alpha=0.75
        neg_loss = loss_fn(pred, torch.zeros(1, 1))   # negative miss: alpha=0.25
        assert pos_loss.item() > neg_loss.item()
        assert pos_loss.item() / neg_loss.item() == pytest.approx(3.0, rel=1e-5)


class TestChannelAttention:

    def test_shape_preserved(self):
        ca = ChannelAttention(32, reduction=4)
        x = torch.randn(2, 32, 7, 7)
        out = ca(x)
        assert out.shape == (2, 32, 7, 7)

    def test_output_range(self):
        ca = ChannelAttention(16, reduction=4)
        x = torch.ones(1, 16, 7, 7)
        out = ca(x)
        # Attention scales output; should not be all zeros
        assert out.abs().sum() > 0


class TestSpatialAttention:

    def test_shape_preserved(self):
        sa = SpatialAttention(kernel_size=3)
        x = torch.randn(2, 32, 7, 7)
        out = sa(x)
        assert out.shape == (2, 32, 7, 7)


class TestCBAM:

    def test_shape_preserved(self):
        cbam = CBAM(64, reduction=4)
        x = torch.randn(2, 64, 7, 7)
        out = cbam(x)
        assert out.shape == (2, 64, 7, 7)

    def test_gradient_flows(self):
        cbam = CBAM(32, reduction=4)
        x = torch.randn(2, 32, 7, 7, requires_grad=True)
        out = cbam(x)
        out.sum().backward()
        assert x.grad is not None


class TestTemporalAttention:

    def test_shape_preserved(self):
        ta = TemporalAttention(seq_length=20)
        x = torch.randn(2, 20, 7, 7)
        out = ta(x)
        assert out.shape == (2, 20, 7, 7)

    def test_recency_bias_init(self):
        """초기 가중치가 최근 회차에 높은지 확인"""
        ta = TemporalAttention(seq_length=10)
        attn = torch.softmax(ta.temporal_weights, dim=0)
        # 마지막(최근) > 처음(과거)
        assert attn[-1].item() > attn[0].item()

    def test_weights_learnable(self):
        ta = TemporalAttention(seq_length=5)
        x = torch.randn(2, 5, 7, 7)
        out = ta(x)
        loss = out.sum()
        loss.backward()
        assert ta.temporal_weights.grad is not None


# ============================================================
# V2 네트워크 forward
# ============================================================

class TestLottoCNNGridNetV2:

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

    def test_all_features_disabled(self):
        """모든 v2 기능 비활성화 → 여전히 정상 동작"""
        net = LottoCNNGridNet(
            seq_length=20, aux_dim=0,
            use_cbam=False, use_residual=False, use_temporal_attention=False,
        )
        grid = torch.randn(2, 20, 7, 7)
        out = net(grid)
        assert out.shape == (2, 45)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient_flows_full(self):
        """전체 네트워크 gradient flow 확인"""
        net = LottoCNNGridNet(seq_length=10, aux_dim=29)
        grid = torch.randn(2, 10, 7, 7, requires_grad=True)
        aux = torch.randn(2, 29, requires_grad=True)
        out = net(grid, aux)
        loss = out.sum()
        loss.backward()
        assert grid.grad is not None
        assert aux.grad is not None


# ============================================================
# V1 네트워크 (호환성)
# ============================================================

class TestLottoCNNGridNetV1:

    def test_forward_basic(self):
        net = LottoCNNGridNetV1(seq_length=20, aux_dim=0)
        grid = torch.randn(2, 20, 7, 7)
        out = net(grid)
        assert out.shape == (2, 45)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_with_aux(self):
        net = LottoCNNGridNetV1(seq_length=20, aux_dim=29)
        grid = torch.randn(2, 20, 7, 7)
        aux = torch.randn(2, 29)
        out = net(grid, aux)
        assert out.shape == (2, 45)


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

    def test_v2_features_default_enabled(self, model_basic):
        assert model_basic.use_cbam is True
        assert model_basic.use_residual is True
        assert model_basic.use_temporal_attention is True
        assert model_basic.use_focal_loss is True

    def test_v2_features_disabled(self):
        model = CNNGridModel({
            'input_dim': 45,
            'use_cbam': False,
            'use_residual': False,
            'use_temporal_attention': False,
            'use_focal_loss': False,
        })
        X = np.random.rand(1, 20, 45).astype(np.float32)
        proba = model.predict_proba(X)
        assert proba.probabilities.shape == (45,)


# ============================================================
# Train / Save / Load 라운드트립
# ============================================================

class TestTrainSaveLoad:

    def _make_data(self, n, seq_len, feat_dim):
        X = np.random.rand(n, seq_len, feat_dim).astype(np.float32)
        y = np.zeros((n, 45), dtype=np.float32)
        for i in range(n):
            idx = np.random.choice(45, 6, replace=False)
            y[i, idx] = 1.0
        return X, y

    def test_train_smoke(self):
        """학습이 에러 없이 완료되는지 확인"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        X, y = self._make_data(8, 10, 45)
        history = model.train(X, y, epochs=3, verbose=False)
        assert model.is_trained
        assert len(history.train_losses) == 3

    def test_train_with_focal_loss(self):
        """Focal Loss로 학습"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10, 'use_focal_loss': True})
        X, y = self._make_data(8, 10, 45)
        history = model.train(X, y, epochs=3, verbose=False)
        assert model.is_trained

    def test_train_with_bce_loss(self):
        """BCE Loss로 학습 (focal loss off)"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10, 'use_focal_loss': False})
        X, y = self._make_data(8, 10, 45)
        history = model.train(X, y, epochs=3, verbose=False)
        assert model.is_trained

    def test_train_batch1_remainder_no_crash(self):
        """n=33, batch_size=32 → 마지막 배치 1개여도 LayerNorm이라 크래시 안 남"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        X, y = self._make_data(33, 10, 45)
        history = model.train(X, y, epochs=3, batch_size=32, verbose=False)
        assert model.is_trained

    def test_train_batch_size_1_no_crash(self):
        """batch_size=1에서도 LayerNorm 덕분에 크래시 안 남"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        X, y = self._make_data(4, 10, 45)
        history = model.train(X, y, epochs=3, batch_size=1, verbose=False)
        assert model.is_trained

    def test_save_load_roundtrip(self, tmp_path):
        """저장 → 로드 후 동일 예측"""
        model = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        X, y = self._make_data(8, 10, 74)
        model.train(X, y, epochs=3, verbose=False)

        path = str(tmp_path / "cnn_grid_test.pt")
        model.save(path)

        model2 = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        model2.load(path)

        test_X = np.random.rand(1, 10, 74).astype(np.float32)
        p1 = model.predict_proba(test_X).probabilities
        p2 = model2.predict_proba(test_X).probabilities
        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_save_load_preserves_version(self, tmp_path):
        """V2 저장 → 로드 시 model_version=2 유지"""
        model = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        X, y = self._make_data(8, 10, 45)
        model.train(X, y, epochs=3, verbose=False)

        path = str(tmp_path / "v2.pt")
        model.save(path)

        model2 = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        model2.load(path)
        assert model2._model_version == 2

    def test_load_v1_checkpoint(self, tmp_path):
        """V1 형식 checkpoint를 로드할 수 있는지 확인"""
        # V1 모델 직접 생성 및 저장
        v1_net = LottoCNNGridNetV1(seq_length=10, aux_dim=29, output_dim=45)
        path = str(tmp_path / "v1.pt")
        torch.save({
            'model_state_dict': v1_net.state_dict(),
            'config': {'input_dim': 74, 'seq_length': 10, 'output_dim': 45, 'dropout': 0.3},
            'model_type': 'cnn_grid',
            # model_version 키가 없으면 V1으로 판단
        }, path)

        model = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        model.load(path)
        assert model._model_version == 1
        assert model.is_trained

        # 예측도 정상 동작
        test_X = np.random.rand(1, 10, 74).astype(np.float32)
        proba = model.predict_proba(test_X)
        assert proba.probabilities.shape == (45,)

    def test_load_old_v2_batchnorm_checkpoint(self, tmp_path):
        """구 v2 체크포인트(BatchNorm1d 키)를 LayerNorm 모델로 로드"""
        # BatchNorm1d 기반 v2 네트워크 시뮬레이션
        import torch.nn as nn
        class OldV2Net(nn.Module):
            """bn1/bn2 키를 가진 구 v2 구조 (FC head만 재현)"""
            def __init__(self):
                super().__init__()
                # 실제 구 v2와 동일한 구조로 state_dict 키 생성
                self.temporal_attn = TemporalAttention(10)
                self.branch_3x3_conv1 = nn.Sequential(nn.Conv2d(10, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.branch_3x3_conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.branch_3x3_skip = nn.Conv2d(10, 32, 1)
                self.branch_3x3_cbam = CBAM(32, 4)
                self.branch_5x5_conv1 = nn.Sequential(nn.Conv2d(10, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU())
                self.branch_5x5_conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
                self.branch_5x5_skip = nn.Conv2d(10, 32, 1)
                self.branch_5x5_cbam = CBAM(32, 4)
                self.branch_1x1_conv1 = nn.Sequential(nn.Conv2d(10, 16, 1), nn.BatchNorm2d(16), nn.ReLU())
                self.branch_1x1_conv2 = nn.Sequential(nn.Conv2d(16, 16, 1), nn.BatchNorm2d(16), nn.ReLU())
                self.branch_1x1_skip = nn.Conv2d(10, 16, 1)
                self.branch_1x1_cbam = CBAM(16, 4)
                self.fusion_conv = nn.Sequential(nn.Conv2d(80, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.fusion_skip = nn.Conv2d(80, 64, 1)
                self.fusion_cbam = CBAM(64, 4)
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Linear(64, 128)
                self.bn1 = nn.BatchNorm1d(128)  # old key
                self.fc2 = nn.Linear(128, 128)
                self.bn2 = nn.BatchNorm1d(128)  # old key
                self.fc_out = nn.Linear(128, 45)
                self.dropout = nn.Dropout(0.3)
                self.fc_skip = nn.Linear(64, 128)

        old_net = OldV2Net()
        path = str(tmp_path / "old_v2.pt")
        torch.save({
            'model_state_dict': old_net.state_dict(),
            'config': {'input_dim': 45, 'seq_length': 10, 'output_dim': 45, 'dropout': 0.3},
            'model_type': 'cnn_grid',
            'model_version': 2,
        }, path)

        model = CNNGridModel({'input_dim': 45, 'seq_length': 10})
        model.load(path)
        assert model.is_trained
        assert model._model_version == 2

        test_X = np.random.rand(1, 10, 45).astype(np.float32)
        proba = model.predict_proba(test_X)
        assert proba.probabilities.shape == (45,)

    def test_load_rebuilds_from_saved_config(self, tmp_path):
        """다른 config로 생성한 모델이 load 시 저장된 config로 재구성"""
        model = CNNGridModel({'input_dim': 74, 'seq_length': 10})
        X, y = self._make_data(8, 10, 74)
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
