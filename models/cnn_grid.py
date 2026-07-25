"""CNN Grid 모델 - 7x7 이미지 패턴 기반 로또 예측 (v2: CBAM, Residual, Focal Loss, Temporal Attention)"""
import copy
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory


def _to_grid(X: np.ndarray, feature_dim: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    시퀀스 데이터를 7x7 그리드로 변환

    Args:
        X: (batch, seq_length, feature_dim) 시퀀스 데이터
        feature_dim: 피처 차원 (45 또는 74)

    Returns:
        grids: (batch, seq_length, 7, 7) 바이너리 그리드
        aux: (batch, 29) 확장 피처 또는 None
    """
    batch_size, seq_length = X.shape[0], X.shape[1]

    # multi-hot 45개 추출
    multihot = X[:, :, :45]  # (batch, seq, 45)

    # 4개 zero pad → 49개 → reshape (7, 7)
    pad = np.zeros((batch_size, seq_length, 4), dtype=multihot.dtype)
    padded = np.concatenate([multihot, pad], axis=2)  # (batch, seq, 49)
    grids = padded.reshape(batch_size, seq_length, 7, 7)

    # 확장 피처 (마지막 타임스텝)
    aux = None
    if feature_dim > 45:
        aux = X[:, -1, 45:]  # (batch, feature_dim - 45)

    return grids, aux


# ============================================================
# V1 네트워크 (기존 모델 호환용)
# ============================================================

class LottoCNNGridNetV1(nn.Module):
    """V1 멀티 브랜치 CNN - 기존 저장 모델 로드용"""

    def __init__(self, seq_length=20, aux_dim=0, output_dim=45, dropout=0.3):
        super().__init__()

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(seq_length, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(seq_length, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(seq_length, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(80, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        fc_input_dim = 64 + (aux_dim if aux_dim > 0 else 0)

        if aux_dim > 0:
            self.aux_fc = nn.Sequential(
                nn.Linear(aux_dim, 32),
                nn.ReLU(),
            )
            fc_input_dim = 64 + 32

        self.aux_dim = aux_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, grid, aux=None):
        b1 = self.branch_3x3(grid)
        b2 = self.branch_5x5(grid)
        b3 = self.branch_1x1(grid)

        merged = torch.cat([b1, b2, b3], dim=1)
        fused = self.fusion(merged)
        pooled = self.global_pool(fused)
        features = pooled.view(pooled.size(0), -1)

        if self.aux_dim > 0 and aux is not None:
            aux_features = self.aux_fc(aux)
            features = torch.cat([features, aux_features], dim=1)

        return self.fc(features)


# ============================================================
# V2 컴포넌트: Focal Loss, CBAM, Temporal Attention
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss - class imbalance 해결 (6/45 = 13.3% positive rate)"""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class ChannelAttention(nn.Module):
    """Channel Attention: 어떤 feature channel이 중요한지 학습"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_pool = x.mean(dim=(2, 3))  # (B, C)
        max_pool = x.amax(dim=(2, 3))  # (B, C)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial Attention: 그리드의 어느 위치가 중요한지 학습"""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_pool = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        spatial_map = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(spatial_map))  # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module = Channel + Spatial Attention"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class TemporalAttention(nn.Module):
    """시간 차원 어텐션 - 20개 타임스텝에 learnable weight 적용"""

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        # Recency bias 초기화: 최근 회차에 더 높은 가중치
        init_weights = torch.linspace(-1.0, 1.0, seq_length)
        self.temporal_weights = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_length, 7, 7)
        attn = torch.softmax(self.temporal_weights, dim=0)  # (seq_length,)
        # 각 타임스텝에 가중치 적용
        return x * attn.view(1, -1, 1, 1)


# ============================================================
# V2 네트워크 (CBAM + Residual + Temporal Attention + Improved FC)
# ============================================================

class LottoCNNGridNet(nn.Module):
    """V2 멀티 브랜치 CNN - CBAM, Residual, Temporal Attention, Improved FC"""

    def __init__(self, seq_length=20, aux_dim=0, output_dim=45, dropout=0.3,
                 use_cbam=True, cbam_reduction=4,
                 use_residual=True,
                 use_temporal_attention=True):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_residual = use_residual
        self.use_temporal_attention = use_temporal_attention

        # Temporal Attention
        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(seq_length)

        # Branch 1: 3x3 - 인접 번호 패턴 (2 conv layers)
        self.branch_3x3_conv1 = nn.Sequential(
            nn.Conv2d(seq_length, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.branch_3x3_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        if use_residual:
            self.branch_3x3_skip = nn.Conv2d(seq_length, 32, kernel_size=1)
        if use_cbam:
            self.branch_3x3_cbam = CBAM(32, cbam_reduction)

        # Branch 2: 5x5 - 행/열/대각선 패턴 (2 conv layers for deeper)
        self.branch_5x5_conv1 = nn.Sequential(
            nn.Conv2d(seq_length, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.branch_5x5_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        if use_residual:
            self.branch_5x5_skip = nn.Conv2d(seq_length, 32, kernel_size=1)
        if use_cbam:
            self.branch_5x5_cbam = CBAM(32, cbam_reduction)

        # Branch 3: 1x1 - 셀별 시간 패턴 (2 conv layers for deeper)
        self.branch_1x1_conv1 = nn.Sequential(
            nn.Conv2d(seq_length, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.branch_1x1_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        if use_residual:
            self.branch_1x1_skip = nn.Conv2d(seq_length, 16, kernel_size=1)
        if use_cbam:
            self.branch_1x1_cbam = CBAM(16, cbam_reduction)

        # Fusion: 32+32+16=80 → 64 with residual + CBAM
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(80, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        if use_residual:
            self.fusion_skip = nn.Conv2d(80, 64, kernel_size=1)
        if use_cbam:
            self.fusion_cbam = CBAM(64, cbam_reduction)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # FC layers with residual
        fc_input_dim = 64
        self.aux_dim = aux_dim

        if aux_dim > 0:
            self.aux_fc = nn.Sequential(
                nn.Linear(aux_dim, 32),
                nn.ReLU(),
            )
            fc_input_dim = 64 + 32

        # Improved FC Head: 2 layers with residual + LayerNorm
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc_out = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)

        if use_residual and fc_input_dim != 128:
            self.fc_skip = nn.Linear(fc_input_dim, 128)

    def _branch_forward(self, x, conv1, conv2, skip=None, cbam=None):
        """Branch forward with optional residual + CBAM"""
        out = conv1(x)
        out = conv2(out)
        if self.use_residual and skip is not None:
            out = out + skip(x)
        if self.use_cbam and cbam is not None:
            out = cbam(out)
        return out

    def forward(self, grid, aux=None):
        """
        Args:
            grid: (batch, seq_length, 7, 7)
            aux: (batch, aux_dim) or None
        """
        x = grid

        # Temporal Attention (before spatial processing)
        if self.use_temporal_attention:
            x = self.temporal_attn(x)

        # Multi-branch CNN with residual + CBAM
        b1 = self._branch_forward(
            x, self.branch_3x3_conv1, self.branch_3x3_conv2,
            getattr(self, 'branch_3x3_skip', None),
            getattr(self, 'branch_3x3_cbam', None),
        )
        b2 = self._branch_forward(
            x, self.branch_5x5_conv1, self.branch_5x5_conv2,
            getattr(self, 'branch_5x5_skip', None),
            getattr(self, 'branch_5x5_cbam', None),
        )
        b3 = self._branch_forward(
            x, self.branch_1x1_conv1, self.branch_1x1_conv2,
            getattr(self, 'branch_1x1_skip', None),
            getattr(self, 'branch_1x1_cbam', None),
        )

        # Fusion with residual + CBAM
        merged = torch.cat([b1, b2, b3], dim=1)  # (batch, 80, 7, 7)
        fused = self.fusion_conv(merged)           # (batch, 64, 7, 7)
        if self.use_residual:
            fused = fused + self.fusion_skip(merged)
        if self.use_cbam:
            fused = self.fusion_cbam(fused)

        pooled = self.global_pool(fused)            # (batch, 64, 1, 1)
        features = pooled.view(pooled.size(0), -1)  # (batch, 64)

        if self.aux_dim > 0 and aux is not None:
            aux_features = self.aux_fc(aux)
            features = torch.cat([features, aux_features], dim=1)

        # Improved FC Head with residual
        h = F.relu(self.ln1(self.fc1(features)))
        h = self.dropout(h)
        h2 = F.relu(self.ln2(self.fc2(h)))
        if self.use_residual:
            # Residual connection in FC
            if hasattr(self, 'fc_skip'):
                h2 = h2 + self.fc_skip(features)
            else:
                h2 = h2 + h
        h2 = self.dropout(h2)
        return torch.sigmoid(self.fc_out(h2))


class CNNGridModel(BaseModel):
    """CNN Grid 기반 로또 예측 모델 - 7x7 공간 패턴 학습"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "cnn_grid"

        self.input_dim = self.config.get('input_dim', 45)
        self.seq_length = self.config.get('seq_length', 20)
        self.output_dim = self.config.get('output_dim', 45)
        self.dropout = self.config.get('dropout', 0.3)

        # V2 config
        self.use_cbam = self.config.get('use_cbam', True)
        self.cbam_reduction = self.config.get('cbam_reduction', 4)
        self.use_residual = self.config.get('use_residual', True)
        self.use_temporal_attention = self.config.get('use_temporal_attention', True)
        self.use_focal_loss = self.config.get('use_focal_loss', True)
        self.focal_alpha = self.config.get('focal_alpha', 0.75)
        self.focal_gamma = self.config.get('focal_gamma', 2.0)

        self.aux_dim = max(0, self.input_dim - 45)
        self._device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self._model_version = 2  # Track which version is loaded

        self.net = LottoCNNGridNet(
            seq_length=self.seq_length,
            aux_dim=self.aux_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
            use_cbam=self.use_cbam,
            cbam_reduction=self.cbam_reduction,
            use_residual=self.use_residual,
            use_temporal_attention=self.use_temporal_attention,
        ).to(self._device)

    def _prepare_input(self, X: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """numpy 입력을 그리드 텐서로 변환"""
        if X.ndim == 2:
            X = X[np.newaxis, :]

        grids, aux = _to_grid(X, self.input_dim)
        grid_tensor = torch.FloatTensor(grids).to(self._device)
        aux_tensor = torch.FloatTensor(aux).to(self._device) if aux is not None else None
        return grid_tensor, aux_tensor

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 100,
              batch_size: int = 32,
              lr: float = 0.001,
              patience: int = 15,
              criterion: nn.Module = None,
              verbose: bool = True,
              **kwargs) -> TrainingHistory:
        """모델 학습"""
        from torch.utils.data import DataLoader, TensorDataset

        # 그리드 변환
        grids, aux = _to_grid(X, self.input_dim)
        grid_tensor = torch.FloatTensor(grids).to(self._device)
        y_tensor = torch.FloatTensor(y).to(self._device)

        if aux is not None:
            aux_tensor = torch.FloatTensor(aux).to(self._device)
            train_dataset = TensorDataset(grid_tensor, aux_tensor, y_tensor)
        else:
            train_dataset = TensorDataset(grid_tensor, y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 검증 데이터
        val_loader = None
        val_has_aux = False
        if validation_data is not None:
            X_val, y_val = validation_data
            grids_val, aux_val = _to_grid(X_val, self.input_dim)
            grid_val_tensor = torch.FloatTensor(grids_val).to(self._device)
            y_val_tensor = torch.FloatTensor(y_val).to(self._device)

            if aux_val is not None:
                aux_val_tensor = torch.FloatTensor(aux_val).to(self._device)
                val_dataset = TensorDataset(grid_val_tensor, aux_val_tensor, y_val_tensor)
                val_has_aux = True
            else:
                val_dataset = TensorDataset(grid_val_tensor, y_val_tensor)

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if criterion is None:
            if self.use_focal_loss:
                criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history = TrainingHistory(epochs=epochs)
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        has_aux = aux is not None

        self.net.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch in train_loader:
                if has_aux:
                    batch_grid, batch_aux, batch_y = batch
                else:
                    batch_grid, batch_y = batch
                    batch_aux = None

                optimizer.zero_grad()
                output = self.net(batch_grid, batch_aux)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_loss = train_loss
            if val_loader is not None:
                self.net.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if val_has_aux:
                            batch_grid, batch_aux, batch_y = batch
                        else:
                            batch_grid, batch_y = batch
                            batch_aux = None

                        output = self.net(batch_grid, batch_aux)
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                self.net.train()

            history.add_epoch(train_loss, val_loss)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_trained = True
        return history

    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        self.net.eval()
        grid_tensor, aux_tensor = self._prepare_input(X)
        with torch.no_grad():
            probs = self.net(grid_tensor, aux_tensor).squeeze(0).cpu().numpy()
        return ProbabilityDistribution(probs, self.model_type)

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'config': self.config,
            'model_type': self.model_type,
            'model_version': self._model_version,
        }, path)

    def load(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)

            saved_config = checkpoint.get('config', {})
            saved_seq = saved_config.get('seq_length', 20)
            saved_input_dim = saved_config.get('input_dim', 45)
            saved_version = checkpoint.get('model_version', 1)

            # Update config from saved
            self.config.update(saved_config)
            self.input_dim = saved_input_dim
            self.seq_length = saved_seq
            self.output_dim = saved_config.get('output_dim', self.output_dim)
            self.dropout = saved_config.get('dropout', self.dropout)
            self.aux_dim = max(0, self.input_dim - 45)
            self._model_version = saved_version

            if saved_version == 1:
                # V1 모델 로드
                self.net = LottoCNNGridNetV1(
                    seq_length=self.seq_length,
                    aux_dim=self.aux_dim,
                    output_dim=self.output_dim,
                    dropout=self.dropout,
                ).to(self._device)
            else:
                # V2 모델 로드
                self.use_cbam = saved_config.get('use_cbam', True)
                self.cbam_reduction = saved_config.get('cbam_reduction', 4)
                self.use_residual = saved_config.get('use_residual', True)
                self.use_temporal_attention = saved_config.get('use_temporal_attention', True)
                self.use_focal_loss = saved_config.get('use_focal_loss', True)
                self.focal_alpha = saved_config.get('focal_alpha', 0.75)
                self.focal_gamma = saved_config.get('focal_gamma', 2.0)

                self.net = LottoCNNGridNet(
                    seq_length=self.seq_length,
                    aux_dim=self.aux_dim,
                    output_dim=self.output_dim,
                    dropout=self.dropout,
                    use_cbam=self.use_cbam,
                    cbam_reduction=self.cbam_reduction,
                    use_residual=self.use_residual,
                    use_temporal_attention=self.use_temporal_attention,
                ).to(self._device)

            state_dict = checkpoint['model_state_dict']

            # Migrate old v2 checkpoints (BatchNorm1d → LayerNorm)
            if saved_version >= 2:
                remap = {
                    'bn1.weight': 'ln1.weight', 'bn1.bias': 'ln1.bias',
                    'bn1.running_mean': None, 'bn1.running_var': None, 'bn1.num_batches_tracked': None,
                    'bn2.weight': 'ln2.weight', 'bn2.bias': 'ln2.bias',
                    'bn2.running_mean': None, 'bn2.running_var': None, 'bn2.num_batches_tracked': None,
                }
                migrated = {}
                for k, v in state_dict.items():
                    if k in remap:
                        new_key = remap[k]
                        if new_key is not None:
                            migrated[new_key] = v
                    else:
                        migrated[k] = v
                if migrated != state_dict:
                    state_dict = migrated

            self.net.load_state_dict(state_dict)
            self.is_trained = True
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}") from e

    @property
    def requires_sequence(self) -> bool:
        return True

    def to(self, device):
        self._device = torch.device(device)
        self.net = self.net.to(self._device)
        return self
