"""CNN Grid 모델 - 7x7 이미지 패턴 기반 로또 예측"""
import copy
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

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


class LottoCNNGridNet(nn.Module):
    """멀티 브랜치 CNN - 7x7 그리드에서 공간 패턴 학습"""

    def __init__(self, seq_length=20, aux_dim=0, output_dim=45, dropout=0.3):
        super().__init__()

        # Branch 1: 3x3 - 인접 번호 패턴
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(seq_length, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Branch 2: 5x5 - 행/열/대각선 패턴
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(seq_length, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Branch 3: 1x1 - 셀별 시간 패턴
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(seq_length, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Fusion: 32+32+16=80 → 64
        self.fusion = nn.Sequential(
            nn.Conv2d(80, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # FC layers
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
        """
        Args:
            grid: (batch, seq_length, 7, 7)
            aux: (batch, aux_dim) or None
        """
        b1 = self.branch_3x3(grid)
        b2 = self.branch_5x5(grid)
        b3 = self.branch_1x1(grid)

        merged = torch.cat([b1, b2, b3], dim=1)  # (batch, 80, 7, 7)
        fused = self.fusion(merged)               # (batch, 64, 7, 7)
        pooled = self.global_pool(fused)           # (batch, 64, 1, 1)
        features = pooled.view(pooled.size(0), -1) # (batch, 64)

        if self.aux_dim > 0 and aux is not None:
            aux_features = self.aux_fc(aux)         # (batch, 32)
            features = torch.cat([features, aux_features], dim=1)

        return self.fc(features)


class CNNGridModel(BaseModel):
    """CNN Grid 기반 로또 예측 모델 - 7x7 공간 패턴 학습"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "cnn_grid"

        self.input_dim = self.config.get('input_dim', 45)
        self.seq_length = self.config.get('seq_length', 20)
        self.output_dim = self.config.get('output_dim', 45)
        self.dropout = self.config.get('dropout', 0.3)

        self.aux_dim = max(0, self.input_dim - 45)
        self._device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.net = LottoCNNGridNet(
            seq_length=self.seq_length,
            aux_dim=self.aux_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
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
        }, path)

    def load(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)

            # 저장된 config로 네트워크 재구성 (seq_length 등 불일치 방지)
            saved_config = checkpoint.get('config', {})
            saved_seq = saved_config.get('seq_length', 20)
            saved_input_dim = saved_config.get('input_dim', 45)

            if saved_seq != self.seq_length or saved_input_dim != self.input_dim:
                self.config.update(saved_config)
                self.input_dim = saved_input_dim
                self.seq_length = saved_seq
                self.output_dim = saved_config.get('output_dim', self.output_dim)
                self.dropout = saved_config.get('dropout', self.dropout)
                self.aux_dim = max(0, self.input_dim - 45)
                self.net = LottoCNNGridNet(
                    seq_length=self.seq_length,
                    aux_dim=self.aux_dim,
                    output_dim=self.output_dim,
                    dropout=self.dropout,
                ).to(self._device)

            self.net.load_state_dict(checkpoint['model_state_dict'])
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
