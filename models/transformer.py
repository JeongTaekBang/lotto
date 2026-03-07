"""Transformer 기반 예측 모델 - 기존 model.py 래핑"""
import copy
import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn

from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory


class PositionalEncoding(nn.Module):
    """위치 인코딩 (회차 순서 정보)"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LottoTransformerNet(nn.Module):
    """Transformer Encoder 네트워크"""

    def __init__(
        self,
        input_dim=45,
        output_dim=45,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.3,
        seq_length=20,
        use_weighted_pool=True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.seq_length = seq_length
        self.use_weighted_pool = use_weighted_pool

        # 입력 임베딩
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Weighted Pooling
        if use_weighted_pool:
            self.pool_weights = nn.Parameter(torch.linspace(0.5, 1.0, seq_length))

        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        if self.use_weighted_pool:
            seq_len = x.size(1)
            if seq_len == self.seq_length:
                w = self.pool_weights
            else:
                # 길이가 다를 때: 끝부분 기준으로 보간
                w = torch.nn.functional.interpolate(
                    self.pool_weights.view(1, 1, -1),
                    size=seq_len, mode='linear', align_corners=True
                ).view(-1)
            weights = torch.softmax(w, dim=0)
            weights = weights.view(1, -1, 1)
            x = (x * weights).sum(dim=1)
        else:
            x = x.mean(dim=1)

        x = self.fc(x)
        return x


class TransformerModel(BaseModel):
    """Transformer 기반 로또 예측 모델"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "transformer"

        # 기본 설정
        self.input_dim = self.config.get('input_dim', 45)
        self.output_dim = self.config.get('output_dim', 45)
        self.d_model = self.config.get('d_model', 64)
        self.nhead = self.config.get('nhead', 4)
        self.num_layers = self.config.get('num_layers', 3)
        self.dim_feedforward = self.config.get('dim_feedforward', 128)
        self.dropout = self.config.get('dropout', 0.3)
        self.seq_length = self.config.get('seq_length', 20)
        self.use_weighted_pool = self.config.get('use_weighted_pool', True)

        # 디바이스 설정
        self._device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 네트워크 생성
        self.net = LottoTransformerNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            seq_length=self.seq_length,
            use_weighted_pool=self.use_weighted_pool
        ).to(self._device)

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
        from torch.utils.data import TensorDataset, DataLoader

        # 데이터 준비
        X_tensor = torch.FloatTensor(X).to(self._device)
        y_tensor = torch.FloatTensor(y).to(self._device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 검증 데이터
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val).to(self._device)
            y_val_tensor = torch.FloatTensor(y_val).to(self._device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 손실함수 및 옵티마이저
        if criterion is None:
            criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 학습 이력
        history = TrainingHistory(epochs=epochs)
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.net.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.net(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 검증
            val_loss = train_loss
            if val_loader is not None:
                self.net.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        output = self.net(batch_x)
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                self.net.train()

            history.add_epoch(train_loss, val_loss)
            scheduler.step(val_loss)

            # Early stopping
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

        # 최고 상태 복원
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_trained = True
        return history

    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        """확률 분포 예측"""
        self.net.eval()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self._device)

        with torch.no_grad():
            probs = self.net(X).squeeze(0).cpu().numpy()

        return ProbabilityDistribution(probs, self.model_type)

    def save(self, path: str) -> None:
        """모델 저장"""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'config': self.config,
            'model_type': self.model_type,
        }, path)
        print(f"모델 저장: {path}")

    def load(self, path: str) -> None:
        """모델 로드

        Raises:
            FileNotFoundError: 파일이 없을 때
            RuntimeError: 로드 실패 시
        """
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
                self.input_dim = saved_config.get('input_dim', self.input_dim)
                self.output_dim = saved_config.get('output_dim', self.output_dim)
                self.d_model = saved_config.get('d_model', self.d_model)
                self.nhead = saved_config.get('nhead', self.nhead)
                self.num_layers = saved_config.get('num_layers', self.num_layers)
                self.dim_feedforward = saved_config.get('dim_feedforward', self.dim_feedforward)
                self.dropout = saved_config.get('dropout', self.dropout)
                self.seq_length = saved_seq
                self.use_weighted_pool = saved_config.get('use_weighted_pool', self.use_weighted_pool)
                self.net = LottoTransformerNet(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_layers=self.num_layers,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    seq_length=self.seq_length,
                    use_weighted_pool=self.use_weighted_pool,
                ).to(self._device)

            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = True
            print(f"모델 로드: {path}")
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}") from e

    @property
    def requires_sequence(self) -> bool:
        return True

    def to(self, device):
        """디바이스 이동"""
        self._device = torch.device(device)
        self.net = self.net.to(self._device)
        return self


if __name__ == "__main__":
    # 테스트
    model = TransformerModel({'input_dim': 45, 'd_model': 64})
    print(f"모델 파라미터: {sum(p.numel() for p in model.net.parameters()):,}")

    # 더미 입력
    dummy = np.random.rand(1, 20, 45).astype(np.float32)
    probs = model.predict_proba(dummy)
    print(f"Top-6: {probs.top_k(6)}")

    # 번호 예측
    predictions = model.predict_numbers(dummy, num_sets=3)
    for i, pred in enumerate(predictions):
        print(f"세트 {i+1}: {pred.numbers} (신뢰도: {pred.confidence:.3f})")
