"""GRU 기반 예측 모델"""
import copy
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory


class LottoGRUNet(nn.Module):
    """GRU 네트워크 - LSTM보다 파라미터가 적고 빠름"""

    def __init__(
        self,
        input_dim=45,
        hidden_dim=128,
        num_layers=2,
        output_dim=45,
        dropout=0.3,
        bidirectional=False
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gru_out, h_n = self.gru(x)

        if self.num_directions == 2:
            last_out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_out = h_n[-1]

        output = self.fc(last_out)
        return output


class GRUModel(BaseModel):
    """GRU 기반 로또 예측 모델"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "gru"

        self.input_dim = self.config.get('input_dim', 45)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.num_layers = self.config.get('num_layers', 2)
        self.output_dim = self.config.get('output_dim', 45)
        self.dropout = self.config.get('dropout', 0.3)
        self.bidirectional = self.config.get('bidirectional', False)

        self._device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.net = LottoGRUNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout,
            bidirectional=self.bidirectional
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

        X_tensor = torch.FloatTensor(X).to(self._device)
        y_tensor = torch.FloatTensor(y).to(self._device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val).to(self._device)
            y_val_tensor = torch.FloatTensor(y_val).to(self._device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if criterion is None:
            criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self._device)
        with torch.no_grad():
            probs = self.net(X).squeeze(0).cpu().numpy()
        return ProbabilityDistribution(probs, self.model_type)

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'config': self.config,
            'model_type': self.model_type,
        }, path)

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
