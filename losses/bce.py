"""Binary Cross Entropy Loss"""
import torch
import torch.nn as nn
from core.base_loss import BaseLoss


class BCELoss(BaseLoss):
    """Binary Cross Entropy 손실함수"""

    def __init__(self):
        super().__init__()
        self.name = "bce"
        self._bce = nn.BCELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._bce(inputs, targets)
