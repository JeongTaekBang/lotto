"""Focal Loss - 어려운 샘플에 집중"""
import torch
import torch.nn.functional as F
from core.base_loss import BaseLoss


class FocalLoss(BaseLoss):
    """
    Focal Loss - 어려운 샘플에 더 높은 가중치
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.name = "focal"
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
