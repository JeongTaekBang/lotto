"""Combined Loss - BCE + Ranking"""
import torch
import torch.nn as nn
from core.base_loss import BaseLoss
from .ranking import RankingLoss


class CombinedLoss(BaseLoss):
    """BCE + Ranking Loss 조합"""

    def __init__(self, bce_weight: float = 0.7, ranking_weight: float = 0.3):
        super().__init__()
        self.name = "combined"
        self.bce = nn.BCELoss()
        self.ranking = RankingLoss()
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        ranking_loss = self.ranking(inputs, targets)
        return self.bce_weight * bce_loss + self.ranking_weight * ranking_loss
