"""Ranking Loss - Top-K 순위 학습"""
import torch
from core.base_loss import BaseLoss


class RankingLoss(BaseLoss):
    """
    Ranking Loss - Top-6 예측이 실제 번호와 일치하도록 최적화
    당첨번호는 높은 확률, 비당첨번호는 낮은 확률을 갖도록 유도
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.name = "ranking"
        self.margin = margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, 45) 확률
        # targets: (batch, 45) multi-hot

        pos_mask = targets == 1
        neg_mask = targets == 0

        # 당첨/비당첨 확률의 평균
        pos_probs = (inputs * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        neg_probs = (inputs * neg_mask).sum(dim=1) / neg_mask.sum(dim=1)

        # Margin ranking loss
        loss = torch.clamp(self.margin - (pos_probs - neg_probs), min=0)
        return loss.mean()
