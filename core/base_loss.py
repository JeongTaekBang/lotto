"""손실함수 추상 클래스"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLoss(ABC, nn.Module):
    """모든 손실함수의 베이스 클래스"""

    def __init__(self):
        super().__init__()
        self.name: str = "base"

    @abstractmethod
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        손실 계산

        Args:
            inputs: 모델 출력 (batch, 45) - 확률 (0~1)
            targets: 타겟 (batch, 45) - multi-hot (0 or 1)

        Returns:
            스칼라 손실 값
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
