"""공통 타입 정의"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class LottoRecord:
    """단일 회차 데이터"""
    round_num: int
    numbers: List[int]  # 6개 당첨번호 (정렬됨)
    bonus: int
    winners: Optional[int] = None
    prize: Optional[int] = None

    def __post_init__(self):
        self.numbers = sorted(self.numbers)


@dataclass
class Prediction:
    """예측 결과"""
    numbers: List[int]  # 6개 번호 (정렬됨)
    confidence: float   # 신뢰도 (0.0 ~ 1.0)
    model_name: str     # 예측한 모델 이름
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.numbers = sorted(self.numbers)

    def hits(self, actual: List[int]) -> int:
        """실제 당첨번호와 비교하여 적중 수 반환"""
        return len(set(self.numbers) & set(actual))


@dataclass
class ProbabilityDistribution:
    """확률 분포"""
    probabilities: np.ndarray  # shape: (45,)
    model_name: str

    def top_k(self, k: int = 6) -> List[int]:
        """상위 k개 번호 반환 (1-based)"""
        indices = np.argsort(self.probabilities)[-k:][::-1]
        return [int(i + 1) for i in indices]

    def get_prob(self, number: int) -> float:
        """특정 번호의 확률 반환 (1-based)"""
        return float(self.probabilities[number - 1])

    def normalize(self) -> 'ProbabilityDistribution':
        """확률 합이 1이 되도록 정규화"""
        total = self.probabilities.sum()
        if total > 0:
            normalized = self.probabilities / total
        else:
            normalized = np.ones(45) / 45
        return ProbabilityDistribution(normalized, self.model_name)


@dataclass
class EvaluationResult:
    """평가 결과"""
    total_tests: int
    hits_distribution: Dict[int, int]  # {0: count, 1: count, ..., 6: count}
    avg_hits: float
    hit_3plus_rate: float
    hit_4plus_rate: float
    expected_prize: float
    perfect_matches: List[Dict] = field(default_factory=list)  # 6개 적중 케이스
    model_name: str = ""

    def summary(self) -> str:
        """평가 결과 요약 문자열"""
        lines = [
            f"=== {self.model_name} 평가 결과 ===",
            f"총 테스트: {self.total_tests}회",
            f"평균 적중: {self.avg_hits:.2f}개",
            f"3개+ 적중률: {self.hit_3plus_rate*100:.1f}%",
            f"4개+ 적중률: {self.hit_4plus_rate*100:.1f}%",
            f"기대 수익: {self.expected_prize:,.0f}원",
            "",
            "적중 분포:",
        ]
        for hits in range(7):
            count = self.hits_distribution.get(hits, 0)
            pct = count / self.total_tests * 100 if self.total_tests > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {hits}개: {count:4d} ({pct:5.1f}%) {bar}")

        return "\n".join(lines)


@dataclass
class TrainingHistory:
    """학습 이력"""
    epochs: int
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_epoch(self, train_loss: float, val_loss: float = None):
        """에포크 결과 추가"""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = len(self.train_losses)
