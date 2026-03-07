"""모델 평가기"""
from typing import Dict, List
import numpy as np

from core.base_model import BaseModel
from core.types import EvaluationResult


class ModelEvaluator:
    """
    모델 평가기

    - 백테스팅 수행
    - 성능 지표 계산
    - 모델 비교
    """

    PRIZE_TABLE = {3: 5000, 4: 50000, 5: 1500000, 6: 2000000000}

    def __init__(self):
        self.results: Dict[str, EvaluationResult] = {}

    def evaluate(self, model: BaseModel,
                 X: np.ndarray,
                 y: np.ndarray,
                 model_name: str = None) -> EvaluationResult:
        """
        모델 평가

        Args:
            model: 평가할 모델
            X: 입력 데이터
            y: 타겟 (multi-hot)
            model_name: 모델 이름

        Returns:
            평가 결과
        """
        if model_name is None:
            model_name = model.model_type

        hits_dist = {i: 0 for i in range(7)}
        perfect_matches = []
        high_hits = []
        total = len(X)

        for i in range(total):
            sample = X[i:i+1]
            pred = model.predict_proba(sample)
            top6_pred = set(pred.top_k(6))
            actual = set(np.where(y[i] > 0.5)[0] + 1)

            hits = len(top6_pred & actual)
            hits_dist[hits] += 1

            if hits == 6:
                perfect_matches.append({
                    'index': i,
                    'predicted': sorted(top6_pred),
                    'actual': sorted(actual)
                })
            elif hits >= 4:
                high_hits.append({
                    'index': i,
                    'hits': hits,
                    'predicted': sorted(top6_pred),
                    'actual': sorted(actual)
                })

        # 지표 계산
        avg_hits = sum(k * v for k, v in hits_dist.items()) / total
        hit_3plus = sum(hits_dist[k] for k in range(3, 7)) / total
        hit_4plus = sum(hits_dist[k] for k in range(4, 7)) / total

        expected_prize = sum(
            hits_dist[k] / total * self.PRIZE_TABLE.get(k, 0)
            for k in range(3, 7)
        )

        result = EvaluationResult(
            total_tests=total,
            hits_distribution=hits_dist,
            avg_hits=avg_hits,
            hit_3plus_rate=hit_3plus,
            hit_4plus_rate=hit_4plus,
            expected_prize=expected_prize,
            perfect_matches=perfect_matches,
            model_name=model_name
        )

        self.results[model_name] = result
        return result

    def compare_models(self, verbose: bool = True) -> List[tuple]:
        """모델 성능 비교"""
        rankings = sorted(
            [(name, res.avg_hits) for name, res in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )

        if verbose:
            print("\n모델 성능 비교 (평균 적중 기준)")
            print("=" * 50)
            for rank, (name, avg_hits) in enumerate(rankings, 1):
                result = self.results[name]
                print(f"{rank}. {name:15s} | 평균: {avg_hits:.2f} | "
                      f"3+: {result.hit_3plus_rate*100:5.1f}% | "
                      f"4+: {result.hit_4plus_rate*100:5.1f}%")

        return rankings

    def get_best_model(self) -> str:
        """최고 성능 모델 반환"""
        if not self.results:
            return None
        return max(self.results.items(), key=lambda x: x[1].avg_hits)[0]
