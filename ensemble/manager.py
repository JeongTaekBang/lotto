"""앙상블 매니저 - 전체 파이프라인 관리"""
import json
from typing import Dict, List, Optional, Any, Type
import numpy as np

from core.base_model import BaseModel
from core.base_ensemble import BaseEnsemble
from core.base_filter import BaseFilter
from core.types import Prediction, ProbabilityDistribution, EvaluationResult
from core.constants import PRIZE_TABLE


class EnsembleManager:
    """
    앙상블 시스템 매니저

    - 여러 모델 등록 및 관리
    - 앙상블 전략 설정
    - 필터 파이프라인 구성
    - 예측 및 평가 실행
    """

    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.ensemble_strategy: Optional[BaseEnsemble] = None
        self.filters: List[BaseFilter] = []
        self._model_performances: Dict[str, float] = {}

    def register_model(self, name: str, model: BaseModel, weight: float = 1.0):
        """
        모델 등록

        Args:
            name: 모델 이름 (고유 식별자)
            model: BaseModel 인스턴스
            weight: 앙상블 가중치
        """
        self.models[name] = model
        if self.ensemble_strategy is not None:
            self.ensemble_strategy.add_model(model, weight)

    def unregister_model(self, name: str) -> bool:
        """모델 등록 해제"""
        if name in self.models:
            model = self.models.pop(name)
            if self.ensemble_strategy is not None:
                self.ensemble_strategy.remove_model(model.model_type)
            return True
        return False

    def set_strategy(self, strategy: BaseEnsemble):
        """
        앙상블 전략 설정

        Args:
            strategy: BaseEnsemble 인스턴스
        """
        self.ensemble_strategy = strategy

        # 기존 등록된 모델들을 전략에 추가
        for name, model in self.models.items():
            self.ensemble_strategy.add_model(model)

    def add_filter(self, filter_: BaseFilter):
        """필터 추가"""
        self.filters.append(filter_)

    def remove_filter(self, filter_name: str) -> bool:
        """필터 제거"""
        for i, f in enumerate(self.filters):
            if f.name == filter_name:
                self.filters.pop(i)
                return True
        return False

    def clear_filters(self):
        """모든 필터 제거"""
        self.filters.clear()

    def predict(self, X: np.ndarray, num_sets: int = 5,
                temperature: float = 1.0,
                apply_filters: bool = True,
                mode: str = None) -> List[Prediction]:
        """
        앙상블 예측 실행

        Args:
            X: 입력 데이터
            num_sets: 생성할 예측 세트 수
            temperature: 샘플링 온도
            apply_filters: 필터 적용 여부
            mode: 선택 모드 ('safe', 'balanced', 'aggressive')
                  지정 시 MMR 기반 다양성 선택 사용

        Returns:
            예측 결과 리스트
        """
        if not self.models:
            raise ValueError("등록된 모델이 없습니다.")

        # MMR 기반 다양성 선택 모드
        if mode is not None:
            return self._predict_with_selector(X, num_sets, mode)

        # 기존 방식
        if self.ensemble_strategy is None:
            first_model = list(self.models.values())[0]
            predictions = first_model.predict_numbers(X, num_sets, temperature)
        else:
            extra_sets = num_sets * 2 if apply_filters and self.filters else num_sets
            predictions = self.ensemble_strategy.predict_numbers(X, extra_sets, temperature)

        if apply_filters:
            for filter_ in self.filters:
                predictions = filter_(predictions)
                if not predictions:
                    break

        return predictions[:num_sets]

    def _predict_with_selector(self, X: np.ndarray, num_sets: int,
                               mode: str) -> List[Prediction]:
        """MMR 기반 후보 생성 → 스코어링 → 다양성 선택"""
        from core.selector import CandidateSelector

        proba = self.predict_proba(X)
        selector = CandidateSelector(filters=self.filters, preset=mode)
        ensemble_name = self.ensemble_strategy.name if self.ensemble_strategy else "ensemble"
        return selector.select(proba, num_sets=num_sets, model_name=ensemble_name)

    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        """앙상블 확률 분포 예측"""
        if self.ensemble_strategy is not None:
            return self.ensemble_strategy.predict_proba(X)
        elif self.models:
            first_model = list(self.models.values())[0]
            return first_model.predict_proba(X)
        else:
            raise ValueError("등록된 모델이 없습니다.")

    def predict_individual(self, X: np.ndarray) -> Dict[str, ProbabilityDistribution]:
        """각 모델의 개별 예측 반환"""
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict_proba(X)
        return results

    def train_all(self, X: np.ndarray, y: np.ndarray,
                  validation_data=None,
                  epochs: int = 100,
                  verbose: bool = True,
                  **kwargs):
        """모든 등록된 모델 학습"""
        histories = {}

        for name, model in self.models.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"모델 학습: {name}")
                print('='*50)

            history = model.train(
                X, y,
                validation_data=validation_data,
                epochs=epochs,
                verbose=verbose,
                **kwargs
            )
            histories[name] = history

        return histories

    def evaluate_all(self, X: np.ndarray, y: np.ndarray,
                     verbose: bool = True) -> Dict[str, EvaluationResult]:
        """모든 모델 평가"""
        results = {}

        for name, model in self.models.items():
            result = self._evaluate_model(model, X, y, name)
            results[name] = result
            self._model_performances[name] = result.avg_hits

            if verbose:
                print(f"\n{result.summary()}")

        # 앙상블 평가
        if self.ensemble_strategy is not None and len(self.models) > 1:
            ensemble_result = self._evaluate_ensemble(X, y)
            results['ensemble'] = ensemble_result
            if verbose:
                print(f"\n{ensemble_result.summary()}")

        return results

    def _evaluate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                        model_name: str) -> EvaluationResult:
        """단일 모델 평가"""
        hits_dist = {i: 0 for i in range(7)}
        perfect_matches = []
        total = len(X)

        if total == 0:
            return EvaluationResult(
                total_tests=0, hits_distribution=hits_dist,
                avg_hits=0.0, hit_3plus_rate=0.0, hit_4plus_rate=0.0,
                expected_prize=0.0, model_name=model_name
            )

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

        avg_hits = sum(k * v for k, v in hits_dist.items()) / total
        hit_3plus = sum(hits_dist[k] for k in range(3, 7)) / total
        hit_4plus = sum(hits_dist[k] for k in range(4, 7)) / total

        # 기대 수익 계산
        expected_prize = sum(
            hits_dist[k] / total * PRIZE_TABLE.get(k, 0)
            for k in range(3, 7)
        )

        return EvaluationResult(
            total_tests=total,
            hits_distribution=hits_dist,
            avg_hits=avg_hits,
            hit_3plus_rate=hit_3plus,
            hit_4plus_rate=hit_4plus,
            expected_prize=expected_prize,
            perfect_matches=perfect_matches,
            model_name=model_name
        )

    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
        """앙상블 평가"""
        hits_dist = {i: 0 for i in range(7)}
        perfect_matches = []
        total = len(X)

        if total == 0:
            return EvaluationResult(
                total_tests=0, hits_distribution=hits_dist,
                avg_hits=0.0, hit_3plus_rate=0.0, hit_4plus_rate=0.0,
                expected_prize=0.0,
                model_name=f"Ensemble ({self.ensemble_strategy.name})"
            )

        for i in range(total):
            sample = X[i:i+1]
            pred = self.ensemble_strategy.predict_proba(sample)
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

        avg_hits = sum(k * v for k, v in hits_dist.items()) / total
        hit_3plus = sum(hits_dist[k] for k in range(3, 7)) / total
        hit_4plus = sum(hits_dist[k] for k in range(4, 7)) / total

        expected_prize = sum(
            hits_dist[k] / total * PRIZE_TABLE.get(k, 0)
            for k in range(3, 7)
        )

        return EvaluationResult(
            total_tests=total,
            hits_distribution=hits_dist,
            avg_hits=avg_hits,
            hit_3plus_rate=hit_3plus,
            hit_4plus_rate=hit_4plus,
            expected_prize=expected_prize,
            perfect_matches=perfect_matches,
            model_name=f"Ensemble ({self.ensemble_strategy.name})"
        )

    def get_model_rankings(self) -> List[tuple]:
        """모델 성능 순위 반환"""
        return sorted(
            self._model_performances.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def auto_weight_by_performance(self, X: np.ndarray, y: np.ndarray,
                                    method: str = 'softmax',
                                    verbose: bool = True) -> Dict[str, float]:
        """
        성능 기반 자동 가중치 계산

        Args:
            X: 평가용 입력 데이터
            y: 평가용 타겟
            method: 가중치 계산 방식 ('softmax', 'linear', 'rank')
            verbose: 출력 여부

        Returns:
            모델별 가중치 딕셔너리
        """
        # 각 모델 평가
        for name, model in self.models.items():
            if name not in self._model_performances:
                result = self._evaluate_model(model, X, y, name)
                self._model_performances[name] = result.avg_hits

        performances = self._model_performances

        if not performances:
            return {}

        # 가중치 계산
        names = list(performances.keys())
        scores = np.array([performances[n] for n in names])

        if method == 'softmax':
            # Softmax: 성능 차이를 부드럽게 반영
            temperature = 0.5  # 낮을수록 차이 극대화
            exp_scores = np.exp(scores / temperature)
            weights = exp_scores / exp_scores.sum()

        elif method == 'linear':
            # Linear: 성능에 비례
            min_score = scores.min()
            adjusted = scores - min_score + 0.1  # 최소값도 약간의 가중치
            weights = adjusted / adjusted.sum()

        elif method == 'rank':
            # Rank: 순위 기반 (1등이 가장 높은 가중치)
            ranks = len(scores) - np.argsort(np.argsort(scores))  # 역순위
            weights = ranks / ranks.sum()

        else:
            # 균등 가중치
            weights = np.ones(len(names)) / len(names)

        weight_dict = {n: float(w) for n, w in zip(names, weights)}

        # 앙상블 전략에 가중치 적용
        if self.ensemble_strategy is not None:
            self.ensemble_strategy.weights = weight_dict

        if verbose:
            print("\n자동 가중치 계산 완료:")
            for name, weight in sorted(weight_dict.items(), key=lambda x: -x[1]):
                perf = performances[name]
                print(f"  {name:15s}: {weight:.3f} (성능: {perf:.2f})")

        return weight_dict

    def save_config(self, path: str):
        """앙상블 설정 저장"""
        config = {
            'models': list(self.models.keys()),
            'strategy': self.ensemble_strategy.name if self.ensemble_strategy else None,
            'weights': self.ensemble_strategy.weights if self.ensemble_strategy else {},
            'filters': [f.name for f in self.filters],
            'performances': self._model_performances
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (f"EnsembleManager("
                f"models={list(self.models.keys())}, "
                f"strategy={self.ensemble_strategy.name if self.ensemble_strategy else None}, "
                f"filters={[f.name for f in self.filters]})")
