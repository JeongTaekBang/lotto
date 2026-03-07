"""스태킹 앙상블"""
from typing import List, Optional, Tuple
import numpy as np

from core.base_ensemble import BaseEnsemble
from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory


class StackingEnsemble(BaseEnsemble):
    """
    스태킹 앙상블

    Level 0: 여러 기반 모델
    Level 1: 메타 모델이 기반 모델들의 예측을 조합
    """

    def __init__(self, models=None, meta_model: Optional[BaseModel] = None):
        super().__init__(models)
        self.name = "stacking_ensemble"
        self.meta_model = meta_model
        self.meta_trained = False

    def set_meta_model(self, meta_model: BaseModel):
        """메타 모델 설정"""
        self.meta_model = meta_model
        self.meta_trained = False

    def train_meta(self, X: np.ndarray, y: np.ndarray,
                   validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   epochs: int = 50,
                   verbose: bool = True) -> TrainingHistory:
        """
        메타 모델 학습

        각 기반 모델의 예측을 메타 피처로 사용

        Args:
            X: 원본 입력 데이터
            y: 타겟
            validation_data: 검증 데이터
            epochs: 메타 모델 학습 에포크
        """
        if self.meta_model is None:
            raise ValueError("메타 모델이 설정되지 않았습니다.")

        if not self.models:
            raise ValueError("기반 모델이 없습니다.")

        # 메타 피처 생성
        meta_features = self._generate_meta_features(X)

        val_meta = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_meta_features = self._generate_meta_features(X_val)
            val_meta = (val_meta_features, y_val)

        # 메타 모델 학습
        history = self.meta_model.train(
            meta_features, y,
            validation_data=val_meta,
            epochs=epochs,
            verbose=verbose
        )

        self.meta_trained = True
        return history

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """기반 모델들의 예측으로 메타 피처 생성"""
        n_samples = X.shape[0]
        meta_features = []

        for i in range(n_samples):
            sample = X[i:i+1]
            sample_features = []

            for model in self.models:
                pred = model.predict_proba(sample)
                sample_features.extend(pred.probabilities)

            meta_features.append(sample_features)

        return np.array(meta_features)

    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """
        메타 모델로 최종 예측

        기반 모델의 예측들을 연결하여 메타 모델 입력으로 사용
        """
        if self.meta_model is None or not self.meta_trained:
            # 메타 모델이 없으면 단순 평균
            avg_probs = np.mean([p.probabilities for p in predictions], axis=0)
            return ProbabilityDistribution(avg_probs, self.name)

        # 메타 입력 생성
        meta_input = np.concatenate([p.probabilities for p in predictions])
        meta_input = meta_input.reshape(1, -1)

        # 메타 모델 예측
        return self.meta_model.predict_proba(meta_input)


class BlendingEnsemble(BaseEnsemble):
    """
    블렌딩 앙상블

    홀드아웃 검증 세트를 사용하여 메타 모델 학습
    스태킹보다 간단하지만 데이터 효율성은 낮음
    """

    def __init__(self, models=None, blend_weights: Optional[np.ndarray] = None):
        super().__init__(models)
        self.name = "blending_ensemble"
        self.blend_weights = blend_weights  # 학습된 블렌딩 가중치

    def learn_weights(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        최적 블렌딩 가중치 학습

        단순 최소제곱법으로 가중치 최적화
        """
        if not self.models:
            raise ValueError("기반 모델이 없습니다.")

        n_samples = X.shape[0]
        n_models = len(self.models)

        # 각 모델의 예측 수집
        predictions = np.zeros((n_samples, n_models, 45))

        for i in range(n_samples):
            sample = X[i:i+1]
            for j, model in enumerate(self.models):
                pred = model.predict_proba(sample)
                predictions[i, j] = pred.probabilities

        # 간단한 가중치 최적화 (평균 손실 최소화)
        best_weights = np.ones(n_models) / n_models
        best_loss = float('inf')

        # 그리드 서치 (단순화)
        for _ in range(1000):
            weights = np.random.dirichlet(np.ones(n_models))
            weighted_pred = np.tensordot(weights, predictions, axes=([0], [1]))

            # BCE 손실 계산
            eps = 1e-7
            loss = -np.mean(
                y * np.log(weighted_pred + eps) +
                (1 - y) * np.log(1 - weighted_pred + eps)
            )

            if loss < best_loss:
                best_loss = loss
                best_weights = weights

        self.blend_weights = best_weights

        if verbose:
            print(f"학습된 블렌딩 가중치:")
            for model, weight in zip(self.models, self.blend_weights):
                print(f"  {model.model_type}: {weight:.4f}")

    def combine_predictions(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """블렌딩 가중치로 예측 결합"""
        if self.blend_weights is None:
            # 가중치가 없으면 균등 평균
            avg_probs = np.mean([p.probabilities for p in predictions], axis=0)
            return ProbabilityDistribution(avg_probs, self.name)

        # 가중 합
        weighted_sum = np.zeros(45)
        for pred, weight in zip(predictions, self.blend_weights):
            weighted_sum += pred.probabilities * weight

        return ProbabilityDistribution(weighted_sum, self.name)
