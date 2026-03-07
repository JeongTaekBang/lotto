"""XGBoost 기반 예측 모델"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pickle

from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory
from core.utils import flatten_sequence

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostModel(BaseModel):
    """XGBoost 기반 로또 예측 모델 - 각 번호별 이진 분류기"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "xgboost"

        if not HAS_XGBOOST:
            raise ImportError("xgboost 패키지가 필요합니다. pip install xgboost")

        # XGBoost 설정
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 5)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.min_child_weight = self.config.get('min_child_weight', 1)
        self.subsample = self.config.get('subsample', 0.8)
        self.colsample_bytree = self.config.get('colsample_bytree', 0.8)

        # 45개 번호에 대한 분류기
        self.classifiers: List[Optional[xgb.XGBClassifier]] = [None] * 45

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 100,  # XGBoost에서는 n_estimators로 사용
              verbose: bool = True,
              **kwargs) -> TrainingHistory:
        """45개 번호 각각에 대해 이진 분류기 학습"""

        X_flat = flatten_sequence(X)

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_flat = flatten_sequence(X_val)

        history = TrainingHistory(epochs=1)  # XGBoost는 내부적으로 반복

        for num in range(45):
            y_num = (y[:, num] > 0).astype(int)

            # 클래스 불균형 처리 (당첨 번호가 적음)
            pos_count = y_num.sum()
            neg_count = len(y_num) - pos_count
            scale_pos_weight = neg_count / max(pos_count, 1)

            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=0
            )

            eval_set = None
            if validation_data is not None:
                y_val_num = (y_val[:, num] > 0).astype(int)
                eval_set = [(X_val_flat, y_val_num)]

            clf.fit(
                X_flat, y_num,
                eval_set=eval_set,
                verbose=False
            )

            self.classifiers[num] = clf

            if verbose and (num + 1) % 10 == 0:
                print(f"번호 {num+1}/45 학습 완료")

        self.is_trained = True
        return history

    def predict_proba(self, X: np.ndarray) -> ProbabilityDistribution:
        """각 번호의 당첨 확률 예측"""
        X_flat = flatten_sequence(X)

        probs = np.zeros(45)
        for num in range(45):
            if self.classifiers[num] is not None:
                prob = self.classifiers[num].predict_proba(X_flat)[:, 1]
                probs[num] = prob[0]

        return ProbabilityDistribution(probs, self.model_type)

    def get_feature_importance(self) -> Dict[int, np.ndarray]:
        """각 번호별 피처 중요도 반환"""
        importance = {}
        for num in range(45):
            if self.classifiers[num] is not None:
                importance[num + 1] = self.classifiers[num].feature_importances_
        return importance

    def save(self, path: str) -> None:
        """모델 저장"""
        with open(path, 'wb') as f:
            pickle.dump({
                'classifiers': self.classifiers,
                'config': self.config,
                'model_type': self.model_type,
            }, f)

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
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.classifiers = data['classifiers']
                self.config = data.get('config', self.config)
            self.is_trained = True
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}") from e

    @property
    def requires_sequence(self) -> bool:
        return False  # 내부에서 flatten 처리
