"""RandomForest 기반 예측 모델"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pickle

from core.base_model import BaseModel
from core.types import ProbabilityDistribution, TrainingHistory
from core.utils import flatten_sequence

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class RandomForestModel(BaseModel):
    """RandomForest 기반 로또 예측 모델"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = "random_forest"

        if not HAS_SKLEARN:
            raise ImportError("scikit-learn 패키지가 필요합니다. pip install scikit-learn")

        # RF 설정
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.min_samples_split = self.config.get('min_samples_split', 5)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 2)
        self.n_jobs = self.config.get('n_jobs', -1)  # 병렬 처리

        # 45개 번호에 대한 분류기
        self.classifiers: List[Optional[RandomForestClassifier]] = [None] * 45

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 100,
              verbose: bool = True,
              **kwargs) -> TrainingHistory:
        """45개 번호 각각에 대해 RF 분류기 학습"""

        X_flat = flatten_sequence(X)
        history = TrainingHistory(epochs=1)

        for num in range(45):
            y_num = (y[:, num] > 0).astype(int)

            # 클래스 불균형 처리
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight='balanced',  # 클래스 불균형 자동 처리
                n_jobs=self.n_jobs,
                random_state=42
            )

            clf.fit(X_flat, y_num)
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
                prob = self.classifiers[num].predict_proba(X_flat)
                # 클래스 1(당첨)의 확률
                if prob.shape[1] > 1:
                    probs[num] = prob[0, 1]
                else:
                    probs[num] = prob[0, 0]

        return ProbabilityDistribution(probs, self.model_type)

    def get_feature_importance(self) -> Dict[int, np.ndarray]:
        """각 번호별 피처 중요도 반환"""
        importance = {}
        for num in range(45):
            if self.classifiers[num] is not None:
                importance[num + 1] = self.classifiers[num].feature_importances_
        return importance

    def save(self, path: str) -> None:
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
        return False
