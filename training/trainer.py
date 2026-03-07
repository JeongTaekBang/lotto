"""통합 트레이너 - 모든 모델 학습 지원"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from core.base_model import BaseModel
from core.base_datasource import BaseDataSource
from core.types import TrainingHistory
from core.feature_extractor import FeatureExtractor
from core.constants import (
    FEATURE_DIM_BASIC,
    FEATURE_DIM_EXTENDED,
    FEATURE_DIM_XGBOOST,
    FEATURE_DIM_BONUS,
    DEFAULT_SEQ_LENGTH,
    FeatureMode,
)
from core.utils import numbers_to_multihot
from models.factory import ModelFactory


class UnifiedTrainer:
    """
    통합 트레이너

    - 다양한 모델 학습 지원
    - 데이터 전처리 통합
    - 학습 파이프라인 관리
    """

    def __init__(self,
                 datasource: BaseDataSource,
                 seq_length: int = DEFAULT_SEQ_LENGTH,
                 use_extended_features: bool = False,
                 use_bonus: bool = False,
                 feature_mode: str = None):
        """
        Args:
            datasource: 데이터 소스
            seq_length: 시퀀스 길이
            use_extended_features: 확장 피처 사용 여부 (deprecated, use feature_mode)
            use_bonus: 보너스 번호 피처 사용 여부
            feature_mode: 피처 모드 ('basic', 'extended', 'xgboost')
                - basic: 45차원 (multi-hot only)
                - extended: 74차원 (전체 확장 피처)
                - xgboost: 66차원 (중복 제거된 확장 피처)
        """
        self.datasource = datasource
        self.seq_length = seq_length
        self.use_bonus = use_bonus

        # feature_mode 결정 (하위 호환성)
        if feature_mode:
            self.feature_mode = feature_mode
        elif use_extended_features:
            self.feature_mode = FeatureMode.EXTENDED
        else:
            self.feature_mode = FeatureMode.BASIC

        # FeatureExtractor 초기화
        self._feature_extractor = FeatureExtractor(self.feature_mode, use_bonus)
        self.feature_dim = self._feature_extractor.feature_dim

        # 하위 호환성을 위한 차원 상수
        self.base_dim = FEATURE_DIM_BASIC
        self.extended_dim = FEATURE_DIM_EXTENDED
        self.xgboost_dim = FEATURE_DIM_XGBOOST
        self.bonus_dim = FEATURE_DIM_BONUS

        # 하위 호환성
        self.use_extended_features = self.feature_mode in (FeatureMode.EXTENDED, FeatureMode.XGBOOST)

        self._sequences = None
        self._targets = None

    def numbers_to_multihot(self, numbers: List[int]) -> np.ndarray:
        """번호를 multi-hot 벡터로 변환"""
        return numbers_to_multihot(numbers)

    def numbers_to_features(self, numbers: List[int],
                           bonus: int = None,
                           prev_numbers: List[int] = None,
                           round_num: int = None) -> np.ndarray:
        """번호를 피처로 변환 (FeatureExtractor 위임)"""
        return self._feature_extractor.extract(numbers, bonus, prev_numbers, round_num)

    def prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """학습용 시퀀스 생성"""
        if self._sequences is not None:
            return self._sequences, self._targets

        records = self.datasource.load()
        n_records = len(records)

        sequences = []
        targets = []

        for i in range(self.seq_length, n_records):
            seq = []
            for j in range(i - self.seq_length, i):
                rec = records[j]
                prev_rec = records[j-1] if j > 0 else None

                if self.use_extended_features or self.use_bonus:
                    feat = self.numbers_to_features(
                        rec.numbers,
                        bonus=rec.bonus if self.use_bonus else None,
                        prev_numbers=prev_rec.numbers if prev_rec else None,
                        round_num=rec.round_num
                    )
                else:
                    feat = self.numbers_to_multihot(rec.numbers)

                seq.append(feat)

            sequences.append(seq)
            targets.append(self.numbers_to_multihot(records[i].numbers))

        self._sequences = np.array(sequences, dtype=np.float32)
        self._targets = np.array(targets, dtype=np.float32)

        return self._sequences, self._targets

    def get_train_test_split(self, test_ratio: float = 0.1) -> Tuple:
        """학습/테스트 분할"""
        X, y = self.prepare_sequences()
        split_idx = int(len(X) * (1 - test_ratio))

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        return X_train, y_train, X_test, y_test

    def get_latest_sequence(self) -> np.ndarray:
        """최신 시퀀스 반환 (예측용)"""
        records = self.datasource.load()
        n_records = len(records)

        if n_records < self.seq_length:
            raise ValueError(
                f"데이터 부족: {n_records}개 레코드, 최소 {self.seq_length}개 필요"
            )

        seq = []
        for i in range(n_records - self.seq_length, n_records):
            rec = records[i]
            prev_rec = records[i-1] if i > 0 else None

            if self.use_extended_features or self.use_bonus:
                feat = self.numbers_to_features(
                    rec.numbers,
                    bonus=rec.bonus if self.use_bonus else None,
                    prev_numbers=prev_rec.numbers if prev_rec else None,
                    round_num=rec.round_num
                )
            else:
                feat = self.numbers_to_multihot(rec.numbers)

            seq.append(feat)

        return np.array([seq], dtype=np.float32)

    def train_model(self, model: BaseModel,
                    epochs: int = 100,
                    batch_size: int = 32,
                    test_ratio: float = 0.1,
                    verbose: bool = True,
                    **kwargs) -> TrainingHistory:
        """단일 모델 학습"""
        X_train, y_train, X_test, y_test = self.get_train_test_split(test_ratio)

        if verbose:
            print(f"학습 데이터: {len(X_train)}개, 테스트: {len(X_test)}개")
            print(f"피처 차원: {self.feature_dim}")

        return model.train(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs
        )

    def create_and_train(self, model_name: str,
                        config: Dict[str, Any] = None,
                        epochs: int = 100,
                        **kwargs) -> Tuple[BaseModel, TrainingHistory]:
        """모델 생성 및 학습"""
        # 설정에 input_dim, seq_length 추가
        config = config or {}
        config['input_dim'] = self.feature_dim
        config.setdefault('seq_length', self.seq_length)

        model = ModelFactory.create(model_name, config)
        history = self.train_model(model, epochs=epochs, **kwargs)

        return model, history
