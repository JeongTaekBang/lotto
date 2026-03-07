"""공통 유틸리티 함수"""
from typing import List
import numpy as np

from .constants import LOTTO_POOL_SIZE


def sample_numbers(probs: np.ndarray, k: int = 6, temperature: float = 1.0) -> List[int]:
    """
    확률 분포 기반 번호 샘플링

    Args:
        probs: 45개 번호의 확률 분포
        k: 선택할 번호 개수
        temperature: 샘플링 온도 (높을수록 다양성 증가)

    Returns:
        선택된 번호 리스트 (1-based)
    """
    probs = probs.copy()

    # 온도 스케일링
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    selected = []
    temp_probs = probs.copy()

    for _ in range(k):
        # 정규화
        if temp_probs.sum() > 0:
            temp_probs = temp_probs / temp_probs.sum()
        else:
            temp_probs = np.ones(LOTTO_POOL_SIZE) / LOTTO_POOL_SIZE

        # 샘플링
        idx = np.random.choice(LOTTO_POOL_SIZE, p=temp_probs)
        selected.append(idx + 1)  # 1-based
        temp_probs[idx] = 0  # 중복 방지

    return selected


def flatten_sequence(X: np.ndarray) -> np.ndarray:
    """
    시퀀스를 통계 피처로 변환

    (batch, seq_len, feature_dim) → (batch, flattened_features)

    추출되는 통계:
    - mean: 평균
    - std: 표준편차
    - last: 마지막 시점
    - max: 최대값
    - min: 최소값
    - trend: 트렌드 (마지막 - 첫번째)

    Args:
        X: 입력 시퀀스 (3D 또는 2D)

    Returns:
        평탄화된 피처 (2D)
    """
    if X.ndim == 2:
        return X  # 이미 flatten됨

    batch_size = X.shape[0]
    features_list = []

    for b in range(batch_size):
        seq = X[b]  # (seq_len, feature_dim)

        # 통계 피처 추출
        mean_feat = seq.mean(axis=0)       # 평균
        std_feat = seq.std(axis=0)         # 표준편차
        last_feat = seq[-1]                # 마지막 시점
        max_feat = seq.max(axis=0)         # 최대값
        min_feat = seq.min(axis=0)         # 최소값
        trend_feat = seq[-1] - seq[0]      # 트렌드

        feat = np.concatenate([
            mean_feat,
            std_feat,
            last_feat,
            max_feat,
            min_feat,
            trend_feat
        ])
        features_list.append(feat)

    return np.array(features_list)


def numbers_to_multihot(numbers: List[int], pool_size: int = LOTTO_POOL_SIZE) -> np.ndarray:
    """
    번호 리스트를 multi-hot 벡터로 변환

    Args:
        numbers: 번호 리스트 (1-based)
        pool_size: 전체 번호 풀 크기

    Returns:
        multi-hot 벡터
    """
    multihot = np.zeros(pool_size)
    for num in numbers:
        if 1 <= num <= pool_size:
            multihot[num - 1] = 1
    return multihot


def multihot_to_numbers(multihot: np.ndarray, threshold: float = 0.5) -> List[int]:
    """
    multi-hot 벡터를 번호 리스트로 변환

    Args:
        multihot: multi-hot 벡터
        threshold: 임계값

    Returns:
        번호 리스트 (1-based)
    """
    return [i + 1 for i, v in enumerate(multihot) if v > threshold]


def calc_ac_value(numbers: List[int]) -> int:
    """
    AC값 (Arithmetic Complexity) 계산

    6개 번호 간 모든 차이값의 고유 개수 - 5

    Args:
        numbers: 6개 번호 리스트

    Returns:
        AC값 (0-10)
    """
    sorted_nums = sorted(numbers)
    differences = set()
    for i in range(len(sorted_nums)):
        for j in range(i + 1, len(sorted_nums)):
            differences.add(sorted_nums[j] - sorted_nums[i])
    return len(differences) - 5


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """값을 0-1 범위로 정규화"""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)
