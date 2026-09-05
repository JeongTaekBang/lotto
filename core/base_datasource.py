"""데이터 소스 추상 클래스"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from .types import LottoRecord


class BaseDataSource(ABC):
    """모든 데이터 소스의 베이스 클래스"""

    def __init__(self):
        self._records: List[LottoRecord] = []
        self._loaded = False

    @abstractmethod
    def load(self) -> List[LottoRecord]:
        """
        데이터 로드

        Returns:
            LottoRecord 리스트 (회차 오름차순 정렬)
        """
        pass

    def get_records(self, start: int = None, end: int = None) -> List[LottoRecord]:
        """
        특정 범위의 레코드 반환

        Args:
            start: 시작 회차 (포함)
            end: 종료 회차 (포함)

        Returns:
            해당 범위의 LottoRecord 리스트
        """
        if not self._loaded:
            self.load()

        records = self._records
        if start is not None:
            records = [r for r in records if r.round_num >= start]
        if end is not None:
            records = [r for r in records if r.round_num <= end]

        return records

    def get_latest(self, n: int = 1) -> List[LottoRecord]:
        """
        최근 n개 레코드 반환

        Args:
            n: 반환할 레코드 수

        Returns:
            최근 n개의 LottoRecord 리스트 (최신순)
        """
        if not self._loaded:
            self.load()

        return self._records[-n:][::-1]  # 최신순 정렬

    def get_last_round_num(self) -> int:
        """마지막 회차 번호"""
        if not self._loaded:
            self.load()

        if not self._records:
            return 0
        return self._records[-1].round_num

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        numpy 배열로 변환

        Returns:
            (numbers, bonus_numbers): ((N, 6), (N,))
        """
        if not self._loaded:
            self.load()

        numbers = np.array([r.numbers for r in self._records])
        bonus = np.array([r.bonus for r in self._records])
        return numbers, bonus

    def get_raw_data(self) -> np.ndarray:
        """
        원본 형태의 2차원 배열로 변환

        Returns:
            (N, 8) 배열 - [회차, 번호1~6, 보너스]
        """
        if not self._loaded:
            self.load()

        return np.array([
            [rec.round_num] + rec.numbers + [rec.bonus]
            for rec in self._records
        ])

    def to_multihot(self, include_bonus: bool = False) -> np.ndarray:
        """
        Multi-hot 인코딩으로 변환

        Args:
            include_bonus: 보너스 번호 포함 여부

        Returns:
            (N, 45) 또는 (N, 46) multi-hot 배열
        """
        if not self._loaded:
            self.load()

        n_records = len(self._records)
        dim = 46 if include_bonus else 45
        result = np.zeros((n_records, dim), dtype=np.float32)

        for i, rec in enumerate(self._records):
            for num in rec.numbers:
                result[i, num - 1] = 1.0
            if include_bonus:
                result[i, 45] = rec.bonus / 45.0  # 정규화

        return result

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._records)

    def __getitem__(self, idx: int) -> LottoRecord:
        if not self._loaded:
            self.load()
        return self._records[idx]
