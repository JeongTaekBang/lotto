"""MySQL 데이터 소스 - 기존 data.py 기반"""
import os
from typing import List, Optional
import numpy as np
import pymysql
from dotenv import load_dotenv

from core.base_datasource import BaseDataSource
from core.types import LottoRecord


class MySQLDataSource(BaseDataSource):
    """MySQL 기반 로또 데이터 소스"""

    def __init__(self, db_config: dict = None):
        """
        Args:
            db_config: DB 연결 설정 (None이면 환경변수에서 로드)
        """
        super().__init__()
        load_dotenv()

        if db_config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'db': os.getenv('DB_NAME'),
            }
        else:
            self.db_config = db_config

    def _parse_int(self, value) -> Optional[int]:
        """쉼표가 포함된 숫자 문자열을 int로 변환"""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.replace(',', ''))
        return int(value)

    def load(self) -> List[LottoRecord]:
        """MySQL에서 데이터 로드

        Raises:
            ConnectionError: DB 연결 실패 시
            ValueError: 데이터 파싱 오류 시
        """
        conn = None
        try:
            conn = pymysql.connect(**self.db_config, charset='utf8mb4')
        except pymysql.Error as e:
            raise ConnectionError(f"DB 연결 실패: {e}") from e

        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT count, `1`,`2`,`3`,`4`,`5`,`6`,`7`, person, amount
                    FROM lotto ORDER BY count ASC
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
        except pymysql.Error as e:
            raise RuntimeError(f"쿼리 실행 실패: {e}") from e
        finally:
            if conn:
                conn.close()

        try:
            self._records = []
            for row in rows:
                round_num = int(row[0])
                numbers = sorted([int(row[i]) for i in range(1, 7)])
                bonus = int(row[7])
                winners = self._parse_int(row[8])
                prize = self._parse_int(row[9])

                self._records.append(LottoRecord(
                    round_num=round_num,
                    numbers=numbers,
                    bonus=bonus,
                    winners=winners,
                    prize=prize
                ))

            self._loaded = True
            print(f"데이터 로드 완료: {len(self._records)}회차")
            return self._records

        except (TypeError, ValueError) as e:
            raise ValueError(f"데이터 파싱 오류: {e}") from e

    def get_raw_data(self) -> np.ndarray:
        """
        기존 LottoData 호환용: 원본 데이터 반환
        Returns: (N, 8) - [count, 1, 2, 3, 4, 5, 6, bonus]
        """
        if not self._loaded:
            self.load()

        result = []
        for rec in self._records:
            row = [rec.round_num] + rec.numbers + [rec.bonus]
            result.append(row)

        return np.array(result)


if __name__ == "__main__":
    # 테스트
    source = MySQLDataSource()
    records = source.load()

    print(f"\n최근 5회차:")
    for rec in source.get_latest(5):
        print(f"  {rec.round_num}회: {rec.numbers} + {rec.bonus}")

    print(f"\n총 {len(source)}회차 로드됨")
