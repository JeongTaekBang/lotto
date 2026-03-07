"""
로또 당첨번호 크롤링 모듈
동행복권 사이트에서 최신 당첨번호를 수집
"""
import os
import requests
import pymysql
from dotenv import load_dotenv

load_dotenv()


class LottoCrawler:
    """로또 당첨번호 크롤러"""

    def __init__(self):
        self.api_url = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
        self.session = requests.Session()

        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'db': os.getenv('DB_NAME'),
        }

    def get_connection(self):
        """DB 연결"""
        return pymysql.connect(**self.db_config, charset='utf8mb4')

    def get_db_last_round(self):
        """DB에 저장된 마지막 회차"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MAX(count) FROM lotto")
                result = cursor.fetchone()
                return result[0] or 0

    def get_latest_round(self):
        """동행복권 사이트의 최신 회차"""
        response = self.session.get(self.api_url, timeout=10)
        data = response.json()

        if not data.get('data') or not data['data'].get('list'):
            raise ValueError("최신 회차 정보를 찾을 수 없습니다")

        return data['data']['list'][0]['ltEpsd']

    def crawl_round(self, round_num):
        """특정 회차 크롤링"""
        response = self.session.get(
            self.api_url,
            params={'srchLtEpsd': round_num},
            timeout=10
        )
        data = response.json()

        if not data.get('data') or not data['data'].get('list'):
            return None

        info = data['data']['list'][0]

        numbers = [
            info['tm1WnNo'],
            info['tm2WnNo'],
            info['tm3WnNo'],
            info['tm4WnNo'],
            info['tm5WnNo'],
            info['tm6WnNo'],
        ]

        return {
            'round': info['ltEpsd'],
            'numbers': numbers,
            'bonus': info['bnsWnNo'],
            'winners': info['rnk1WnNope'],
            'prize': info['rnk1WnAmt']
        }

    def insert_round(self, data):
        """회차 데이터 DB 삽입"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO lotto (count, `1`, `2`, `3`, `4`, `5`, `6`, `7`, person, amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    data['round'],
                    *data['numbers'],
                    data['bonus'],
                    data['winners'],
                    data['prize']
                ))
            conn.commit()

    def crawl_latest(self):
        """최신 데이터 크롤링 및 저장"""
        db_last = self.get_db_last_round()
        site_latest = self.get_latest_round()

        print(f"DB 마지막 회차: {db_last}")
        print(f"사이트 최신 회차: {site_latest}")

        if db_last >= site_latest:
            print("이미 최신 상태입니다.")
            return 0

        count = 0
        for round_num in range(db_last + 1, site_latest + 1):
            print(f"크롤링 중: {round_num}회차...", end=" ")

            try:
                data = self.crawl_round(round_num)
                if data:
                    self.insert_round(data)
                    nums = data['numbers']
                    print(f"완료 [{nums[0]}, {nums[1]}, {nums[2]}, {nums[3]}, {nums[4]}, {nums[5]}] + {data['bonus']}")
                    count += 1
                else:
                    print("데이터 없음")

            except Exception as e:
                print(f"오류: {e}")

        print(f"\n총 {count}개 회차 추가 완료")
        return count


if __name__ == "__main__":
    crawler = LottoCrawler()
    crawler.crawl_latest()
