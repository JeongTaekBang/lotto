import os
import requests
from bs4 import BeautifulSoup
import pymysql
import pandas as pd
import collections
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Dict, List

# .env 파일에서 환경 변수 로드
# .env 파일 형식은 .env.example 참조
load_dotenv()


class LottoDBManager:
    def __init__(self):
        # 환경 변수에서 데이터베이스 설정 로드
        # .env 파일에 다음 변수들이 정의되어 있어야 함:
        # DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),  # 민감 정보는 .env 파일에 저장
            'db': os.getenv('DB_NAME'),
            'cursorclass': pymysql.cursors.DictCursor
        }
        
        # .env 파일이 올바르게 설정되었는지 확인
        if not self.db_config['host'] or not self.db_config['password']:
            print("경고: 데이터베이스 설정이 올바르지 않습니다.")
            print("README.md 파일의 '환경 설정' 섹션을 참조하여 .env 파일을 설정하세요.")

    def get_connection(self):
        return pymysql.connect(**self.db_config)

    def get_last_count(self) -> int:
        with self.get_connection() as db:
            with db.cursor() as cursor:
                cursor.execute("SELECT MAX(count) FROM lotto")
                result = cursor.fetchone()
                return result['MAX(count)'] or 0


class LottoCrawler:
    def __init__(self):
        self.main_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin"
        self.draw_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin&drwNo="
        self.session = requests.Session()
        self.db_manager = LottoDBManager()
        self.lotto_list: List[Dict] = []

    def get_last_draw(self) -> int:
        response = self.session.get(self.main_url, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")
        meta = soup.find("meta", {"id": "desc", "name": "description"})
        if not meta:
            raise ValueError("Could not find draw information")

        content = meta['content']
        begin = content.find(" ")
        end = content.find("회")

        if begin == -1 or end == -1:
            raise ValueError("Invalid draw number format")

        return int(content[begin + 1:end])

    def crawl_draws(self, from_draw: int, to_draw: int):
        self.lotto_list.clear()

        for draw_num in range(from_draw + 1, to_draw + 1):
            url = f"{self.draw_url}{draw_num}"
            print(f"Crawling: {url}")

            try:
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "lxml")
                meta = soup.find("meta", {"id": "desc", "name": "description"})

                if not meta:
                    print(f"Skip draw {draw_num}: No data found")
                    continue

                content = meta['content']

                # Parse numbers
                numbers_start = content.find("당첨번호") + 4
                numbers_end = content.find(".", numbers_start)
                numbers = content[numbers_start:numbers_end].strip()

                # Parse winners
                winners_start = content.find("총") + 1
                winners_end = content.find("명", winners_start)
                winners = content[winners_start:winners_end].strip()

                # Parse prize
                prize_start = content.find("당첨금액") + 5
                prize_end = content.find("원", prize_start)
                prize = content[prize_start:prize_end].strip()

                self.lotto_list.append({
                    "회차": draw_num,
                    "번호": numbers,
                    "당첨자": winners,
                    "금액": prize
                })

            except requests.RequestException as e:
                print(f"Error crawling draw {draw_num}: {e}")
                continue

    def insert_draws(self):
        with self.db_manager.get_connection() as db:
            with db.cursor() as cursor:
                for draw in self.lotto_list:
                    try:
                        numbers = draw["번호"].split(",")
                        bonus = numbers[-1].split("+")

                        sql = """
                        INSERT INTO lotto (count, `1`, `2`, `3`, `4`, `5`, `6`, `7`, person, amount)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """

                        cursor.execute(sql, (
                            draw["회차"],
                            int(numbers[0]),
                            int(numbers[1]),
                            int(numbers[2]),
                            int(numbers[3]),
                            int(numbers[4]),
                            int(bonus[0]),
                            int(bonus[1]),
                            int(draw["당첨자"]),
                            draw["금액"]
                        ))

                        print(f"Inserted draw {draw['회차']}")

                    except (ValueError, IndexError) as e:
                        print(f"Error parsing draw {draw['회차']}: {e}")
                        db.rollback()
                        continue

                    except pymysql.Error as e:
                        print(f"Database error for draw {draw['회차']}: {e}")
                        db.rollback()
                        continue

                db.commit()

    def analyze_data(self):
        with self.db_manager.get_connection() as db:
            with db.cursor() as cursor:
                cursor.execute("SELECT * FROM lotto")
                results = cursor.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.to_csv('lotto_results.csv', index=False)

        # Prepare number lists
        number_columns = [str(i) for i in range(1, 7)]
        all_numbers = df[number_columns].values.flatten()

        # Plot overall distribution
        plt.figure(figsize=(15, 8))
        pd.Series(collections.Counter(all_numbers)).sort_index().plot(
            kind='bar',
            title='Overall Number Distribution'
        )
        plt.tight_layout()
        plt.savefig('overall_distribution.png')

        # Plot position-wise distributions
        for pos in range(1, 7):
            plt.figure(figsize=(15, 8))
            pd.Series(collections.Counter(df[str(pos)])).sort_index().plot(
                kind='bar',
                title=f'Numbers in Position {pos}'
            )
            plt.tight_layout()
            plt.savefig(f'position_{pos}_distribution.png')


def main():
    try:
        crawler = LottoCrawler()
        latest_draw = crawler.get_last_draw()
        db_last_draw = crawler.db_manager.get_last_count()

        print(f"Latest draw: {latest_draw}")
        print(f"Last draw in database: {db_last_draw}")

        if db_last_draw < latest_draw:
            print(f"Updating draws from {db_last_draw} to {latest_draw}")
            crawler.crawl_draws(db_last_draw, latest_draw)
            crawler.insert_draws()
            crawler.analyze_data()
        else:
            print("Database is up to date")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()