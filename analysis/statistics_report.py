"""
로또 번호 통계 분석 리포트
Phase 1: 데이터 기반 패턴/편향 검증
"""
import os
import sys

# Windows 한글 출력 인코딩 설정
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
from scipy import stats
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasources.mysql_source import MySQLDataSource


class LottoStatistics:
    """로또 번호 통계 분석 클래스"""

    def __init__(self):
        self.datasource = MySQLDataSource()
        self.numbers = None
        self.rounds = None

    def load_data(self):
        """데이터 로드"""
        self.datasource.load()
        raw_data = self.datasource.get_raw_data()
        self.numbers = raw_data[:, 1:7].astype(int)  # shape: (N, 6)
        self.rounds = raw_data[:, 0]  # 회차 번호
        print(f"총 {len(self.numbers)}회차 데이터 로드")
        return self

    def analyze_frequency(self):
        """1. 번호별 출현 빈도 분석"""
        print("\n" + "="*60)
        print("1. 번호별 출현 빈도 분석")
        print("="*60)

        # 전체 번호 카운트
        all_numbers = self.numbers.flatten()
        freq = Counter(all_numbers)

        # 기대 빈도 (균등분포 가정)
        total_draws = len(self.numbers) * 6  # 전체 뽑힌 번호 수
        expected_freq = total_draws / 45  # 각 번호의 기대 빈도

        print(f"\n총 추첨 횟수: {len(self.numbers)}회")
        print(f"전체 뽑힌 번호 수: {total_draws}개")
        print(f"각 번호 기대 빈도: {expected_freq:.1f}회")

        # 빈도 정렬
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        print("\n[상위 10개 번호]")
        for num, count in sorted_freq[:10]:
            diff = count - expected_freq
            pct = (count / expected_freq - 1) * 100
            bar = "+" * int(abs(pct) / 2) if pct > 0 else "-" * int(abs(pct) / 2)
            print(f"  {num:2d}번: {count:3d}회 (기대값 대비 {pct:+.1f}% {bar})")

        print("\n[하위 10개 번호]")
        for num, count in sorted_freq[-10:]:
            diff = count - expected_freq
            pct = (count / expected_freq - 1) * 100
            bar = "+" * int(abs(pct) / 2) if pct > 0 else "-" * int(abs(pct) / 2)
            print(f"  {num:2d}번: {count:3d}회 (기대값 대비 {pct:+.1f}% {bar})")

        return freq, expected_freq

    def chi_square_test(self, freq, expected_freq):
        """2. 카이제곱 검정 - 균등분포 여부 검증"""
        print("\n" + "="*60)
        print("2. 카이제곱 검정 (균등분포 검증)")
        print("="*60)

        # 관측 빈도 (1~45 순서로)
        observed = [freq.get(i, 0) for i in range(1, 46)]
        expected = [expected_freq] * 45

        # 카이제곱 검정
        chi2, p_value = stats.chisquare(observed, expected)

        print(f"\n카이제곱 통계량: {chi2:.2f}")
        print(f"p-value: {p_value:.6f}")
        print(f"자유도: 44")

        if p_value < 0.05:
            print("\n결론: p < 0.05 → 균등분포가 아닐 가능성 있음!")
            print("      (번호별 빈도에 유의미한 차이 존재)")
        else:
            print("\n결론: p >= 0.05 → 균등분포로 볼 수 있음")
            print("      (번호별 빈도 차이는 우연의 범위)")

        return chi2, p_value

    def analyze_recent_vs_all(self, recent_n=100):
        """3. 최근 N회차 vs 전체 비교"""
        print("\n" + "="*60)
        print(f"3. 최근 {recent_n}회차 vs 전체 비교")
        print("="*60)

        recent_numbers = self.numbers[-recent_n:].flatten()
        all_numbers = self.numbers.flatten()

        recent_freq = Counter(recent_numbers)
        all_freq = Counter(all_numbers)

        # 각 번호의 출현 비율 비교
        diffs = []
        for num in range(1, 46):
            recent_rate = recent_freq.get(num, 0) / (recent_n * 6)
            all_rate = all_freq.get(num, 0) / len(all_numbers)
            diff = recent_rate - all_rate
            diffs.append((num, diff, recent_rate, all_rate))

        # 차이가 큰 순서로 정렬
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n[최근 {recent_n}회 vs 전체 차이가 큰 번호]")
        print("번호 |  최근   |  전체   |   차이")
        print("-" * 40)
        for num, diff, recent, all_r in diffs[:10]:
            arrow = "↑" if diff > 0 else "↓"
            print(f" {num:2d}  | {recent*100:5.1f}% | {all_r*100:5.1f}% | {diff*100:+5.2f}% {arrow}")

        return diffs

    def analyze_odd_even(self):
        """4. 홀짝 비율 분석"""
        print("\n" + "="*60)
        print("4. 홀짝 비율 분석")
        print("="*60)

        odd_counts = []
        for row in self.numbers:
            odd = sum(1 for n in row if n % 2 == 1)
            odd_counts.append(odd)

        distribution = Counter(odd_counts)
        total = len(odd_counts)

        print("\n홀수 개수 | 빈도  | 비율   | 기대비율")
        print("-" * 45)

        # 이항분포 기대값 계산 (p=0.5, n=6)
        for odd in range(7):
            count = distribution.get(odd, 0)
            rate = count / total * 100
            # 이항분포 확률: C(6,k) * 0.5^6
            expected_rate = stats.binom.pmf(odd, 6, 0.5) * 100
            diff = rate - expected_rate
            print(f"   {odd}:{6-odd}   | {count:4d}  | {rate:5.1f}% | {expected_rate:5.1f}% ({diff:+.1f}%)")

        # 가장 흔한 홀짝 비율
        most_common = distribution.most_common(1)[0]
        print(f"\n가장 흔한 홀짝비율: {most_common[0]}:{6-most_common[0]} ({most_common[1]}회, {most_common[1]/total*100:.1f}%)")

        return distribution

    def analyze_high_low(self):
        """5. 고저 비율 분석 (1-22: 저, 23-45: 고)"""
        print("\n" + "="*60)
        print("5. 고저 비율 분석 (저:1-22, 고:23-45)")
        print("="*60)

        low_counts = []
        for row in self.numbers:
            low = sum(1 for n in row if n <= 22)
            low_counts.append(low)

        distribution = Counter(low_counts)
        total = len(low_counts)

        print("\n저번호수 | 빈도  | 비율")
        print("-" * 35)
        for low in range(7):
            count = distribution.get(low, 0)
            rate = count / total * 100
            print(f"  {low}:{6-low}   | {count:4d}  | {rate:5.1f}%")

        most_common = distribution.most_common(1)[0]
        print(f"\n가장 흔한 고저비율: 저{most_common[0]}:고{6-most_common[0]} ({most_common[1]}회, {most_common[1]/total*100:.1f}%)")

        return distribution

    def analyze_consecutive(self):
        """6. 연속번호 출현 빈도"""
        print("\n" + "="*60)
        print("6. 연속번호 출현 빈도")
        print("="*60)

        consecutive_counts = []
        for row in self.numbers:
            sorted_row = sorted(row)
            consec = sum(1 for i in range(5) if sorted_row[i+1] - sorted_row[i] == 1)
            consecutive_counts.append(consec)

        distribution = Counter(consecutive_counts)
        total = len(consecutive_counts)

        print("\n연속쌍수 | 빈도  | 비율")
        print("-" * 35)
        for consec in range(6):
            count = distribution.get(consec, 0)
            rate = count / total * 100
            print(f"   {consec}개   | {count:4d}  | {rate:5.1f}%")

        # 연속번호 있는 회차 비율
        has_consec = total - distribution.get(0, 0)
        print(f"\n연속번호 있는 회차: {has_consec}회 ({has_consec/total*100:.1f}%)")

        return distribution

    def analyze_ending_digits(self):
        """7. 끝자리 분포"""
        print("\n" + "="*60)
        print("7. 끝자리(일의자리) 분포")
        print("="*60)

        endings = [n % 10 for row in self.numbers for n in row]
        distribution = Counter(endings)
        total = len(endings)
        expected = total / 10

        print("\n끝자리 | 빈도  | 비율   | 기대값 대비")
        print("-" * 45)
        for digit in range(10):
            count = distribution.get(digit, 0)
            rate = count / total * 100
            diff_pct = (count / expected - 1) * 100
            print(f"   {digit}   | {count:4d}  | {rate:5.1f}% | {diff_pct:+.1f}%")

        return distribution

    def analyze_sum_range(self):
        """8. 번호 합계 분석"""
        print("\n" + "="*60)
        print("8. 번호 합계 분석")
        print("="*60)

        sums = [sum(row) for row in self.numbers]

        print(f"\n평균 합계: {np.mean(sums):.1f}")
        print(f"최소 합계: {min(sums)} (회차: {self.rounds[sums.index(min(sums))]})")
        print(f"최대 합계: {max(sums)} (회차: {self.rounds[sums.index(max(sums))]})")
        print(f"표준편차: {np.std(sums):.1f}")

        # 합계 구간별 분포
        bins = [0, 80, 100, 120, 140, 160, 180, 200, 300]
        hist, _ = np.histogram(sums, bins=bins)

        print("\n합계 구간 | 빈도  | 비율")
        print("-" * 35)
        for i in range(len(bins)-1):
            rate = hist[i] / len(sums) * 100
            print(f" {bins[i]:3d}-{bins[i+1]:3d} | {hist[i]:4d}  | {rate:5.1f}%")

        return sums

    def analyze_gaps(self):
        """9. 번호 간 간격 분석"""
        print("\n" + "="*60)
        print("9. 번호 간 간격 분석")
        print("="*60)

        all_gaps = []
        for row in self.numbers:
            sorted_row = sorted(row)
            gaps = [sorted_row[i+1] - sorted_row[i] for i in range(5)]
            all_gaps.extend(gaps)

        distribution = Counter(all_gaps)
        total = len(all_gaps)

        print("\n간격 | 빈도   | 비율")
        print("-" * 35)
        for gap in range(1, 16):
            count = distribution.get(gap, 0)
            rate = count / total * 100
            bar = "*" * int(rate)
            print(f" {gap:2d}  | {count:5d}  | {rate:5.1f}% {bar}")

        # 나머지
        other = sum(c for g, c in distribution.items() if g >= 16)
        if other:
            print(f" 16+ | {other:5d}  | {other/total*100:5.1f}%")

        print(f"\n평균 간격: {np.mean(all_gaps):.2f}")

        return distribution

    def find_patterns(self):
        """10. 반복되는 패턴 탐색"""
        print("\n" + "="*60)
        print("10. 반복 패턴 탐색")
        print("="*60)

        # 동일 번호 조합 찾기
        combos = [tuple(sorted(row)) for row in self.numbers]
        combo_counts = Counter(combos)

        repeated = [(combo, count) for combo, count in combo_counts.items() if count > 1]

        if repeated:
            print(f"\n동일 번호 조합 반복: {len(repeated)}개")
            for combo, count in sorted(repeated, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {combo} - {count}회 출현")
        else:
            print("\n동일 번호 조합 반복 없음")

        # 2개 번호 조합 빈도
        pair_counts = Counter()
        for row in self.numbers:
            sorted_row = sorted(row)
            for i in range(6):
                for j in range(i+1, 6):
                    pair_counts[(sorted_row[i], sorted_row[j])] += 1

        print("\n[가장 자주 나오는 2개 조합]")
        for pair, count in pair_counts.most_common(10):
            expected = len(self.numbers) * (6*5/2) / (45*44/2)  # 기대값
            diff = (count / expected - 1) * 100
            print(f"  {pair[0]:2d} - {pair[1]:2d}: {count}회 (기대값 대비 {diff:+.1f}%)")

        return combo_counts, pair_counts

    def run_full_analysis(self):
        """전체 분석 실행"""
        self.load_data()

        freq, expected = self.analyze_frequency()
        chi2, p_value = self.chi_square_test(freq, expected)
        self.analyze_recent_vs_all()
        self.analyze_odd_even()
        self.analyze_high_low()
        self.analyze_consecutive()
        self.analyze_ending_digits()
        self.analyze_sum_range()
        self.analyze_gaps()
        self.find_patterns()

        # 결론
        print("\n" + "="*60)
        print("분석 결론")
        print("="*60)

        if p_value < 0.05:
            print("\n[주목] 번호별 빈도가 균등분포가 아닐 가능성 있음 (p < 0.05)")
            print("      → 특정 번호가 통계적으로 더 자주/덜 나올 수 있음")
        else:
            print("\n번호별 빈도는 균등분포 범위 내 (p >= 0.05)")
            print("      → 특정 번호 편향 발견 안됨")

        print("\n다음 단계: 위 패턴들을 피처로 활용하여 모델 개선")

        return {
            'chi2': chi2,
            'p_value': p_value,
            'frequency': freq,
        }


if __name__ == "__main__":
    analyzer = LottoStatistics()
    results = analyzer.run_full_analysis()
