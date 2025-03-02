import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pymysql
from typing import List, Tuple, Dict, Set
import os
import math
from dotenv import load_dotenv
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import traceback
# 특성 중요도 분석을 위한 라이브러리 추가
from sklearn.inspection import permutation_importance
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error
import sqlite3
import random
from datetime import datetime
import seaborn as sns
# SHAP 분석을 위한 라이브러리 추가
import shap
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.font_manager as fm
import platform
# PDF 보고서 생성을 위한 라이브러리 추가
from fpdf import FPDF
import glob
from PIL import Image
warnings.filterwarnings('ignore')

# 한글 폰트 설정
system_platform = platform.system()
if system_platform == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system_platform == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    # 나눔고딕이나 다른 한글 폰트가 설치되어 있다고 가정
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

load_dotenv()

# Mac Silicon에서 MPS (Metal Performance Shaders) 사용 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"PyTorch 실행 장치: {device}")

class CombinationLottoModel(nn.Module):
    """
    개선된 로또 예측 모델 - 멀티라벨 접근법 (순서 없는 조합)으로 번호 예측
    45개 번호 각각의 포함 여부를 예측 (이진 분류 문제)
    
    입력 특성 (30개):
    - 기본 번호 데이터 (6개)
    - 기본 통계적 특성: 평균, 표준편차, 최소값, 최대값 (4개)
    - 홀짝/저고 분포: 짝수 비율, 낮은 번호 비율 (2개)
    - 간격 정보 (5개)
    - 범위 정보 (1개)
    - 이전 회차와의 중복 정보 (1개)
    - 숫자 간 표준편차 (1개)
    - 범위별 번호 분포: 1-15, 16-30, 31-45 구간 비율 (3개)
    - 연속된 번호 비율 (1개)
    - 번호 합계 정규화 (1개)
    - 홀짝 교차 패턴 (1개)
    - 증가/감소 패턴 (1개)
    - 엔트로피 (1개)
    - 첨도 (1개)
    - 간격의 균일성 (1개)
    
    추가된 기능:
    - Self-Attention 메커니즘: 시퀀스 내 중요한 시점 강조
    """
    def __init__(self, input_dim=30, hidden_size=48, dropout_rate=0.4):
        super(CombinationLottoModel, self).__init__()
        
        # 단순화된 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Self-Attention 레이어 추가
        lstm_output_size = hidden_size * 2  # 양방향이므로 2배
        self.attention_query = nn.Linear(lstm_output_size, lstm_output_size)
        self.attention_key = nn.Linear(lstm_output_size, lstm_output_size)
        self.attention_value = nn.Linear(lstm_output_size, lstm_output_size)
        
        # 드롭아웃 (비율 증가)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 완전 연결 레이어 (간소화)
        self.fc1 = nn.Linear(lstm_output_size, 96)
        self.fc2 = nn.Linear(96, 45)  # 45개 번호에 대한 포함 확률 (이진 분류)
        
        # 어텐션 가중치 저장을 위한 속성
        self.attention_weights = None
        
    def forward(self, x):
        # LSTM 처리
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Self-Attention 메커니즘 적용
        batch_size, seq_len, hidden_size = lstm_out.size()
        
        # Q, K, V 계산
        Q = self.attention_query(lstm_out)  # [batch_size, seq_len, hidden_size]
        K = self.attention_key(lstm_out)    # [batch_size, seq_len, hidden_size]
        V = self.attention_value(lstm_out)  # [batch_size, seq_len, hidden_size]
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_size ** 0.5)  # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 어텐션 가중치 저장 (추후 분석용)
        self.attention_weights = attention_weights.detach()
        
        # 가중 합계 계산
        context = torch.matmul(attention_weights, V)  # [batch_size, seq_len, hidden_size]
        
        # 어텐션 출력의 마지막 타임스텝 사용
        last_output = context[:, -1, :]  # [batch_size, hidden_size]
        
        # 드롭아웃 적용
        x = self.dropout(last_output)
        
        # 완전 연결 레이어
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 최종 출력 - 45개 번호에 대한 이진 로짓
        logits = self.fc2(x)  # [batch_size, 45]
        
        return logits
        
    def get_attention_weights(self):
        """어텐션 가중치 반환 (시각화 및 분석용)"""
        return self.attention_weights

class LottoAIPredictor:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'db': os.getenv('DB_NAME'),
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.sequence_length = 10
        self.model = None
        
    def get_data(self):
        """데이터베이스에서 로또 당첨 번호 데이터 가져오기"""
        try:
            conn = pymysql.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                db=self.db_config['db'],
                charset='utf8mb4'
            )
            
            with conn.cursor() as cursor:
                sql = "SELECT count, `1`,`2`,`3`,`4`,`5`,`6` FROM lotto ORDER BY count ASC"
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                df = pd.DataFrame(rows, columns=['count', '1', '2', '3', '4', '5', '6'])
                print(f"데이터베이스에서 {len(df)}개의 로또 당첨 번호를 로드했습니다.")
                return df

        except Exception as e:
            print(f"데이터베이스 연결 오류: {str(e)}")
            return pd.DataFrame()

    def prepare_features(self, df=None):
        """
        간소화된 특성 추출 - 필수적인 특성만 유지
        """
        if df is None:
            df = self.get_data()
        
        # 번호 컬럼 추출
        number_cols = [str(i) for i in range(1, 7)]
        data = df[number_cols].values
        
        # 시퀀스 데이터 준비
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]  # 6개 번호 세트
            
            sequences.append(seq)
            targets.append(target)
        
        # 시퀀스를 numpy 배열로 변환
        X = np.array(sequences)
        
        # 타겟을 원-핫 인코딩으로 변환 (45개 번호 중 6개만 1)
        y_onehot = np.zeros((len(targets), 45))
        for i, target_set in enumerate(targets):
            for num in target_set:
                y_onehot[i, num-1] = 1  # 1-based to 0-based
        
        # 특성 확장: 꼭 필요한 특성만 추가
        batch_size, seq_len, num_numbers = X.shape
        essential_features = np.zeros((batch_size, seq_len, 30))  # 기본 6 + 추가 특성 24개로 확장
        
        for i in range(batch_size):
            for j in range(seq_len):
                # 기본 번호 정복사
                essential_features[i, j, :num_numbers] = X[i, j]
                
                # 현재 번호 세트
                numbers = X[i, j]
                
                # 1. 기본 통계적 특성 (4개)
                essential_features[i, j, 6] = np.mean(numbers) / 45  # 평균
                essential_features[i, j, 7] = np.std(numbers) / 45   # 표준편차
                essential_features[i, j, 8] = np.min(numbers) / 45   # 최소값
                essential_features[i, j, 9] = np.max(numbers) / 45   # 최대값
                
                # 2. 홀짝/저고 분포 (4개)
                even_count = sum(1 for n in numbers if n % 2 == 0)
                essential_features[i, j, 10] = even_count / len(numbers)  # 짝수 비율
                
                low_count = sum(1 for n in numbers if n <= 22)
                essential_features[i, j, 11] = low_count / len(numbers)  # 낮은 번호 비율
                
                # 3. 간격 정보 (5개)
                sorted_numbers = sorted(numbers)
                for k in range(len(sorted_numbers) - 1):
                    gap = sorted_numbers[k + 1] - sorted_numbers[k]
                    essential_features[i, j, 12 + k] = gap / 45  # 정규화된 간격
                
                # 4. 범위 정보 (1개)
                essential_features[i, j, 17] = (sorted_numbers[-1] - sorted_numbers[0]) / 45
                
                # 5. 이전 회차와 중복 정보 (j > 0일 때) (1개)
                if j > 0:
                    prev_numbers = X[i, j - 1]
                    overlap_count = sum(1 for n in numbers if n in prev_numbers)
                    essential_features[i, j, 18] = overlap_count / len(numbers)
                
                # 6. 숫자 간 표준편차 (1개)
                essential_features[i, j, 19] = np.std(np.diff(sorted_numbers)) / 45
                
                # 7. 범위별 번호 분포 특성 (3개)
                range1_count = sum(1 for n in numbers if 1 <= n <= 15)
                range2_count = sum(1 for n in numbers if 16 <= n <= 30)
                range3_count = sum(1 for n in numbers if 31 <= n <= 45)
                
                essential_features[i, j, 20] = range1_count / len(numbers)  # 1-15 범위 비율
                essential_features[i, j, 21] = range2_count / len(numbers)  # 16-30 범위 비율
                essential_features[i, j, 22] = range3_count / len(numbers)  # 31-45 범위 비율
                
                # 8. 연속된 번호 특성 (1개)
                consecutive_count = 0
                for k in range(len(sorted_numbers) - 1):
                    if sorted_numbers[k + 1] - sorted_numbers[k] == 1:
                        consecutive_count += 1
                essential_features[i, j, 23] = consecutive_count / (len(sorted_numbers) - 1)  # 연속된 번호 비율
                
                # 9. 번호 합계 특성 (1개)
                numbers_sum = sum(numbers)
                essential_features[i, j, 24] = numbers_sum / (45 * 6)  # 번호 합계 정규화 (최대 가능 합계로 나눔)
                
                # 10. 홀짝 교차 패턴 (1개)
                odd_even_pattern = 0
                for k in range(len(sorted_numbers) - 1):
                    if (sorted_numbers[k] % 2) != (sorted_numbers[k + 1] % 2):
                        odd_even_pattern += 1
                essential_features[i, j, 25] = odd_even_pattern / (len(sorted_numbers) - 1)  # 홀짝 교차 비율
                
                # 11. 증가/감소 패턴 (1개)
                # 번호 간 증가량의 변화 패턴 (증가하다 감소하는 패턴 등)
                diffs = np.diff(sorted_numbers)
                direction_changes = 0
                if len(diffs) > 1:  # 차이값이 2개 이상 있을 때만 계산
                    for k in range(1, len(diffs)):  # 1부터 시작하여 이전 값과 비교
                        if (diffs[k] > diffs[k-1]) or (diffs[k] < diffs[k-1]):
                            direction_changes += 1
                    essential_features[i, j, 26] = direction_changes / (len(diffs) - 1)
                else:
                    essential_features[i, j, 26] = 0  # 충분한 데이터가 없는 경우
                
                # 12. 엔트로피 (번호 선택의 무작위성) (1개)
                # 번호 분포에 대한 엔트로피를 측정 (0-1 사이 값)
                bins = np.zeros(45)
                for num in numbers:
                    bins[num-1] += 1  # 1-based to 0-based index
                # 확률이 0인 경우를 피하기 위해 작은 값 추가
                probs = bins + 0.01
                probs = probs / np.sum(probs)
                entropy_val = stats.entropy(probs) / np.log(45)  # 정규화된 엔트로피
                essential_features[i, j, 27] = entropy_val
                
                # 13. 첨도(Kurtosis) - 분포의 뾰족한 정도 (1개)
                try:
                    kurt = stats.kurtosis(numbers)
                    essential_features[i, j, 28] = kurt / 10  # 스케일링
                except:
                    essential_features[i, j, 28] = 0
                
                # 14. 번호 간 간격의 균일성 (1개)
                # 간격이 균일할수록 1에 가까움, 불균일할수록 0에 가까움
                if len(diffs) > 0:
                    # 간격의 표준편차가 작을수록 균일함을 의미
                    gap_uniformity = 1 - (np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0)
                    essential_features[i, j, 29] = max(0, min(1, gap_uniformity))  # 0-1 사이로 제한
                else:
                    essential_features[i, j, 29] = 0
        
        print(f"특성 차원: {essential_features.shape}")
        return essential_features, y_onehot, targets  # 원래 타겟도 함께 반환

    def get_device(self):
        """사용 가능한 최적의 장치(CPU/GPU/MPS) 반환"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def build_model(self, feature_dim):
        """개선된 모델 구축 - 조합 기반 접근법"""
        device = self.get_device()
        
        # 입력 특성 차원을 기반으로 모델 초기화
        self.model = CombinationLottoModel(
            input_dim=feature_dim,
            hidden_size=48,  # 크기 감소
            dropout_rate=0.4  # 드롭아웃 증가
        ).to(device)
        
        print(f"모델 구축 완료 (장치: {device})")
        return self.model

    def save_model(self, filename):
        """모델을 파일로 저장"""
        if self.model is not None:
            torch.save(self.model.state_dict(), filename)
            print(f"모델이 '{filename}'로 저장되었습니다.")
        else:
            print("저장할 모델이 없습니다.")

    def load_model(self, filename):
        """모델을 파일로부터 로드"""
        device = self.get_device()
        
        # 데이터 준비를 통해 입력 차원 결정
        X, _, _ = self.prepare_features(self.get_data())
        feature_dim = X.shape[2]
        
        # 모델 초기화 후 가중치 로드
        self.model = CombinationLottoModel(
            input_dim=feature_dim,
            hidden_size=48,
            dropout_rate=0.4
        ).to(device)
        
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.model.eval()
        print(f"모델이 '{filename}'에서 로드되었습니다.")
        return self.model

    def train_with_timeseries_cv(self):
        """시계열 교차 검증을 사용한 모델 학습"""
        print("특성 준비 중...")
        X, y_onehot, targets = self.prepare_features(self.get_data())
        
        # 입력 특성 차원 계산
        feature_dim = X.shape[2]
        
        # 시계열 교차 검증 설정
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X)*0.2))
        
        device = self.get_device()
        
        # 손실 함수와 최적 검증 손실 초기화
        best_val_loss = float('inf')
        best_model_state = None
        
        fold = 1
        for train_index, val_index in tscv.split(X):
            print(f"\n===== 시계열 교차 검증 Fold {fold}/5 =====")
            fold += 1
            
            # 현재 폴드의 학습/검증 데이터 분할
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y_onehot[train_index], y_onehot[val_index]
            targets_train, targets_val = [targets[i] for i in train_index], [targets[i] for i in val_index]
            
            # 모델 구축 (또는 재설정)
            self.build_model(feature_dim)
            
            # 손실 함수, 옵티마이저 정의
            criterion = nn.BCEWithLogitsLoss()  # 이진 분류에 적합
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # 학습률 스케줄러
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # 조기 종료
            early_stopping = EarlyStopping(patience=15, verbose=True)
            
            # 데이터 로더 설정
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), 
                torch.FloatTensor(y_train)  # BCEWithLogitsLoss는 float 타겟 필요
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), 
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 학습 시작
            print("현재 폴드 학습 시작...")
            n_epochs = 100
            
            train_losses = []
            val_losses = []
            
            for epoch in range(n_epochs):
                # 학습 모드
                self.model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # 예측
                    optimizer.zero_grad()
                    logits = self.model(batch_X)
                    
                    # 손실 계산 (BCEWithLogitsLoss)
                    loss = criterion(logits, batch_y)
                    
                    # 역전파 및 최적화
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # 검증 모드
                self.model.eval()
                val_loss = 0
                
                # 번호 일치 평가를 위한 변수
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        
                        logits = self.model(batch_X)
                        
                        # 손실 계산
                        loss = criterion(logits, batch_y)
                        val_loss += loss.item()
                        
                        # 상위 6개 확률에 해당하는 번호 선택
                        probs = torch.sigmoid(logits)
                        _, top_indices = torch.topk(probs, k=6, dim=1)
                        
                        # 예측과 실제 타겟 수집
                        all_predictions.extend(top_indices.cpu().numpy() + 1)  # 0-based to 1-based
                        
                        # 원-핫 인코딩 타겟을 실제 번호로 변환
                        target_indices = torch.nonzero(batch_y == 1, as_tuple=True)
                        batch_size = batch_y.size(0)
                        for i in range(batch_size):
                            batch_targets = target_indices[1][target_indices[0] == i].cpu().numpy() + 1
                            all_targets.append(batch_targets)
                
                # 에포크 손실 계산
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # 번호 일치도 평가
                match_stats = self.evaluate_lotto_matches(all_predictions, all_targets)
                
                # 학습 진행 상황 출력
                print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                print(f"평가: 정확한 예측(6개): {match_stats['exact']:.2f}%, 3개+ 일치: {match_stats['three_plus']:.2f}%")
                
                # 학습률 스케줄러 업데이트
                scheduler.step(val_loss)
                
                # 조기 종료 확인
                early_stopping(val_loss, self.model)
                
                # 최고 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                if early_stopping.early_stop:
                    print(f"조기 종료 (에포크 {epoch+1})")
                    break
        
        # 최고 성능 모델 로드
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.save_model('combination_lotto_model.pth')
            print(f"최고 성능 모델 저장 완료 (검증 손실: {best_val_loss:.4f})")
        
        return train_losses, val_losses

    def evaluate_lotto_matches(self, predicted_nums, actual_nums):
        """로또 번호 일치 개수 기반 평가"""
        matches = {i: 0 for i in range(7)}  # 0~6개 일치
        total = len(predicted_nums)
        
        for i, pred in enumerate(predicted_nums):
            # 셋으로 변환하여 교집합 계산
            pred_set = set(pred)
            actual_set = set(actual_nums[i])
            match_count = len(pred_set.intersection(actual_set))
            matches[match_count] += 1
        
        # 평가 지표 계산
        match_stats = {}
        
        # 정확히 일치하는 비율 (6개 모두 맞춤)
        match_stats['exact'] = (matches[6] / total) * 100
        
        # 각 일치 개수별 비율
        for i in range(7):
            match_stats[f'match_{i}'] = (matches[i] / total) * 100
        
        # 3개 이상 일치 비율 (당첨 기준)
        three_plus = sum(matches[i] for i in range(3, 7))
        match_stats['three_plus'] = (three_plus / total) * 100
        
        return match_stats

    def predict_next_numbers(self, num_combinations=5, temperature=0.8):
        """
        모델을 사용하여 다음 로또 번호 조합 예측
        - 45개 번호의 확률 기반 선택
        """
        print("\n2. 예측 수행 중...")
        
        # 모델이 없으면 모델 로드
        if self.model is None:
            try:
                self.load_model('combination_lotto_model.pth')
            except FileNotFoundError:
                print("저장된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
                return []
        
        device = self.get_device()
        self.model.eval()
        
        # 가장 최근 데이터 가져오기
        X, _, _ = self.prepare_features(self.get_data())
        latest_data = X[-1:]  # 가장 최근 데이터
        
        # 텐서로 변환하여 예측
        with torch.no_grad():
            input_tensor = torch.FloatTensor(latest_data).to(device)
            logits = self.model(input_tensor)  # [batch=1, num_classes=45]
            
            # sigmoid로 확률 변환
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        combinations = []
        
        # 다양한 조합 생성
        for _ in range(num_combinations):
            # 확률에 온도 적용
            if temperature != 1.0:
                adjusted_probs = np.power(probs, 1.0 / temperature)
                adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
            else:
                adjusted_probs = probs / np.sum(probs)
            
            # 조합 빌드
            selected_numbers = []
            remaining_probs = adjusted_probs.copy()
            
            while len(selected_numbers) < 6:
                # NaN 확인 및 처리
                if np.isnan(remaining_probs).any() or np.sum(remaining_probs) == 0:
                    remaining_probs = np.ones(45)
                    for num in selected_numbers:
                        remaining_probs[num-1] = 0
                    remaining_probs = remaining_probs / np.sum(remaining_probs)
                
                # 확률적 샘플링 (num+1: 0-based to 1-based)
                num = np.random.choice(45, p=remaining_probs) + 1
                
                if num not in selected_numbers:
                    selected_numbers.append(num)
                
                # 선택된 번호 제외
                remaining_probs[num-1] = 0
                if np.sum(remaining_probs) > 0:
                    remaining_probs = remaining_probs / np.sum(remaining_probs)
            
            # 오름차순 정렬
            selected_numbers.sort()
            combinations.append(selected_numbers)
        
        return combinations

    def analyze_feature_importance(self, top_n=10):
        """
        개선된 특성 중요도 분석 (Permutation Importance)
        - 시퀀스 정보를 효과적으로 활용
        - 로또 특화 평가 메트릭 적용
        - 특성 그룹별 분석 추가
        """
        try:
            print("\n4. 특성 중요도 분석 (Permutation Importance)")
            
            # 데이터 준비
            numbers_df = self.get_data()
            X_train, y_train, targets = self.prepare_features(numbers_df)
            
            print(f"특성 차원: {X_train.shape}")
            
            # 시퀀스 활용 방식 개선 - 전체 시퀀스 정보 활용
            # 1. 시퀀스를 펼쳐서 특성으로 사용 (배치, 시퀀스*특성)
            seq_len, feature_dim = X_train.shape[1], X_train.shape[2]
            X_flattened = X_train.reshape(X_train.shape[0], -1)  # 시퀀스 전체를 펼침
            
            # scikit-learn 호환을 위한 개선된 모델 래퍼
            class EnhancedModelWrapper:
                def __init__(self, model, sequence_length=10, feature_dim=30):
                    self.model = model
                    self.seq_length = sequence_length
                    self.feature_dim = feature_dim
                
                def fit(self, X, y):
                    # scikit-learn 인터페이스 호환을 위한 더미 메서드
                    return self
                
                def predict(self, X):
                    # 입력을 3D 텐서로 변환 (배치, 시퀀스 길이, 특성)
                    batch_size = X.shape[0]
                    X_reshaped = X.reshape(batch_size, self.seq_length, self.feature_dim)
                    
                    # 텐서 변환
                    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
                    
                    # 모델의 첫 번째 파라미터가 있는 장치를 확인하거나 CPU 사용
                    device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device('cpu')
                    X_tensor = X_tensor.to(device)
                    
                    # 모델 평가 모드 설정
                    self.model.eval()
                    
                    with torch.no_grad():
                        y_pred = self.model(X_tensor)
                        # 예측 결과를 넘파이 배열로 변환
                        return y_pred.cpu().numpy()
                
                def score(self, X, y):
                    """
                    로또 특화 평가 메트릭 - 몇 개의 번호를 맞추는지에 기반한 점수
                    """
                    y_pred = self.predict(X)
                    
                    # 각 예측에서 가장 가능성이 높은 6개 번호 선택
                    top_indices = np.argsort(y_pred, axis=1)[:, -6:]
                    
                    # 실제 번호와 비교하여 일치하는 개수 계산
                    match_counts = []
                    for i in range(len(y)):
                        true_indices = np.where(y[i] == 1)[0]
                        matches = np.sum(np.isin(top_indices[i], true_indices))
                        match_counts.append(matches)
                    
                    # 평균 일치 개수 (0-6 사이) / 6 -> 0-1 사이 점수로 변환
                    avg_match_ratio = np.mean(match_counts) / 6
                    return avg_match_ratio  # 높을수록 좋음
            
            # 개선된 모델 래퍼 생성
            model_wrapper = EnhancedModelWrapper(
                self.model, 
                sequence_length=seq_len, 
                feature_dim=feature_dim
            )
            
            print("특성 중요도 계산 중... (시간이 소요될 수 있습니다)")
            
            # 특성 중요도 계산
            r = permutation_importance(
                model_wrapper, X_flattened, y_train,
                n_repeats=5,
                random_state=42,
                scoring=None  # 로또 특화 점수 메서드 사용
            )
            
            # 특성 이름 정의
            feature_names = [
                "번호1", "번호2", "번호3", "번호4", "번호5", "번호6",
                "평균", "표준편차", "최소값", "최대값",
                "짝수비율", "낮은번호비율", 
                "간격1", "간격2", "간격3", "간격4", "간격5",
                "범위", "이전회차중복", "숫자간표준편차",
                "1-15구간비율", "16-30구간비율", "31-45구간비율", 
                "연속번호비율", "번호합계", "홀짝교차패턴",
                "증감패턴", "엔트로피", "첨도", "간격균일성"
            ]
            
            # 각 시점별 특성 이름 생성 (시퀀스 * 특성)
            seq_feature_names = []
            for t in range(seq_len):
                for f in feature_names:
                    seq_feature_names.append(f"t-{seq_len-t}_{f}")
            
            # 결과 정렬 및 시각화
            importance_scores = pd.Series(r.importances_mean, index=seq_feature_names)
            
            # 특성 그룹 정의
            feature_groups = {
                "기본통계량": ["평균", "표준편차", "최소값", "최대값", "범위", "숫자간표준편차", "첨도"],
                "분포특성": ["짝수비율", "낮은번호비율", "1-15구간비율", "16-30구간비율", "31-45구간비율", "번호합계", "엔트로피"],
                "패턴특성": ["홀짝교차패턴", "증감패턴", "연속번호비율"],
                "간격정보": ["간격1", "간격2", "간격3", "간격4", "간격5", "간격균일성"],
                "번호특성": ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"],
                "시간특성": ["이전회차중복"]
            }
            
            # 그룹별 중요도 계산
            group_importance = defaultdict(float)
            for i, score in enumerate(importance_scores):
                feature = seq_feature_names[i]
                for group_name, group_features in feature_groups.items():
                    for group_feature in group_features:
                        if group_feature in feature:
                            group_importance[group_name] += score
                            break
            
            # 그룹 중요도 정규화
            total_group_importance = sum(group_importance.values())
            for group in group_importance:
                group_importance[group] /= total_group_importance
            
            # 시간대별 중요도
            time_importance = defaultdict(float)
            for i, score in enumerate(importance_scores):
                feature = seq_feature_names[i]
                time_point = feature.split('_')[0]  # t-1, t-2, ...
                time_importance[time_point] += score
            
            # 시간 중요도 정규화
            total_time_importance = sum(time_importance.values())
            for time_point in time_importance:
                time_importance[time_point] /= total_time_importance
            
            # 개별 특성 중요도
            # t-1 시점(가장 최근)의 특성만으로 필터링하여 상위 특성 출력
            t1_features = {name: score for name, score in zip(seq_feature_names, r.importances_mean) 
                          if "t-1_" in name and not any(num in name for num in ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"])}
            t1_features = {name.replace("t-1_", ""): score for name, score in t1_features.items()}
            sorted_t1_features = {k: v for k, v in sorted(t1_features.items(), key=lambda item: item[1], reverse=True)}
            
            # 번호1~6 제외하고 가장 최근 시점의 중요도만 표시
            filtered_importance = {k: v for k, v in sorted_t1_features.items() if not k.startswith("번호")}
            
            # 상위 N개 특성 출력
            print("\n=== Permutation 기반 특성 중요도 순위 (번호 특성 제외) ===")
            for i, (name, score) in enumerate(list(filtered_importance.items())[:top_n], 1):
                print(f"{i}. {name}: {score:.6f}")
            
            # 특성 그룹별 중요도 출력
            print("\n=== 특성 그룹별 중요도 ===")
            sorted_group_importance = {k: v for k, v in sorted(group_importance.items(), key=lambda item: item[1], reverse=True)}
            for i, (group, score) in enumerate(sorted_group_importance.items(), 1):
                if group != "번호특성":  # 번호 특성 그룹 제외
                    print(f"{i}. {group}: {score:.6f}")
            
            # 시간대별 중요도 출력
            print("\n=== 시간대별 중요도 ===")
            sorted_time_importance = {k: v for k, v in sorted(time_importance.items(), key=lambda item: item[0])}  # 시간 순서대로
            for time_point, score in sorted_time_importance.items():
                print(f"{time_point}: {score:.6f}")
            
            # 1. 개별 특성 중요도 시각화
            plt.figure(figsize=(10, 8))
            pd.Series(filtered_importance).head(top_n).plot(kind='barh')
            plt.title('특성 중요도 (Permutation Importance, 번호 특성 제외)')
            plt.xlabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('permutation_importance.png')
            plt.close()
            
            # 2. 특성 그룹별 중요도 시각화
            filtered_groups = {k: v for k, v in sorted_group_importance.items() if k != "번호특성"}
            plt.figure(figsize=(10, 6))
            plt.barh(list(filtered_groups.keys()), list(filtered_groups.values()))
            plt.title('특성 그룹별 중요도 (Permutation Importance)')
            plt.xlabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('group_importance.png')
            plt.close()
            
            # 3. 시간대별 중요도 시각화
            plt.figure(figsize=(10, 6))
            plt.bar(list(sorted_time_importance.keys()), list(sorted_time_importance.values()))
            plt.title('시간대별 특성 중요도')
            plt.xlabel('시간 포인트')
            plt.ylabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('time_importance.png')
            plt.close()
            
            print(f"\nPermutation 중요도 그래프가 'permutation_importance.png'로 저장되었습니다.")
            print(f"그룹별 중요도 그래프가 'group_importance.png'로 저장되었습니다.")
            print(f"시간대별 중요도 그래프가 'time_importance.png'로 저장되었습니다.")
            
        except Exception as e:
            print(f"특성 중요도 분석 중 오류 발생: {str(e)}")
            traceback.print_exc()
        
        # 모델 가중치 기반 특성 중요도 분석도 함께 수행
        self.analyze_weight_importance(top_n=top_n)
        
        # SHAP 분석 추가
        self.analyze_with_shap(top_n=top_n)

    def analyze_weight_importance(self, top_n=10):
        """
        개선된 가중치 기반 특성 중요도 분석
        - LSTM의 모든 가중치(input, forget, cell, output gates) 활용
        - 특성별 중요도 계산 및 시각화
        """
        try:
            print("\n5. 모델 가중치 기반 특성 중요도 분석")
            
            # 모델 구조에서 LSTM 레이어 찾기
            lstm_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, nn.LSTM):
                    lstm_layer = module
                    break
            
            if lstm_layer is None:
                print("LSTM 레이어를 찾을 수 없습니다.")
                return
            
            # LSTM의 weight_ih_l0는 [hidden_size * 4, input_size] 형태
            # 4개 게이트(input, forget, cell, output)에 대한 가중치를 포함
            weight_ih = lstm_layer.weight_ih_l0.detach().cpu().numpy()
            
            # 특성 이름 정의
            feature_names = [
                "번호1", "번호2", "번호3", "번호4", "번호5", "번호6",
                "평균", "표준편차", "최소값", "최대값",
                "짝수비율", "낮은번호비율", 
                "간격1", "간격2", "간격3", "간격4", "간격5",
                "범위", "이전회차중복", "숫자간표준편차",
                "1-15구간비율", "16-30구간비율", "31-45구간비율", 
                "연속번호비율", "번호합계", "홀짝교차패턴",
                "증감패턴", "엔트로피", "첨도", "간격균일성"
            ]
            
            # LSTM 가중치 행렬 형태 확인
            hidden_size = lstm_layer.hidden_size
            input_size = weight_ih.shape[1]
            
            print(f"LSTM 가중치 형태: {weight_ih.shape}")
            print(f"입력 크기: {input_size}, 은닉 크기: {hidden_size}")
            
            # 게이트별 가중치 분리
            input_gate = weight_ih[:hidden_size, :]
            forget_gate = weight_ih[hidden_size:2*hidden_size, :]
            cell_gate = weight_ih[2*hidden_size:3*hidden_size, :]
            output_gate = weight_ih[3*hidden_size:, :]
            
            # 각 게이트별 특성 중요도 계산
            gate_names = ["입력 게이트", "망각 게이트", "셀 게이트", "출력 게이트"]
            gate_weights = [input_gate, forget_gate, cell_gate, output_gate]
            
            # 게이트별 중요도 계산 및 저장
            gate_importance = {}
            
            for gate_name, gate_weight in zip(gate_names, gate_weights):
                # 게이트 가중치의 절대값 합계로 특성 중요도 계산
                importance = np.abs(gate_weight).sum(axis=0)
                
                # 입력 특성이 30개 이상인 경우 (여러 시점의 데이터를 사용하는 경우)
                if len(importance) > len(feature_names):
                    # 마지막 시점(가장 최근)의 특성에 대한 중요도만 추출
                    importance = importance[-len(feature_names):]
                
                # 중요도 정규화 (합이 1이 되도록)
                importance = importance / importance.sum()
                
                # 게이트별 특성 중요도 저장
                gate_importance[gate_name] = dict(zip(feature_names, importance))
            
            # 전체 특성 중요도 계산 (모든 게이트의 가중치 평균)
            all_importance = np.zeros(len(feature_names))
            for gate_imp in gate_importance.values():
                all_importance += np.array(list(gate_imp.values()))
            
            all_importance = all_importance / len(gate_importance)
            overall_importance = dict(zip(feature_names, all_importance))
            
            # 번호1~6 제외하고 중요도 필터링
            filtered_importance = {k: v for k, v in overall_importance.items() if not k.startswith("번호")}
            
            # 중요도 기준으로 정렬
            sorted_importance = {k: v for k, v in sorted(filtered_importance.items(), key=lambda item: item[1], reverse=True)}
            
            # 상위 N개 특성 출력
            print("\n=== 모델 가중치 기반 특성 중요도 순위 (번호 특성 제외) ===")
            for i, (name, score) in enumerate(list(sorted_importance.items())[:top_n], 1):
                print(f"{i}. {name}: {score:.6f}")
            
            # 게이트별 중요도 출력
            for gate_name, gate_imp in gate_importance.items():
                filtered_gate_imp = {k: v for k, v in gate_imp.items() if not k.startswith("번호")}
                sorted_gate_imp = {k: v for k, v in sorted(filtered_gate_imp.items(), key=lambda item: item[1], reverse=True)}
                
                print(f"\n=== {gate_name} 기반 특성 중요도 순위 (상위 5개) ===")
                for i, (name, score) in enumerate(list(sorted_gate_imp.items())[:5], 1):
                    print(f"{i}. {name}: {score:.6f}")
            
            # 특성 그룹 정의
            feature_groups = {
                "기본통계량": ["평균", "표준편차", "최소값", "최대값", "범위", "숫자간표준편차", "첨도"],
                "분포특성": ["짝수비율", "낮은번호비율", "1-15구간비율", "16-30구간비율", "31-45구간비율", "번호합계", "엔트로피"],
                "패턴특성": ["홀짝교차패턴", "증감패턴", "연속번호비율"],
                "간격정보": ["간격1", "간격2", "간격3", "간격4", "간격5", "간격균일성"],
                "번호특성": ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"],
                "시간특성": ["이전회차중복"]
            }
            
            # 그룹별 중요도 계산
            group_importance = defaultdict(float)
            for feature, score in filtered_importance.items():
                for group_name, group_features in feature_groups.items():
                    if feature in group_features:
                        group_importance[group_name] += score
                        break
            
            # 그룹 중요도 정규화
            total_group_importance = sum(group_importance.values())
            for group in group_importance:
                group_importance[group] /= total_group_importance
            
            # 그룹별 중요도 출력
            print("\n=== 특성 그룹별 중요도 ===")
            sorted_group_importance = {k: v for k, v in sorted(group_importance.items(), key=lambda item: item[1], reverse=True)}
            for i, (group, score) in enumerate(sorted_group_importance.items(), 1):
                print(f"{i}. {group}: {score:.6f}")
            
            # 1. 개별 특성 중요도 시각화
            plt.figure(figsize=(10, 8))
            pd.Series(sorted_importance).head(top_n).plot(kind='barh')
            plt.title('모델 가중치 기반 특성 중요도 (번호 특성 제외)')
            plt.xlabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('weight_importance.png')
            
            # 2. 게이트별 특성 중요도 비교 시각화
            plt.figure(figsize=(12, 10))
            top_features = list(sorted_importance.keys())[:5]  # 상위 5개 특성
            
            gate_data = []
            for gate_name, gate_imp in gate_importance.items():
                for feature in top_features:
                    gate_data.append({
                        'Gate': gate_name,
                        'Feature': feature,
                        'Importance': gate_imp[feature]
                    })
            
            gate_df = pd.DataFrame(gate_data)
            pivot_df = gate_df.pivot(index='Feature', columns='Gate', values='Importance')
            
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.4f')
            plt.title('LSTM 게이트별 특성 중요도 (상위 5개 특성)')
            plt.tight_layout()
            plt.savefig('gate_importance.png')
            plt.close()
            
            # 3. 특성 그룹별 중요도 시각화
            plt.figure(figsize=(10, 6))
            plt.barh(list(sorted_group_importance.keys()), list(sorted_group_importance.values()))
            plt.title('특성 그룹별 중요도 (가중치 기반)')
            plt.xlabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('weight_group_importance.png')
            plt.close()
            
            print(f"\n가중치 기반 중요도 그래프가 'weight_importance.png'로 저장되었습니다.")
            print(f"게이트별 중요도 그래프가 'gate_importance.png'로 저장되었습니다.")
            print(f"그룹별 중요도 그래프가 'weight_group_importance.png'로 저장되었습니다.")
            
        except Exception as e:
            print(f"가중치 기반 특성 중요도 분석 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def analyze_with_shap(self, top_n=10):
        """
        SHAP 분석을 통한 특성 중요도 분석
        - 모델 출력에 대한 특성별 기여도 계산
        - 특성 그룹별 SHAP 값 분석
        """
        try:
            print("\n6. SHAP 분석을 통한 특성 중요도")
            
            # 데이터 준비
            numbers_df = self.get_data()
            X_train, y_train, _ = self.prepare_features(numbers_df)
            
            # 특성 이름 정의
            feature_names = [
                "번호1", "번호2", "번호3", "번호4", "번호5", "번호6",
                "평균", "표준편차", "최소값", "최대값",
                "짝수비율", "낮은번호비율", 
                "간격1", "간격2", "간격3", "간격4", "간격5",
                "범위", "이전회차중복", "숫자간표준편차",
                "1-15구간비율", "16-30구간비율", "31-45구간비율", 
                "연속번호비율", "번호합계", "홀짝교차패턴",
                "증감패턴", "엔트로피", "첨도", "간격균일성"
            ]
            
            # 마지막 시점의 데이터만 사용
            X_last = X_train[:, -1, :]
            
            # SHAP 분석용 모델 래퍼
            class SHAPModelWrapper:
                def __init__(self, model):
                    self.model = model
                
                def __call__(self, X):
                    # 입력이 2D인 경우 3D로 변환 (배치, 시퀀스 길이 1, 특성)
                    if len(X.shape) == 2:
                        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                    else:
                        X_reshaped = X
                    
                    # 텐서 변환
                    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
                    
                    # 모델 파라미터 장치 확인
                    device = next(self.model.parameters()).device
                    X_tensor = X_tensor.to(device)
                    
                    # 모델 평가 모드 설정
                    self.model.eval()
                    
                    with torch.no_grad():
                        y_pred = self.model(X_tensor)
                        return y_pred.cpu().numpy()
            
            # SHAP 분석
            print("SHAP 분석 중... (시간이 소요될 수 있습니다)")
            
            # SHAP 모델 래퍼 생성
            shap_model = SHAPModelWrapper(self.model)
            
            # 배경 데이터 샘플링 (100개 정도로 제한)
            background_data = shap.sample(X_last, min(100, len(X_last)))
            
            # SHAP 설명자 생성
            explainer = shap.KernelExplainer(shap_model, background_data)
            
            # 샘플 데이터 선정 (20개 정도로 제한)
            sample_indices = np.random.choice(len(X_last), min(20, len(X_last)), replace=False)
            shap_values = explainer.shap_values(X_last[sample_indices])
            
            # SHAP 값의 절대값 평균으로 특성 중요도 계산
            feature_importance = {}
            for i, feature in enumerate(feature_names):
                feature_importance[feature] = np.abs(shap_values[0][:, i]).mean()
            
            # 번호1~6 제외하고 중요도 필터링
            filtered_importance = {k: v for k, v in feature_importance.items() if not k.startswith("번호")}
            
            # 중요도 정규화
            total = sum(filtered_importance.values())
            for feature in filtered_importance:
                filtered_importance[feature] /= total
            
            # 중요도 기준으로 정렬
            sorted_importance = {k: v for k, v in sorted(filtered_importance.items(), key=lambda item: item[1], reverse=True)}
            
            # 상위 N개 특성 출력
            print("\n=== SHAP 기반 특성 중요도 순위 (번호 특성 제외) ===")
            for i, (name, score) in enumerate(list(sorted_importance.items())[:top_n], 1):
                print(f"{i}. {name}: {score:.6f}")
            
            # 특성 그룹 정의
            feature_groups = {
                "기본통계량": ["평균", "표준편차", "최소값", "최대값", "범위", "숫자간표준편차", "첨도"],
                "분포특성": ["짝수비율", "낮은번호비율", "1-15구간비율", "16-30구간비율", "31-45구간비율", "번호합계", "엔트로피"],
                "패턴특성": ["홀짝교차패턴", "증감패턴", "연속번호비율"],
                "간격정보": ["간격1", "간격2", "간격3", "간격4", "간격5", "간격균일성"],
                "시간특성": ["이전회차중복"]
            }
            
            # 그룹별 중요도 계산
            group_importance = defaultdict(float)
            for feature, score in filtered_importance.items():
                for group_name, group_features in feature_groups.items():
                    if feature in group_features:
                        group_importance[group_name] += score
                        break
            
            # 그룹 중요도 정규화
            total_group_importance = sum(group_importance.values())
            for group in group_importance:
                group_importance[group] /= total_group_importance
            
            # 그룹별 중요도 출력
            print("\n=== 특성 그룹별 SHAP 중요도 ===")
            sorted_group_importance = {k: v for k, v in sorted(group_importance.items(), key=lambda item: item[1], reverse=True)}
            for i, (group, score) in enumerate(sorted_group_importance.items(), 1):
                print(f"{i}. {group}: {score:.6f}")
            
            # 1. 개별 특성 중요도 시각화
            plt.figure(figsize=(10, 8))
            pd.Series(sorted_importance).head(top_n).plot(kind='barh')
            plt.title('SHAP 기반 특성 중요도 (번호 특성 제외)')
            plt.xlabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('shap_importance.png')
            plt.close()
            
            # 2. SHAP 요약 플롯
            plt.figure(figsize=(12, 8))
            # feature_names에서 '번호' 시작하는 항목들의 인덱스 찾기
            non_number_indices = [i for i, name in enumerate(feature_names) if not name.startswith("번호")]
            # 해당 인덱스만 사용하여 SHAP 요약 플롯 생성
            shap.summary_plot(
                [shap_values[0][:, non_number_indices]], 
                X_last[sample_indices][:, non_number_indices],
                feature_names=[feature_names[i] for i in non_number_indices],
                max_display=top_n,
                show=False
            )
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            plt.close()
            
            # 3. 그룹별 중요도 시각화
            plt.figure(figsize=(10, 6))
            plt.barh(list(sorted_group_importance.keys()), list(sorted_group_importance.values()))
            plt.title('특성 그룹별 SHAP 중요도')
            plt.xlabel('중요도 점수')
            plt.tight_layout()
            plt.savefig('shap_group_importance.png')
            plt.close()
            
            print(f"\nSHAP 중요도 그래프가 'shap_importance.png'로 저장되었습니다.")
            print(f"SHAP 요약 그래프가 'shap_summary.png'로 저장되었습니다.")
            print(f"그룹별 SHAP 중요도 그래프가 'shap_group_importance.png'로 저장되었습니다.")
            
        except Exception as e:
            print(f"SHAP 분석 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def analyze_attention_weights(self):
        """
        Self-Attention 가중치 분석 및 시각화
        - 시간 포인트 간 주목도 시각화
        - 주요 시간 포인트 식별
        - 특정 시퀀스 샘플에 대한 자세한 분석
        """
        try:
            print("\n7. Self-Attention 가중치 분석")
            
            # 모델이 없으면 모델 로드
            if self.model is None:
                try:
                    self.load_model('combination_lotto_model.pth')
                except FileNotFoundError:
                    print("저장된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
                    return
            
            # 테스트 데이터 준비
            numbers_df = self.get_data()
            X_train, _, _ = self.prepare_features(numbers_df)
            
            # 가장 최근 20개 시퀀스 선택 (시각화에 적합)
            X_recent = X_train[-20:]
            
            # 테스트 데이터 배치로 변환
            device = self.get_device()
            X_tensor = torch.FloatTensor(X_recent).to(device)
            
            # 모델 평가 모드 설정
            self.model.eval()
            
            # 어텐션 가중치 추출
            with torch.no_grad():
                _ = self.model(X_tensor)  # 예측 및 어텐션 가중치 계산
                attention_weights = self.model.get_attention_weights()  # [batch_size, seq_len, seq_len]
            
            if attention_weights is None:
                print("어텐션 가중치를 추출할 수 없습니다. 모델에 어텐션 레이어가 없을 수 있습니다.")
                return
                
            # 평균 어텐션 가중치 계산 (모든 배치에 대해)
            avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
            
            # 시간 포인트 이름 정의
            seq_len = avg_attention.shape[0]
            time_points = [f"t-{seq_len-i}" for i in range(seq_len)]
            
            # 1. 평균 어텐션 가중치 히트맵 시각화
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attention, annot=True, fmt='.2f', cmap='viridis',
                       xticklabels=time_points, yticklabels=time_points)
            plt.title('평균 Self-Attention 가중치')
            plt.xlabel('시간 포인트 (Key)')
            plt.ylabel('시간 포인트 (Query)')
            plt.tight_layout()
            plt.savefig('attention_heatmap.png')
            plt.close()
            
            # 2. 시간 포인트별 중요도 분석 (평균 어텐션 점수)
            importance_to_last = avg_attention[-1, :]  # 마지막 시점에서 각 시점을 얼마나 중요하게 보는지
            
            plt.figure(figsize=(12, 6))
            plt.bar(time_points, importance_to_last)
            plt.title('예측에 대한 각 시간 포인트의 중요도 (마지막 시점 기준)')
            plt.xlabel('시간 포인트')
            plt.ylabel('어텐션 점수')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('attention_importance.png')
            plt.close()
            
            # 3. 특정 시퀀스 샘플에 대한 어텐션 가중치 분석 (첫 번째 샘플)
            sample_attention = attention_weights[0].cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(sample_attention, annot=True, fmt='.2f', cmap='viridis',
                       xticklabels=time_points, yticklabels=time_points)
            plt.title('샘플 시퀀스에 대한 Self-Attention 가중치')
            plt.xlabel('시간 포인트 (Key)')
            plt.ylabel('시간 포인트 (Query)')
            plt.tight_layout()
            plt.savefig('sample_attention_heatmap.png')
            plt.close()
            
            # 4. 시간별 주목 패턴 분석 (특정 시간 포인트가 다른 시간 포인트를 얼마나 주목하는지)
            plt.figure(figsize=(14, 10))
            for i in range(min(5, seq_len)):  # 처음 5개 시간 포인트만 표시
                plt.subplot(5, 1, i+1)
                plt.bar(time_points, sample_attention[i])
                plt.title(f'시간 포인트 {time_points[i]}에서의 주목 패턴')
                plt.ylabel('어텐션 점수')
                plt.ylim(0, 1)
                if i < 4:  # 마지막 subplot이 아니면 x축 레이블 숨김
                    plt.xticks([])
                else:
                    plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('attention_patterns.png')
            plt.close()
            
            print(f"Self-Attention 가중치 분석 그래프가 저장되었습니다:")
            print(f"- 평균 어텐션 히트맵: 'attention_heatmap.png'")
            print(f"- 시간 포인트별 중요도: 'attention_importance.png'")
            print(f"- 샘플 어텐션 히트맵: 'sample_attention_heatmap.png'")
            print(f"- 시간별 주목 패턴: 'attention_patterns.png'")
            
            # 주요 시간 포인트 식별
            print("\n=== 마지막 예측에 가장 중요한 시간 포인트 (상위 3개) ===")
            time_importance = [(time_points[i], importance_to_last[i]) for i in range(len(time_points))]
            time_importance.sort(key=lambda x: x[1], reverse=True)
            for i, (time_point, score) in enumerate(time_importance[:3], 1):
                print(f"{i}. {time_point}: {score:.4f}")
                
            # 어텐션 중요도 저장
            self.attention_importance = time_importance
            
        except Exception as e:
            print(f"어텐션 가중치 분석 중 오류 발생: {str(e)}")
            traceback.print_exc()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model = model.state_dict().copy()
        self.val_loss_min = val_loss

def display_lotto_stats(match_stats):
    """로또 일치 통계 표시"""
    print("\n===== 로또 번호 일치 통계 =====")
    print(f"정확한 예측 (6개 모두 일치): {match_stats['exact']:.2f}%")
    print(f"3개 이상 일치 (당첨): {match_stats['three_plus']:.2f}%")
    print("\n일치 개수별 분포:")
    for i in range(7):
        print(f"  {i}개 일치: {match_stats[f'match_{i}']:.2f}%")

def generate_pdf_report(predictor, combinations):
    """예측 결과와 분석 정보를 포함한 PDF 보고서를 생성합니다."""
    try:
        # 현재 날짜와 시간으로 파일명 생성
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"lotto_prediction_report_{current_time}.pdf"
        
        # 모델 구조 다이어그램 생성
        generate_model_diagram(predictor)
        
        # PDF 객체 생성
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 분석 결과에 따른 동적 해석 생성 함수
        def generate_permutation_interpretation(perm_data):
            """순열 중요도 결과에 따른 동적 해석을 생성합니다."""
            if not perm_data:
                return "No permutation importance data available."
                
            max_importance = max(perm_data, key=lambda x: x[1])[1] if perm_data else 0
            if max_importance < 0.0001:
                return "The permutation importance analysis shows values near-zero for all features, confirming the random nature of lottery data. This indicates no single feature has consistent predictive power."
            else:
                top_features = [f[0] for f in perm_data[:3]]
                return f"The permutation importance analysis highlights {', '.join(top_features)} as having relatively higher importance with values of {max_importance:.6f}, though still limited due to the inherent randomness of lottery draws."
        
        def generate_weight_interpretation(weight_data):
            """가중치 기반 중요도 결과에 따른 동적 해석을 생성합니다."""
            if not weight_data:
                return "No weight-based importance data available."
                
            top_features = [f[0] for f in weight_data[:3]]
            top_values = [f[1] for f in weight_data[:3]]
            max_diff = max(top_values) - min(top_values)
            
            if max_diff < 0.005:  # 상위 특성들의 중요도가 유사한 경우
                return f"The weight-based analysis reveals that the model assigns similar importance to multiple features including {', '.join(top_features)}. This balanced approach suggests the model is considering various aspects of lottery data, though predictive power remains constrained by lottery randomness."
            else:
                return f"The weight-based analysis reveals that the model assigns highest importance to {', '.join(top_features)}, with {top_features[0]} being particularly influential (importance: {top_values[0]:.6f}). This suggests the model is focusing on these numerical properties when making predictions."
        
        def generate_shap_interpretation(shap_data):
            """SHAP 분석 결과에 따른 동적 해석을 생성합니다."""
            if not shap_data:
                return "No SHAP analysis data available."
                
            top_features = [f[0] for f in shap_data[:3]]
            feature_categories = categorize_features(top_features)
            
            return f"The SHAP analysis indicates that features related to {feature_categories} have the highest impact on predictions. This suggests the model is especially sensitive to {feature_categories} when making predictions. However, the relatively close SHAP values across different features reflect the inherent unpredictability of lottery outcomes."
        
        def generate_attention_interpretation(attention_data):
            """어텐션 가중치 분석 결과에 따른 동적 해석을 생성합니다."""
            if not attention_data:
                return "No attention weight analysis data available."
                
            weights = [w[1] for w in attention_data]
            max_weight = max(weights) if weights else 0
            min_weight = min(weights) if weights else 0
            diff = max_weight - min_weight
            
            if diff < 0.02:  # 가중치가 거의 균일하게 분포된 경우
                return "The attention weights analysis shows nearly uniform attention distribution across all time points. This even distribution is meaningful - it indicates that the model has correctly learned that no particular past draw has significantly more predictive power than others, reflecting the independent nature of lottery draws."
            else:
                top_timepoints = [t[0] for t in attention_data[:3]]
                return f"The attention weights analysis shows that the model assigns higher importance to more recent draws ({', '.join(top_timepoints)}). While this might suggest some recency effects in the patterns, the differences are subtle and should be interpreted cautiously given the random nature of lottery draws."
        
        def categorize_features(feature_list):
            """특성 목록을 카테고리로 분류합니다."""
            categories = []
            
            pattern_keywords = ['pattern', 'alternation', 'sequence']
            distribution_keywords = ['ratio', 'range', 'distribution', 'proportion']
            stats_keywords = ['maximum', 'minimum', 'average', 'mean', 'standard deviation', 'entropy']
            
            for feature in feature_list:
                feature_lower = feature.lower()
                
                if any(keyword in feature_lower for keyword in pattern_keywords):
                    categories.append('pattern recognition')
                elif any(keyword in feature_lower for keyword in distribution_keywords):
                    categories.append('number distribution')
                elif any(keyword in feature_lower for keyword in stats_keywords):
                    categories.append('statistical properties')
                else:
                    categories.append('diverse factors')
            
            # 중복 제거
            unique_categories = list(set(categories))
            
            if len(unique_categories) == 1:
                return unique_categories[0]
            elif len(unique_categories) == 0:
                return "various factors"
            else:
                return " and ".join(unique_categories)
        
        def generate_group_interpretation(group_data):
            """그룹별 중요도 결과에 따른 동적 해석을 생성합니다."""
            if not group_data:
                return "No group importance data available."
            
            sorted_groups = sorted(group_data, key=lambda x: x[1], reverse=True)
            top_group, top_value = sorted_groups[0]
            second_group, second_value = sorted_groups[1] if len(sorted_groups) > 1 else (None, 0)
            
            if abs(top_value - second_value) < 0.05:
                return f"The analysis shows that {top_group} and {second_group} have similar importance levels, suggesting the model balances multiple feature types in its predictions."
            else:
                return f"The analysis shows that {top_group} with importance value {top_value:.4f} has significantly higher impact on predictions compared to other feature groups."
        
        def generate_overall_interpretation(permutation_data, weight_data, shap_data, attention_data):
            """모든 분석 결과를 종합한 동적 해석을 생성합니다."""
            insights = []
            
            # 1. 랜덤성 해석
            perm_max = max([x[1] for x in permutation_data]) if permutation_data else 0
            if perm_max < 0.001:
                insights.append("Randomness Confirmation: The minimal permutation importance values confirm the random nature of lottery draws. Our model correctly identifies that lottery draws are independent events without strong predictive patterns.")
            
            # 2. 특성 관련성 해석
            all_top_features = set()
            if weight_data: all_top_features.update([x[0] for x in weight_data[:3]])
            if shap_data: all_top_features.update([x[0] for x in shap_data[:3]])
            if permutation_data and max([x[1] for x in permutation_data]) > 0.0001: 
                all_top_features.update([x[0] for x in permutation_data[:3]])
            
            if all_top_features:
                features_str = ", ".join(list(all_top_features)[:3])  # 너무 길어지지 않도록 상위 3개만
                insights.append(f"Feature Relevance: Features like {features_str} appear consistently important across analyses, suggesting they may offer limited but detectable patterns in the lottery data. However, their overall predictive power remains limited due to the inherent randomness of lottery draws.")
            
            # 3. 모델 동작 해석
            model_focus = set()
            if weight_data and len(weight_data) > 0: model_focus.add(categorize_features([weight_data[0][0]]))
            if shap_data and len(shap_data) > 0: model_focus.add(categorize_features([shap_data[0][0]]))
            
            if model_focus:
                focus_str = " and ".join(list(model_focus))
                insights.append(f"Model Behavior: The model appears to focus more on {focus_str} rather than specific numbers, which aligns with sound statistical approaches to analyzing random data.")
            else:
                insights.append("Model Behavior: The model's internal weights suggest it's attempting to balance various factors rather than focusing heavily on any specific pattern, which is appropriate for random data like lottery draws.")
            
            # 4. 예측 전략 해석
            # 어텐션 가중치의 분포를 확인
            if attention_data:
                weights = [w[1] for w in attention_data]
                max_weight = max(weights)
                min_weight = min(weights)
                diff = max_weight - min_weight
                
                if diff < 0.02:
                    insights.append("Prediction Strategy: The model's uniform attention across time points indicates it correctly treats each draw as independent, avoiding the trap of overemphasizing recent outcomes (gambler's fallacy). This approach is statistically sound for truly random events.")
                else:
                    insights.append("Prediction Strategy: The model shows some preference for more recent draws in its attention mechanism, suggesting it may be identifying short-term patterns. However, users should be cautious as such patterns in lottery data are often temporary and may not persist.")
            
            # 5. 제한사항 해석
            insights.append("Limitations: The analyses collectively highlight the fundamental limitation of lottery prediction - while patterns can be identified in historical data, their predictive value for future draws is inherently limited by the random drawing process.")
            
            return insights
        
        # 제목 및 다양한 텍스트 추가 함수 설정
        def add_title(title, size=16):
            pdf.set_font("Helvetica", "B", size)
            pdf.cell(0, 10, txt=title, ln=True, align="C")
            pdf.ln(5)
        
        def add_subtitle(title, size=12):
            pdf.set_font("Helvetica", "B", size)
            pdf.cell(0, 8, txt=title, ln=True, align="L")
            pdf.ln(2)
        
        def add_text(text, size=10):
            pdf.set_font("Helvetica", "", size)
            # 텍스트에서 ASCII가 아닌 문자 제거
            text = ''.join(char for char in text if ord(char) < 128)
            pdf.multi_cell(0, 5, txt=text)
            pdf.ln(2)
        
        def add_predictions(combinations):
            pdf.set_font("Helvetica", "B", 10)
            for i, combo in enumerate(combinations, 1):
                numbers = ", ".join([f"{num:02d}" for num in combo])
                pdf.cell(0, 7, txt=f"Combination {i}: {numbers}", ln=True)
            pdf.ln(5)
        
        def add_analysis_result(title, result_data):
            """분석 결과와 구체적인 설명을 추가하는 함수"""
            add_subtitle(title)
            
            # 테이블 형식으로 상위 항목 표시
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(85, 7, "Feature", 1, 0, "C")
            pdf.cell(30, 7, "Importance", 1, 1, "C")
            
            pdf.set_font("Helvetica", "", 9)
            for feature, value in result_data[:5]:  # 상위 5개만 표시
                pdf.cell(85, 6, feature, 1, 0)
                pdf.cell(30, 6, f"{value:.4f}", 1, 1, "C")
            
            pdf.ln(5)
        
        def add_image(image_path, w=0, h=0, caption="", caption_size=8):
            try:
                if not os.path.exists(image_path):
                    pdf.cell(0, 5, txt=f"Image not found: {image_path}", ln=True)
                    return
                
                # 이미지 크기 계산
                try:
                    img = Image.open(image_path)
                    img_w, img_h = img.size
                    
                    # 너비나 높이가 0인 경우 자동 계산
                    if w == 0 and h == 0:
                        w = 180  # 기본 너비
                        h = img_h * (w / img_w)  # 비율 유지
                    elif w == 0:
                        w = img_w * (h / img_h)  # 높이에 맞춘 너비
                    elif h == 0:
                        h = img_h * (w / img_w)  # 너비에 맞춘 높이
                except Exception as e:
                    print(f"이미지 크기 계산 중 오류: {e}")
                    w = 180
                    h = 100
                
                # 이미지가 페이지를 벗어나는지 확인
                if pdf.get_y() + h + 10 > pdf.page_break_trigger:
                    pdf.add_page()
                
                pdf.image(image_path, x=10, y=pdf.get_y(), w=w, h=h)
                pdf.ln(h + 5)
                
                # 이미지 캡션 추가
                if caption:
                    pdf.set_font("Helvetica", "I", caption_size)
                    pdf.cell(0, 5, txt=caption, ln=True, align="C")
                    pdf.ln(2)
            except Exception as e:
                print(f"이미지 추가 중 오류: {e} - {image_path}")
                pdf.cell(0, 5, txt=f"Error adding image: {e}", ln=True)
        
        # 보고서 제목
        today = datetime.now().strftime('%Y-%m-%d')
        add_title(f"Lottery Number Prediction Report - {today}")
        
        # 1. 개요
        add_subtitle("1. Overview")
        add_text("This report presents the results of our deep learning-based lottery number prediction system. The system uses historical lottery data to identify patterns and make predictions for future draws.")
        add_text("Note: This prediction is provided for informational and educational purposes only. Lottery outcomes are random, and no prediction system can guarantee results.")
        
        # 2. 예측 결과
        add_subtitle("2. Prediction Results")
        add_text(f"The following combinations are predicted for the next lottery draw (generated on {today}):")
        add_predictions(combinations)
        
        # 예측 결과에 대한 상세 설명 추가
        add_text("Analysis of Predicted Combinations:")
        
        # 번호대별 분포 계산
        all_predicted_numbers = [num for combo in combinations for num in combo]
        low_range = sum(1 for num in all_predicted_numbers if 1 <= num <= 15)
        mid_range = sum(1 for num in all_predicted_numbers if 16 <= num <= 30)
        high_range = sum(1 for num in all_predicted_numbers if 31 <= num <= 45)
        total_numbers = len(all_predicted_numbers)
        
        # 홀짝 분포 계산
        odd_numbers = sum(1 for num in all_predicted_numbers if num % 2 == 1)
        even_numbers = sum(1 for num in all_predicted_numbers if num % 2 == 0)
        
        add_text(f"- Number Range Distribution: Low (1-15): {low_range}/{total_numbers} ({low_range/total_numbers*100:.1f}%), Mid (16-30): {mid_range}/{total_numbers} ({mid_range/total_numbers*100:.1f}%), High (31-45): {high_range}/{total_numbers} ({high_range/total_numbers*100:.1f}%)")
        add_text(f"- Odd-Even Distribution: Odd numbers: {odd_numbers}/{total_numbers} ({odd_numbers/total_numbers*100:.1f}%), Even numbers: {even_numbers}/{total_numbers} ({even_numbers/total_numbers*100:.1f}%)")
        
        # 번호 간 간격 분석
        gaps = []
        for combo in combinations:
            sorted_combo = sorted(combo)
            for i in range(1, len(sorted_combo)):
                gaps.append(sorted_combo[i] - sorted_combo[i-1])
        
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        add_text(f"- Average gap between consecutive numbers: {avg_gap:.2f}")
        
        # 자주 등장하는 번호 분석
        from collections import Counter
        number_counts = Counter(all_predicted_numbers)
        most_common = number_counts.most_common(3)
        if most_common:
            common_nums = ", ".join([f"{num[0]:02d} (appears {num[1]} times)" for num in most_common])
            add_text(f"- Most frequent numbers in predictions: {common_nums}")
        
        # 조합별 특성 간략 설명
        add_text("Brief analysis of each combination:")
        for i, combo in enumerate(combinations, 1):
            sorted_combo = sorted(combo)
            ranges = max(combo) - min(combo)
            odd_count = sum(1 for num in combo if num % 2 == 1)
            low_count = sum(1 for num in combo if num <= 15)
            mid_count = sum(1 for num in combo if 16 <= num <= 30)
            high_count = sum(1 for num in combo if num >= 31)
            
            add_text(f"Combination {i}: Range: {ranges}, Odd-Even: {odd_count}-{6-odd_count}, Distribution: Low {low_count}, Mid {mid_count}, High {high_count}")
        
        # 3. 모델 구조
        add_subtitle("3. Model Architecture")
        add_text("Our prediction model uses a combination of LSTM (Long Short-Term Memory) networks with self-attention mechanism to capture temporal patterns in lottery data.")
        
        # 모델 세부 정보 추가
        model = predictor.model
        if model:
            # 모델 파라미터 추출
            try:
                input_dim = model.lstm.input_size
                hidden_size = model.lstm.hidden_size
                num_layers = 1  # 현재 모델은 단일 층 LSTM 사용
                
                add_text(f"Model Details:")
                add_text(f"- Input Features: {input_dim} features per time step")
                add_text(f"- LSTM Hidden Size: {hidden_size} units")
                add_text(f"- Number of LSTM Layers: {num_layers}")
                add_text(f"- Self-Attention: Yes (with scaling)")
                add_text(f"- Output: Dense layer projecting to 45-dimensional space (the lottery number space)")
                
                # 내부 동작 설명 추가
                add_text("Model Working Process:")
                add_text("1. The model takes in feature sequences from the past 10 draws as input")
                add_text("2. LSTM layers process the sequence to capture temporal patterns")
                add_text("3. Self-attention mechanism assigns weights to different time points")
                add_text("4. A fully connected layer maps the features to probabilities for each number (1-45)")
                add_text("5. Combinations are generated based on these probabilities with controlled randomness")
            except Exception as e:
                print(f"모델 세부 정보 추출 오류: {e}")
        
        # 모델 아키텍처 다이어그램 추가
        if os.path.exists('model_architecture.png'):
            add_image('model_architecture.png', w=180, caption="Model Architecture Diagram")
        
        # 4. 특성 중요도 분석
        add_subtitle("4. Feature Importance Analysis")
        add_text("We analyze which features contribute most to the prediction using multiple methods:")
        
        # 4.1 순열 중요도
        add_subtitle("4.1 Permutation Importance")
        add_text("This analysis shows how important each feature is by measuring the decrease in model performance when a feature is randomly shuffled.")
        
        # 순열 중요도 결과 세부 정보 추가 (실행 결과에서 추출)
        perm_importance = predictor.permutation_importance if hasattr(predictor, 'permutation_importance') else []
        
        # 표 형식으로 순열 중요도 결과 표시
        if perm_importance:
            add_analysis_result("Top Features by Permutation Importance", perm_importance)
        
        # 동적 해석 추가
        add_text(generate_permutation_interpretation(perm_importance))
        
        # 그룹별 및 시간별 중요도 해석 추가
        add_text("Group-wise Importance Analysis Result:")
        perm_group_importance = predictor.group_importance if hasattr(predictor, 'group_importance') else []
        if perm_group_importance:
            for group, importance in perm_group_importance[:3]:
                add_text(f"- {group}: {importance:.6f}")
            
            # 동적 그룹 해석 추가
            add_text(generate_group_interpretation(perm_group_importance))
        
        add_text("Time-point Importance Analysis Result:")
        time_importance = predictor.time_importance if hasattr(predictor, 'time_importance') else []
        if time_importance:
            for time_point, importance in time_importance[:3]:
                add_text(f"- {time_point}: {importance:.6f}")
            
            # 시간 중요도 동적 해석
            if max([imp[1] for imp in time_importance]) - min([imp[1] for imp in time_importance]) < 0.1:
                add_text("The time-point analysis shows relatively uniform importance across different time points, suggesting the model treats all historical draws with similar weight, aligned with the random nature of lottery drawings.")
            else:
                top_time = time_importance[0][0]
                add_text(f"The time-point analysis shows that more recent draws (especially {top_time}) have somewhat higher importance, which should be interpreted with caution given the randomness of lottery drawings.")
        
        if os.path.exists('permutation_importance.png'):
            add_image('permutation_importance.png', w=180, caption="Feature Importance based on Permutation Method")
        
        if os.path.exists('group_importance.png'):
            add_image('group_importance.png', w=180, caption="Feature Group Importance")
        
        if os.path.exists('time_importance.png'):
            add_image('time_importance.png', w=180, caption="Time Point Importance")
        
        # 4.2 가중치 기반 중요도
        add_subtitle("4.2 Weight-based Importance")
        add_text("This analysis examines the model's internal weights to determine which features it relies on most heavily.")
        
        # 가중치 기반 중요도 결과 세부 정보 추가
        weight_importance = predictor.weight_importance if hasattr(predictor, 'weight_importance') else []
        
        # 표 형식으로 가중치 중요도 결과 표시
        if weight_importance:
            add_analysis_result("Top Features by Weight-based Importance", weight_importance)
            
            # 게이트별 중요도 해석 추가
            add_text("LSTM Gate-specific Analysis:")
            if hasattr(predictor, 'input_gate_importance') and predictor.input_gate_importance:
                top_input = predictor.input_gate_importance[0]
                add_text(f"- Input Gate prioritizes: {top_input[0]} ({top_input[1]:.6f})")
            
            if hasattr(predictor, 'forget_gate_importance') and predictor.forget_gate_importance:
                top_forget = predictor.forget_gate_importance[0]
                add_text(f"- Forget Gate prioritizes: {top_forget[0]} ({top_forget[1]:.6f})")
            
            if hasattr(predictor, 'cell_gate_importance') and predictor.cell_gate_importance:
                top_cell = predictor.cell_gate_importance[0]
                add_text(f"- Cell Gate prioritizes: {top_cell[0]} ({top_cell[1]:.6f})")
            
            if hasattr(predictor, 'output_gate_importance') and predictor.output_gate_importance:
                top_output = predictor.output_gate_importance[0]
                add_text(f"- Output Gate prioritizes: {top_output[0]} ({top_output[1]:.6f})")
        
        # 가중치 기반 중요도 해석 추가 (동적)
        add_text(generate_weight_interpretation(weight_importance))
        
        # 그룹별 가중치 중요도 해석 추가
        add_text("Group-wise Weight Importance Result:")
        weight_group_importance = predictor.weight_group_importance if hasattr(predictor, 'weight_group_importance') else []
        if weight_group_importance:
            for group, importance in weight_group_importance[:3]:
                add_text(f"- {group}: {importance:.6f}")
            
            # 동적 그룹 해석 추가
            add_text(generate_group_interpretation(weight_group_importance))
        
        if os.path.exists('weight_importance.png'):
            add_image('weight_importance.png', w=180, caption="Feature Importance based on Model Weights")
        
        if os.path.exists('gate_importance.png'):
            add_image('gate_importance.png', w=180, caption="LSTM Gate-specific Feature Importance")
        
        if os.path.exists('weight_group_importance.png'):
            add_image('weight_group_importance.png', w=180, caption="Feature Group Importance based on Weights")
        
        # 4.3 SHAP 분석
        add_subtitle("4.3 SHAP Analysis")
        add_text("SHAP (SHapley Additive exPlanations) values help us understand feature importance by measuring each feature's contribution to each prediction.")
        
        # SHAP 분석 결과 세부 정보 추가
        shap_importance = predictor.shap_importance if hasattr(predictor, 'shap_importance') else []
        
        # 표 형식으로 SHAP 중요도 결과 표시
        if shap_importance:
            add_analysis_result("Top Features by SHAP Importance", shap_importance)
        
        # SHAP 그룹별 중요도 해석 추가
        add_text("Group-wise SHAP Importance Result:")
        shap_group_importance = predictor.shap_group_importance if hasattr(predictor, 'shap_group_importance') else []
        if shap_group_importance:
            for group, importance in shap_group_importance[:3]:
                add_text(f"- {group}: {importance:.6f}")
            
            # 동적 그룹 해석 추가
            add_text(generate_group_interpretation(shap_group_importance))
        
        # SHAP 분석 해석 추가 (동적)
        add_text(generate_shap_interpretation(shap_importance))
        
        if os.path.exists('shap_importance.png'):
            add_image('shap_importance.png', w=180, caption="Feature Importance based on SHAP Values")
        
        if os.path.exists('shap_summary.png'):
            add_image('shap_summary.png', w=180, caption="SHAP Summary Plot")
        
        if os.path.exists('shap_group_importance.png'):
            add_image('shap_group_importance.png', w=180, caption="Feature Group Importance based on SHAP Values")
        
        # 5. 어텐션 가중치 분석
        add_subtitle("5. Attention Weight Analysis")
        add_text("The self-attention mechanism allows the model to focus on the most relevant time points when making predictions.")
        
        # 어텐션 가중치 분석 결과 세부 정보 추가
        attention_importance = predictor.attention_importance if hasattr(predictor, 'attention_importance') else []
        if attention_importance:
            add_text("Time Point Importance by Attention Weights:")
            for time_point, importance in attention_importance[:3]:
                add_text(f"- {time_point}: {importance:.6f}")
            
            # 시간 패턴 분석 추가
            attention_patterns = getattr(predictor, 'attention_patterns', None)
            if attention_patterns and isinstance(attention_patterns, dict):
                add_text("Attention Pattern Analysis:")
                for pattern, value in list(attention_patterns.items())[:3]:
                    add_text(f"- {pattern}: {value:.4f}")
        
        # 어텐션 가중치 분석 해석 추가 (동적)
        add_text(generate_attention_interpretation(attention_importance))
        
        if os.path.exists('attention_heatmap.png'):
            add_image('attention_heatmap.png', w=180, caption="Average Attention Weights Heatmap")
        
        if os.path.exists('attention_importance.png'):
            add_image('attention_importance.png', w=180, caption="Time Point Importance based on Attention Weights")
        
        if os.path.exists('sample_attention_heatmap.png'):
            add_image('sample_attention_heatmap.png', w=180, caption="Sample Attention Weights for Last Prediction")
        
        if os.path.exists('attention_patterns.png'):
            add_image('attention_patterns.png', w=180, caption="Attention Patterns Over Time")
        
        # 6. 종합 해석 (동적으로 변경)
        add_subtitle("6. Overall Analysis Interpretation")
        add_text("The various analyses in this report reveal several interesting insights about our lottery prediction model and lottery data in general:")
        
        # 동적으로 전체 해석 생성
        overall_insights = generate_overall_interpretation(
            perm_importance, 
            weight_importance, 
            shap_importance, 
            attention_importance
        )
        
        # 종합 해석 추가
        for i, insight in enumerate(overall_insights, 1):
            pdf.ln(2)
            add_text(f"{i}. {insight}")
        
        # 현재 보고서 결과에 대한 종합 분석 추가
        add_text(f"\nSummary of Current Prediction Results:")
        
        # 번호대 선호도 분석
        range_pref = max(['low', 'mid', 'high'], key=lambda x: {'low': low_range, 'mid': mid_range, 'high': high_range}[x])
        range_count = max(low_range, mid_range, high_range)
        
        # 홀짝 균형 분석
        odd_even_balance = 'balanced' if abs(odd_numbers - even_numbers) <= 5 else 'preference for odd' if odd_numbers > even_numbers else 'preference for even'
        
        add_text(f"Based on the model's current predictions, we observe a slight preference for numbers in the {range_pref} range ({range_count}/{total_numbers} numbers) and a {odd_even_balance} distribution of odd/even numbers.")
        
        # 가장 높은 중요도를 가진 특성 종합 (동적)
        important_features = set()
        if weight_importance: 
            important_features.add(weight_importance[0][0])
        if shap_importance: 
            important_features.add(shap_importance[0][0])
        if perm_importance and len(perm_importance) > 0 and perm_importance[0][1] > 0.0001: 
            important_features.add(perm_importance[0][0])
        
        if important_features:
            features_str = ", ".join(list(important_features))
            add_text(f"The most consistently important features across all analyses were: {features_str}. These features appear to have the strongest influence on the model's predictions for this draw.")
        
        # 7. 결론
        add_subtitle("7. Conclusion")
        add_text("This report presents our lottery number predictions and the analysis behind them. While our model identifies patterns in historical data, lottery draws are fundamentally random events, and these predictions should be used for reference only.")
        add_text("Disclaimer: This model is developed for educational purposes only. The actual lottery results are determined by random draws, and no prediction system can guarantee success. Users should not rely solely on these predictions for purchasing decisions.")
        
        # PDF 저장
        pdf.output(pdf_filename)
        print(f"PDF 보고서가 '{pdf_filename}'로 저장되었습니다.")
        
    except Exception as e:
        print(f"PDF 보고서 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def generate_model_diagram(predictor):
    """모델 구조를 시각화하여 다이어그램으로 저장합니다."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 모델 구조 정보 가져오기
        model = predictor.model
        
        # 모델 구조에서 값을 추출 (접근 방법 변경)
        # LSTM 가중치에서 입력 차원 추출
        lstm_weight = next(param for name, param in model.named_parameters() if 'lstm.weight_ih_l0' in name)
        input_dim = lstm_weight.shape[1]  # LSTM 입력 차원
        
        # LSTM 은닉층 크기 추출
        hidden_size = model.lstm.hidden_size  # LSTM 은닉층 크기
        
        # 그림 그리기
        plt.figure(figsize=(12, 10))
        
        # 노드의 위치를 원형 또는 무작위로 배치하는 방식으로 변경
        # 다양한 레이어의 노드 수 설정
        n_input = min(30, input_dim)  # 모든 입력 피쳐를 표시하지 않고 일부만 표시
        n_lstm = min(24, hidden_size)
        n_attention = 10
        n_fc = 32  # 완전 연결 레이어의 일부 노드만 표시
        n_output = 20  # 45개 출력 중 일부만 표시
        
        # 색상 설정
        colors = {
            'input': 'royalblue',
            'lstm': 'forestgreen', 
            'attention': 'darkorchid',
            'fc': 'darkorange',
            'output': 'crimson'
        }
        
        # 노드 위치 계산 함수 - 원형 배치
        def get_circle_positions(n, center_x, center_y, radius, start_angle=0, end_angle=2*np.pi):
            theta = np.linspace(start_angle, end_angle, n+1)[:-1]  # 균등 간격으로 각도 설정
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            return x, y
        
        # 각 레이어의 중심 위치 - 전체적으로는 좌에서 우로 흐름
        centers = {
            'input': (0.2, 0.5),
            'lstm': (0.4, 0.5),
            'attention': (0.6, 0.5),
            'fc': (0.75, 0.5),
            'output': (0.9, 0.5)
        }
        
        # 각 레이어의 반지름
        radii = {
            'input': 0.15,
            'lstm': 0.12,
            'attention': 0.1,
            'fc': 0.13,
            'output': 0.14
        }
        
        # 노드 위치 계산
        positions = {}
        
        # 입력 레이어 - 왼쪽에 원형으로 배치
        x_input, y_input = get_circle_positions(
            n_input, 
            centers['input'][0], 
            centers['input'][1], 
            radii['input'],
            -np.pi/2, 3*np.pi/2
        )
        positions['input'] = [(x_input[i], y_input[i]) for i in range(n_input)]
        
        # LSTM 레이어 - 타원형으로 배치
        x_lstm, y_lstm = get_circle_positions(
            n_lstm, 
            centers['lstm'][0], 
            centers['lstm'][1], 
            radii['lstm']
        )
        positions['lstm'] = [(x_lstm[i], y_lstm[i]) for i in range(n_lstm)]
        
        # 어텐션 레이어 - 원형으로 배치
        x_att, y_att = get_circle_positions(
            n_attention, 
            centers['attention'][0], 
            centers['attention'][1], 
            radii['attention']
        )
        positions['attention'] = [(x_att[i], y_att[i]) for i in range(n_attention)]
        
        # 완전 연결 레이어 - 타원형으로 배치
        x_fc, y_fc = get_circle_positions(
            n_fc, 
            centers['fc'][0], 
            centers['fc'][1], 
            radii['fc']
        )
        positions['fc'] = [(x_fc[i], y_fc[i]) for i in range(n_fc)]
        
        # 출력 레이어 - 오른쪽에 원형으로 배치
        x_output, y_output = get_circle_positions(
            n_output, 
            centers['output'][0], 
            centers['output'][1], 
            radii['output'],
            -np.pi/2, 3*np.pi/2
        )
        positions['output'] = [(x_output[i], y_output[i]) for i in range(n_output)]
        
        # 노드 그리기
        for layer, pos_list in positions.items():
            x = [pos[0] for pos in pos_list]
            y = [pos[1] for pos in pos_list]
            plt.scatter(x, y, s=100, c=colors[layer], alpha=0.8, edgecolor='white', linewidth=1)
        
        # 연결 그리기 - 각 레이어 간 일부 랜덤 연결 표시
        np.random.seed(42)  # 결과 재현을 위한 시드 설정
        
        # 연결선 추가 함수
        def add_connections(source_layer, target_layer, connection_density=0.2):
            source_positions = positions[source_layer]
            target_positions = positions[target_layer]
            
            # 랜덤 연결 생성
            n_connections = int(len(source_positions) * len(target_positions) * connection_density)
            
            for _ in range(n_connections):
                source_idx = np.random.randint(0, len(source_positions))
                target_idx = np.random.randint(0, len(target_positions))
                
                # 두 레이어 사이의 색상 중간값으로 연결선 색상 설정
                source_color = np.array(plt.cm.colors.to_rgb(colors[source_layer]))
                target_color = np.array(plt.cm.colors.to_rgb(colors[target_layer]))
                line_color = (source_color + target_color) / 2
                
                # 연결선 그리기
                source_pos = source_positions[source_idx]
                target_pos = target_positions[target_idx]
                plt.plot(
                    [source_pos[0], target_pos[0]], 
                    [source_pos[1], target_pos[1]], 
                    alpha=0.3, 
                    c=line_color, 
                    linewidth=0.8,
                    zorder=1  # 노드 아래에 선 그리기
                )
        
        # 각 레이어 간 연결 추가
        add_connections('input', 'lstm', 0.15)
        add_connections('lstm', 'attention', 0.2)
        add_connections('attention', 'fc', 0.15)
        add_connections('fc', 'output', 0.1)
        
        # 레이어 레이블 추가
        layer_labels = {
            'input': f'Input Layer\n({input_dim} features)',
            'lstm': f'LSTM Layer\n({hidden_size} units)',
            'attention': 'Self-Attention\nLayer',
            'fc': 'Fully Connected\nLayer',
            'output': 'Output Layer\n(45 numbers)'
        }
        
        for layer, center in centers.items():
            plt.text(
                center[0], 
                center[1] - radii[layer] - 0.05, 
                layer_labels[layer], 
                ha='center', 
                va='center', 
                fontsize=11, 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
            )
        
        # 그래프 스타일 설정
        plt.title('Lotto Prediction Neural Network Architecture', fontsize=16, pad=20)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # 저장
        plt.tight_layout()
        plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"모델 다이어그램 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    메인 함수 - 로또 AI 예측 시스템 실행
    """
    print("로또 AI 예측 시스템을 시작합니다 (PyTorch 개선버전)...")
    
    # 로또 AI 예측기 인스턴스 생성
    predictor = LottoAIPredictor()
    
    # 1. 모델 학습 (시계열 교차 검증 사용)
    print("\n1. AI 모델 학습 (시계열 교차 검증 사용)")
    predictor.train_with_timeseries_cv()
    
    # 2. 다음 회차 예측
    print("\n2. 다음 회차 예측")
    combinations = predictor.predict_next_numbers(num_combinations=5)
    
    # 예측 결과 출력
    print("\n다음 회차 예상 번호 조합:")
    for i, numbers in enumerate(combinations, 1):
        formatted_numbers = ", ".join(f"{num:02d}" for num in numbers)
        print(f"조합 {i}: {formatted_numbers}")
    
    # 3. 특성 중요도 분석 (순열 중요도)
    print("\n3. 특성 중요도 분석 (순열 중요도)")
    predictor.analyze_feature_importance()
    
    # 4. 모델 가중치 기반 특성 중요도 분석
    print("\n4. 모델 가중치 기반 특성 중요도 분석")
    predictor.analyze_weight_importance()
    
    # 5. SHAP 분석을 통한 특성 중요도
    print("\n5. SHAP 분석을 통한 특성 중요도")
    predictor.analyze_with_shap()
    
    # 6. 어텐션 가중치 분석
    print("\n6. 어텐션 가중치 분석")
    predictor.analyze_attention_weights()
    
    # 7. PDF 보고서 생성
    print("\n7. PDF 보고서 생성")
    generate_pdf_report(predictor, combinations)
    
    print("\n로또 AI 예측 시스템이 종료되었습니다.")
    print("※ 주의: 이 예측 결과는, 참고용으로만 사용하세요.")
    print("※ 실제 로또 당첨은 완전한 무작위 추첨을 통해 결정되며, AI 모델은 보장된 예측을 할 수 없습니다.")
    print("※ 이 모델은 학습 및 교육 목적으로 개발되었으며, 사용자는 이 정보에만 의존해 구매 결정을 내려서는 안 됩니다.")
    
if __name__ == "__main__":
    main()