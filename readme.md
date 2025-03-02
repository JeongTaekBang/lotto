# 로또 AI 예측 시스템 (Lottery AI Prediction System) 🎯

AI 기반 로또 번호 예측 및 분석 시스템입니다. 시계열 데이터 분석과 딥러닝을 활용하여 로또 번호를 예측하고, 다양한 특성 중요도 분석 기법을 통해 결과를 시각화합니다.

> **⚠️ 주의사항**: 이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 실제 로또 결과는 무작위 추첨으로 결정되며, 어떤 예측 시스템도 당첨을 보장할 수 없습니다. 이 모델의 예측 결과만을 기반으로 구매 결정을 내리지 마세요.

## 🌟 주요 기능

- **시계열 기반 딥러닝 모델**: LSTM과 Attention 메커니즘을 활용한 고급 시계열 예측 수행
- **교차 검증**: 시계열 특성을 고려한 5-fold 교차 검증 구현
- **다양한 특성 중요도 분석**:
  - 순열 중요도 (Permutation Importance)
  - 모델 가중치 기반 중요도 분석
  - SHAP (SHapley Additive exPlanations) 분석
  - 어텐션 가중치 분석
- **동적 해석 기능**: 분석 결과에 대한 자동화된 해석 제공
- **PDF 보고서 생성**: 예측 결과와 분석 내용을 포함한 종합 보고서 자동 생성

## 📊 시각화 결과

본 시스템은 다양한 시각화 자료를 생성합니다:
- 특성 중요도 그래프
- 그룹별 중요도 분석
- SHAP 요약 그래프
- 어텐션 히트맵 및 패턴 분석
- 모델 아키텍처 다이어그램

## 🔍 예측 결과 예시

시스템은 다음과 같은 예측 결과를 제공합니다:
- 5개의 번호 조합 생성
- 각 조합에 대한 통계적 분석 (번호 분포, 홀짝 비율 등)
- 예측에 영향을 미치는 주요 요인 분석

## 🛠️ 기술 스택

- **Python 3.10+**
- **PyTorch**: 딥러닝 모델 구현
- **Pandas & NumPy**: 데이터 처리 및 분석
- **Matplotlib & Seaborn**: 데이터 시각화
- **SHAP**: 설명 가능한 AI 구현
- **scikit-learn**: 머신러닝 유틸리티
- **FPDF**: PDF 보고서 생성

## ⚙️ 설치 방법

1. 저장소 클론하기:
```bash
git clone https://github.com/yourusername/lotto-ai-prediction.git
cd lotto-ai-prediction
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요 라이브러리 설치:
```bash
pip install -r requirements.txt
```

4. 환경 설정:
   - `.env.example` 파일을 `.env`로 복사합니다:
   ```bash
   cp .env.example .env
   ```
   - 텍스트 에디터를 사용하여 `.env` 파일을 열고 데이터베이스 정보를 입력합니다:
   ```
   DB_HOST=your_database_host
   DB_PORT=3306
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_NAME=lotto
   ```
   
   > **⚠️ 주의**: `.env` 파일은 민감한 정보를 포함하고 있으므로 버전 관리 시스템에 추가하지 마세요. `.gitignore` 파일에 `.env`가 이미 포함되어 있습니다.

5. 데이터베이스 스키마 설정:
   - MySQL 또는 MariaDB에 접속하여 다음 SQL 명령으로 `lotto` 테이블을 생성합니다:
   ```sql
   CREATE DATABASE IF NOT EXISTS lotto;
   USE lotto;
   
   CREATE TABLE IF NOT EXISTS lotto (
     id INT AUTO_INCREMENT PRIMARY KEY,
     count INT NOT NULL,
     `1` INT NOT NULL,
     `2` INT NOT NULL,
     `3` INT NOT NULL,
     `4` INT NOT NULL,
     `5` INT NOT NULL,
     `6` INT NOT NULL,
     `7` INT NOT NULL,
     person INT NOT NULL,
     amount VARCHAR(255) NOT NULL,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

## 🚀 사용 방법

1. 데이터베이스 설정 및 데이터 수집:
```bash
python lotto_crawling.py
```
이 스크립트는 다음 작업을 수행합니다:
   - 한국 로또 공식 웹사이트에서 최신 당첨 번호 데이터를 크롤링합니다
   - 크롤링한 데이터를 설정한 데이터베이스에 저장합니다
   - 추가로 기본적인 통계 분석 그래프를 생성합니다

2. 모델 실행 및 예측:
```bash
python lotto_predict.py
```

3. 생성된 PDF 보고서 확인:
모델 실행 후 자동으로 생성되는 `lotto_prediction_report_[TIMESTAMP].pdf` 파일 확인

## 📁 프로젝트 구조

- `lotto_predict.py`: 메인 예측 모델 및 분석 시스템
- `lotto_crawling.py`: 로또 데이터 수집 및 데이터베이스 구축
- `combination_lotto_model.pth`: 학습된 모델 파일
- `.env.example`: 환경 변수 설정 예제 파일
- `requirements.txt`: 필요한 Python 패키지 목록
- 다양한 분석 결과 이미지 파일들

## 📖 모델 아키텍처

본 시스템은 다음과 같은 구성요소로 이루어져 있습니다:
- **데이터 전처리**: 로또 당첨 번호 데이터를 시계열 형태로 가공
- **특성 공학**: 30개 이상의 통계적 특성 추출
- **LSTM 네트워크**: 시퀀스 패턴 학습
- **Self-Attention 메커니즘**: 중요 시간 포인트 식별
- **특성 중요도 분석**: 다양한 해석 방법론 적용

## 🤝 기여 방법

1. 이 저장소를 포크합니다
2. 새 브랜치를 생성합니다: `git checkout -b feature/amazing-feature`
3. 변경사항을 커밋합니다: `git commit -m 'Add amazing feature'`
4. 브랜치에 푸시합니다: `git push origin feature/amazing-feature`
5. Pull Request를 제출합니다

## 📄 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📧 연락처

프로젝트 관리자 - bjt0709@gmail.com

프로젝트 링크: https://github.com/JeongTaekBang/lotto
---

⭐ 이 프로젝트가 유용하다고 생각되면 별표를 눌러주세요!