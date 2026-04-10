# TimeFlow — 시계열 예측 웹 애플리케이션
# TimeFlow — Time Series Forecasting Web Application

> CSV 파일 하나로 ETS, ARIMA, XGBoost, RandomForest 앙상블 예측을 브라우저에서 바로 실행
> Upload a CSV and run ensemble forecasting (ETS, ARIMA, XGBoost, RandomForest) directly in your browser.

---

## 소개 | Overview

**TimeFlow**는 시계열 데이터의 전처리부터 앙상블 예측, 진단까지 자동화한 웹 기반 예측 플랫폼입니다.  
반도체·자동차 도메인의 산업 데이터를 중심으로 설계되었으며, 비전문가도 CSV 파일 업로드만으로 다중 모델 예측과 신뢰구간, 백테스트 결과를 시각화할 수 있습니다.

**TimeFlow** is a web-based time series forecasting platform that automates the full pipeline — from preprocessing to ensemble forecasting and model diagnostics. Designed for semiconductor and automotive domain data, it enables anyone to visualize multi-model forecasts, confidence intervals, and backtest results with a simple CSV upload.

---

## 주요 기능 | Features

- **다중 모델 앙상블** — ETS, ARIMA, XGBoost, RandomForest 자동 학습 및 가중 앙상블  
  *Multi-model ensemble: automatic training and weighted combination*
- **STL 분해** — 추세·계절성·잔차 시각화  
  *STL decomposition: trend, seasonality, and residual analysis*
- **백테스트 & ACF 진단** — 롤링 윈도우 검증 및 자기상관 분석  
  *Backtest & ACF diagnostics: rolling-window validation and autocorrelation analysis*
- **신뢰구간 예측** — 설정 가능한 CI (80%/90%/95%)  
  *Confidence interval forecasting with configurable CI levels*
- **인터랙티브 대시보드** — Chart.js 기반 실시간 시각화  
  *Interactive dashboard with real-time Chart.js visualization*

---

## 기술 스택 | Tech Stack

| 구분 | 사용 기술 |
|------|-----------|
| Backend | FastAPI, Python 3.10+ |
| ML Models | statsmodels (ETS/ARIMA), XGBoost, scikit-learn |
| Frontend | HTML/CSS/JS, Chart.js |
| Deploy | Render.com |

---

## 실행 방법 | Getting Started

```bash
pip install -r requirements.txt
uvicorn main_redesigned:app --reload --port 8002
```
접속: http://localhost:8002

---

## 배포 링크 | Live Demo

[https://timeflow.onrender.com](https://timeflow.onrender.com)
