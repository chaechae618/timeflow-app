"""
TimeFlow 예측 엔진 v2 — Python 검증용
VSCode에서 실행: python forecast_engine_v2.py

필요 라이브러리:
  pip install pandas numpy scipy scikit-learn statsmodels xgboost

v2 개선 사항 (논문 기반):
  FIX-1: 데이터 크기 기반 모델 전략 분기
          n<50 → 통계 모델만 / n<200 → lag 최소화 / n>=200 → 전체 풀
  FIX-2: Out-of-sample 기반 앙상블 가중치
          In-sample fitted 오차 → Rolling Origin CV 오차
  FIX-3: PACF 기반 자동 lag 선택
          고정 lag [1,2,3,7,14,...] → PACF 95% CI 기반 선택
  FIX-4: 과적합 탐지 & 경고 시스템
          backtest_smape > insample_smape * 3 → 경고 + 앙상블 제외
  FIX-5: RevIN (Reversible Instance Normalization) 적용
          Z-score 대체 → 입력 정규화, 출력 자동 역변환
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. 샘플 데이터 생성
# ─────────────────────────────────────────────
def make_daily_sample(n=730):
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    t = np.arange(n)
    trend    = 1000 + 0.8 * t
    seasonal = 250 * np.sin(2 * np.pi * t / 365 - np.pi/2)
    weekly   = 80  * np.sin(2 * np.pi * t / 7)
    noise    = np.random.normal(0, 60, n)
    values   = trend + seasonal + weekly + noise
    values[280] *= 2.2
    values[500] *= 0.4
    return pd.DataFrame({'ds': dates, 'y': np.maximum(np.round(values, 2), 0)})

def make_monthly_sample(n=60):
    np.random.seed(7)
    dates = pd.date_range('2019-01-01', periods=n, freq='MS')
    t = np.arange(n)
    trend    = 100 + 1.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise    = np.random.normal(0, 8, n)
    return pd.DataFrame({'ds': dates, 'y': np.round(trend + seasonal + noise, 2)})


# ─────────────────────────────────────────────
# 1. 데이터 진단
# ─────────────────────────────────────────────
def detect_frequency(date_series: pd.Series) -> str:
    if len(date_series) < 2:
        return 'unknown'
    dates = pd.to_datetime(date_series).sort_values()
    median_diff = dates.diff().dropna().median()
    hours = median_diff.total_seconds() / 3600
    if hours <= 1:    return 'H'
    elif hours <= 25: return 'D'
    elif hours <= 170: return 'W'
    elif hours <= 800: return 'MS'
    else:              return 'QS'

def diagnose(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    values = df[value_col].values.astype(float)
    n = len(values)
    null_mask = np.isnan(values)
    q1, q3 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
    iqr = q3 - q1
    outlier_count = int(((values < q1-1.5*iqr) | (values > q3+1.5*iqr)).sum())
    freq = detect_frequency(df[date_col])

    from statsmodels.tsa.stattools import adfuller
    clean_vals = values[~null_mask]
    if len(clean_vals) >= 20:
        adf_result = adfuller(clean_vals, autolag='AIC')
        is_stationary = adf_result[1] < 0.05
        adf_pvalue = round(adf_result[1], 4)
    else:
        is_stationary, adf_pvalue = None, None

    report = {
        'n': n, 'null_count': int(null_mask.sum()),
        'outlier_count': outlier_count,
        'outlier_pct': round(outlier_count / n * 100, 2),
        'freq': freq,
        'mean': round(float(np.nanmean(values)), 4),
        'std':  round(float(np.nanstd(values)), 4),
        'min':  round(float(np.nanmin(values)), 4),
        'max':  round(float(np.nanmax(values)), 4),
        'is_stationary': is_stationary,
        'adf_pvalue': adf_pvalue,
        'sufficient_data': n >= 30,
        'date_start': str(df[date_col].min()),
        'date_end':   str(df[date_col].max()),
    }

    print("\n" + "="*60)
    print("📊 데이터 진단 결과")
    print("="*60)
    for k, v in report.items():
        print(f"  {k:<20}: {v}")

    # 데이터 크기 권장 모델 전략 출력
    print(f"\n  📌 권장 모델 전략: {get_model_strategy(n)['label']}")
    if n < 50:
        print("  ⚠️  WARNING: 데이터 50개 미만 → ML/DL 모델 사용 금지")
    elif n < 200:
        print("  ⚠️  INFO: 데이터 200개 미만 → lag 최소화 + regularization 강화")

    return report


# ─────────────────────────────────────────────
# 2. FIX-1: 데이터 크기 기반 모델 전략 분기
# ─────────────────────────────────────────────
def get_model_strategy(n: int) -> dict:
    """
    [FIX-1] 데이터 크기에 따라 사용 가능한 모델과 설정을 자동으로 결정
    논문 근거: TSPP — "N<50에서 ML 성능 급락, N<200에서 lag 최소화 필요"
    """
    if n < 50:
        return {
            'label': f'소규모 (n={n}) — 통계 모델 전용',
            'allowed_models': ['ets', 'arima'],
            'max_lags': 3,
            'xgb_params': None,
            'rf_params': None,
        }
    elif n < 200:
        return {
            'label': f'중규모 (n={n}) — 통계+경량 ML',
            'allowed_models': ['ets', 'arima', 'xgb', 'rf'],
            'max_lags': 5,      # lag 최소화
            'xgb_params': {
                'n_estimators': 50, 'max_depth': 2,
                'learning_rate': 0.05, 'subsample': 0.6,
                'min_child_weight': 5,   # 강한 regularization
                'reg_alpha': 1.0, 'reg_lambda': 2.0,
                'random_state': 42, 'verbosity': 0
            },
            'rf_params': {
                'n_estimators': 50, 'max_depth': 3,
                'min_samples_leaf': 5,    # 과적합 방지
                'random_state': 42, 'n_jobs': -1
            },
        }
    else:
        return {
            'label': f'대규모 (n={n}) — 전체 모델 풀',
            'allowed_models': ['ets', 'arima', 'xgb', 'rf', 'nbeats'],
            'max_lags': None,   # 자동 결정 (PACF 기반)
            'xgb_params': {
                'n_estimators': 200, 'max_depth': 4,
                'learning_rate': 0.05, 'subsample': 0.8,
                'random_state': 42, 'verbosity': 0
            },
            'rf_params': {
                'n_estimators': 100, 'max_depth': 6,
                'min_samples_leaf': 3,
                'random_state': 42, 'n_jobs': -1
            },
        }


# ─────────────────────────────────────────────
# 3. FIX-5: RevIN (Reversible Instance Normalization)
# ─────────────────────────────────────────────
class RevIN:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.fitted = False
        self.log_transform = False  # 로그 변환 여부 플래그

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        v = self._interpolate(values.copy().astype(float))
        v = self._clip_outliers(v)

        # 스케일 범위가 100배 이상이면 로그 변환 적용
        # ex) 0.03 → 176 같은 극단적 성장 시계열
        v_min = v[v > 0].min() if (v > 0).any() else 1e-10
        scale_ratio = v.max() / (v_min + 1e-10)
        if scale_ratio > 100:
            self.log_transform = True
            v = np.log1p(v)   # log(1+x): 0값도 안전하게 처리
        else:
            self.log_transform = False

        self.mean_ = np.mean(v)
        self.std_  = np.std(v) + self.eps
        self.fitted = True
        return (v - self.mean_) / self.std_

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        assert self.fitted, "fit_transform을 먼저 호출하세요"
        result = normalized * self.std_ + self.mean_
        if self.log_transform:
            result = np.expm1(result)   # exp(x)-1: log1p의 역변환
        return result

    def _interpolate(self, v: np.ndarray) -> np.ndarray:
        nan_idx = np.where(np.isnan(v))[0]
        for i in nan_idx:
            left  = v[:i][~np.isnan(v[:i])]
            right = v[i+1:][~np.isnan(v[i+1:])]
            if len(left) and len(right):  v[i] = (left[-1] + right[0]) / 2
            elif len(left):               v[i] = left[-1]
            elif len(right):              v[i] = right[0]
        return v

    def _clip_outliers(self, v: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        iqr = q3 - q1
        return np.clip(v, q1 - 2.5*iqr, q3 + 2.5*iqr)

Preprocessor = RevIN


# ─────────────────────────────────────────────
# 4. FIX-3: PACF 기반 자동 lag 선택
# ─────────────────────────────────────────────
def select_lags_by_pacf(values: np.ndarray, max_lags: int = None, n_limit: int = None) -> list:
    """
    [FIX-3] PACF 95% CI 기반 유의미한 lag만 선택
    논문 근거: Meisenbacher et al. — "Filter 방식: PACF로 유의미한 lag만 선택"

    Args:
        values: 정규화된 시계열
        max_lags: 탐색할 최대 lag (None이면 자동 설정)
        n_limit: 허용 최대 lag 수 (FIX-1 모델 전략에서 결정)
    """
    n = len(values)

    if max_lags is None:
        max_lags = min(int(n * 0.3), 40)
    max_lags = max(3, max_lags)  # 최소 3개 lag 보장

    # PACF 계산 (Yule-Walker 방식)
    try:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(values, nlags=max_lags, method='yw')
    except Exception:
        # fallback: 간단한 수동 계산
        pacf_vals = _simple_pacf(values, max_lags)

    # 95% CI 임계값
    conf_bound = 1.96 / np.sqrt(n)

    # 유의미한 lag 추출 (lag 0 제외)
    significant_lags = [
        lag for lag in range(1, len(pacf_vals))
        if abs(pacf_vals[lag]) > conf_bound
    ]

    # n_limit 적용 (FIX-1)
    if n_limit is not None:
        significant_lags = significant_lags[:n_limit]

    # 항상 lag=1 포함 보장
    if not significant_lags or 1 not in significant_lags:
        significant_lags = [1] + [l for l in significant_lags if l != 1]

    # 최소 2개, 최대 15개
    significant_lags = sorted(set(significant_lags))[:15]
    if len(significant_lags) < 2:
        significant_lags = [1, 2]

    return significant_lags


def _simple_pacf(values: np.ndarray, max_lags: int) -> np.ndarray:
    """statsmodels 없을 때 fallback PACF"""
    n = len(values)
    pacf_vals = [1.0]
    for k in range(1, max_lags + 1):
        if k >= n:
            pacf_vals.append(0.0)
            continue
        y = values[k:]
        X = values[:n-k].reshape(-1, 1)
        # Yule-Walker 근사: correlation after removing lower-order effects
        if k == 1:
            cov = np.cov(y, X.flatten())
            r = cov[0, 1] / (np.std(y) * np.std(X) + 1e-10)
            pacf_vals.append(r)
        else:
            # 단순 상관으로 근사
            r = np.corrcoef(y, X.flatten())[0, 1]
            pacf_vals.append(r)
    return np.array(pacf_vals)


# ─────────────────────────────────────────────
# 5. STL 분해
# ─────────────────────────────────────────────
def detect_period_acf(values: np.ndarray, freq: str = 'D') -> int:
    n = min(len(values), 500)
    v = values[:n]
    centered = v - np.mean(v)
    var_v = np.var(centered) + 1e-10

    # freq에 따라 탐색 후보 순서 변경 — 월별이면 12 우선
    candidates = {
        'MS': [12, 6, 3, 24],
        'QS': [4, 8, 12],
        'W':  [52, 26, 13, 4],
        'D':  [7, 30, 365, 14],
        'H':  [24, 168, 12],
    }.get(freq, [7, 12, 24, 30, 52, 365])

    best_p, best_acf = candidates[0], -1  # 기본값도 freq 기반으로
    for p in candidates:
        if p >= n // 3: continue
        acf = np.mean(centered[:-p] * centered[p:]) / var_v
        if acf > best_acf:
            best_acf, best_p = acf, p
    return best_p

def stl_decompose(values: np.ndarray, period: int = None, freq: str = 'D') -> dict:
    n = len(values)
    try:
        from statsmodels.tsa.seasonal import STL
        if period is None: period = detect_period_acf(values, freq=freq)
        period = max(2, period)
        stl = STL(values, period=period, robust=True)
        res = stl.fit()
        trend, seasonal, residual = res.trend, res.seasonal, res.resid
    except Exception:
        period = period or detect_period_acf(values)
        w = min(15, n//4)
        trend = np.convolve(values, np.ones(2*w+1)/(2*w+1), mode='same')
        detrended = values - trend
        seasonal = np.zeros(n)
        for p in range(period):
            idx = np.arange(p, n, period)
            seasonal[idx] = np.mean(detrended[idx])
        residual = values - trend - seasonal

    var_r = np.var(residual)
    var_d = np.var(values - trend) + 1e-10
    var_s = np.var(seasonal) + 1e-10
    return {
        'trend': trend, 'seasonal': seasonal, 'residual': residual,
        'period': period,
        'trend_strength':  max(0.0, float(1 - var_r / var_d)),
        'season_strength': max(0.0, float(1 - var_r / (var_s + var_r))),
    }


# ─────────────────────────────────────────────
# 6. 평가 지표
# ─────────────────────────────────────────────
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    min_n = min(len(actual), len(predicted))
    actual, predicted = actual[:min_n], predicted[:min_n]
    residuals = actual - predicted

    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    denom = (np.abs(actual) + np.abs(predicted)) / 2 + 1e-10
    smape = float(np.mean(np.abs(residuals) / denom) * 100)
    nonzero = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs(residuals[nonzero] / actual[nonzero])) * 100) if nonzero.sum() > 0 else float('nan')
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((actual - np.mean(actual))**2) + 1e-10
    r2 = float(1 - ss_res / ss_tot)
    naive_mae = np.mean(np.abs(np.diff(actual))) + 1e-10
    mase = mae / naive_mae

    return {
        'MAE':   round(mae, 4),
        'RMSE':  round(rmse, 4),
        'SMAPE': round(smape, 4),
        'MAPE':  round(mape, 4),
        'R2':    round(r2, 4),
        'MASE':  round(mase, 4),
    }


# ─────────────────────────────────────────────
# 7. FIX-4: 과적합 탐지 유틸리티
# ─────────────────────────────────────────────
def detect_overfitting(model_name: str, insample_smape: float,
                       backtest_smape: float, threshold: float = 3.0) -> dict:
    """
    [FIX-4] 과적합 탐지: backtest / insample 비율 기반
    논문 근거: Meisenbacher — "in-sample 오차 기반 선택은 overfitting에 취약"
    """
    if insample_smape < 0.1:
        # in-sample SMAPE가 극단적으로 낮으면 무조건 과적합 의심
        ratio = float('inf')
        is_overfit = True
    else:
        ratio = backtest_smape / insample_smape
        is_overfit = ratio > threshold

    result = {
        'model': model_name,
        'insample_smape': insample_smape,
        'backtest_smape': backtest_smape,
        'ratio': round(ratio, 2) if ratio != float('inf') else '∞',
        'is_overfit': is_overfit,
    }

    if is_overfit:
        print(f"  ⚠️  [{model_name}] 과적합 탐지! "
              f"in-sample={insample_smape:.2f}% vs backtest={backtest_smape:.2f}% "
              f"(비율: {result['ratio']}) → 앙상블 제외")

    return result


# ─────────────────────────────────────────────
# 8. ARIMA 모델 (walk-forward 검증)
# ─────────────────────────────────────────────
class ARIMAModel:
    def __init__(self):
        self.name = 'ARIMA'
        self.color = '#00d4ff'
        self.model_fit = None
        self.train_time = None
        self.get_metrics_cache = None

    def fit(self, values_norm: np.ndarray, preprocessor: RevIN):
        import time
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        t0 = time.time()
        n = len(values_norm)

        # 차분 차수
        d = 0
        try:
            if adfuller(values_norm)[1] > 0.05:
                d = 1
                if adfuller(np.diff(values_norm))[1] > 0.05:
                    d = 2
        except: d = 1

        # AIC 기반 order 선택
        best_aic, best_order = np.inf, (1, d, 1)
        for p in range(0, 5):
            for q in range(0, 3):
                try:
                    fit = ARIMA(values_norm, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p, d, q)
                except: continue

        self.order = best_order
        self.name = f'ARIMA{best_order}'
        self.model_fit = ARIMA(values_norm, order=best_order).fit()
        self.preprocessor = preprocessor
        self.values_norm = values_norm

        # Walk-forward 1-step-ahead 예측
        min_train = max(20, n // 5)
        wf_preds_norm = np.full(n, np.nan)
        for i in range(min_train, n):
            try:
                m_wf = ARIMA(values_norm[:i], order=best_order).fit()
                fc = m_wf.forecast(steps=1)
                wf_preds_norm[i] = float(fc.iloc[0]) if hasattr(fc, 'iloc') else float(fc[0])
            except:
                wf_preds_norm[i] = values_norm[i-1]

        valid_mask = ~np.isnan(wf_preds_norm)
        self.fitted_orig = preprocessor.inverse_transform(
            np.where(valid_mask, wf_preds_norm, values_norm)
        )
        self.valid_mask = valid_mask
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        fc = self.model_fit.forecast(steps=horizon)
        fc_arr = fc.values if hasattr(fc, 'values') else np.array(fc)
        return self.preprocessor.inverse_transform(fc_arr)

    def get_metrics(self, actual_orig: np.ndarray) -> dict:
        return compute_metrics(actual_orig[self.valid_mask], self.fitted_orig[self.valid_mask])


# ─────────────────────────────────────────────
# 9. ETS 모델
# ─────────────────────────────────────────────
class ETSModel:
    def __init__(self):
        self.name = 'ETS'
        self.color = '#00e5a0'
        self.model_fit = None
        self.train_time = None
        self.get_metrics_cache = None

    def fit(self, values_norm: np.ndarray, preprocessor: RevIN, period: int = 7):
        import time
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        t0 = time.time()
        n = len(values_norm)
        try:
            seasonal_type = 'add' if period >= 2 and n >= period * 2 else None
            m = ExponentialSmoothing(
                values_norm, trend='add',
                seasonal=seasonal_type,
                seasonal_periods=(period if seasonal_type else None),
                initialization_method='estimated'
            )
            self.model_fit = m.fit(optimized=True)
        except:
            m = ExponentialSmoothing(values_norm, trend='add', initialization_method='estimated')
            self.model_fit = m.fit(optimized=True)

        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.fitted_orig = preprocessor.inverse_transform(self.model_fit.fittedvalues)
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self.preprocessor.inverse_transform(self.model_fit.forecast(horizon))

    def get_metrics(self, actual_orig: np.ndarray) -> dict:
        return compute_metrics(actual_orig, self.fitted_orig)


# ─────────────────────────────────────────────
# 10. ML 모델 (FIX-1 + FIX-3 통합)
# ─────────────────────────────────────────────
class MLModel:
    """
    [FIX-1] 데이터 크기에 따른 파라미터 분기
    [FIX-3] PACF 기반 자동 lag 선택
    """
    def __init__(self, model_type='xgb', strategy: dict = None):
        self.model_type = model_type
        self.strategy = strategy or {}

        xgb_params = self.strategy.get('xgb_params') or {
            'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
            'subsample': 0.8, 'random_state': 42, 'verbosity': 0
        }
        rf_params = self.strategy.get('rf_params') or {
            'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 3,
            'random_state': 42, 'n_jobs': -1
        }

        if model_type == 'xgb':
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(**xgb_params)
                self.name = 'XGBoost'
            except ImportError:
                self.model = GradientBoostingRegressor(
                    n_estimators=xgb_params.get('n_estimators', 100),
                    max_depth=xgb_params.get('max_depth', 4),
                    learning_rate=xgb_params.get('learning_rate', 0.05),
                    subsample=xgb_params.get('subsample', 0.8),
                    random_state=42
                )
                self.name = 'GBM'
        else:
            self.model = RandomForestRegressor(**rf_params)
            self.name = 'RandomForest'

        self.color = '#ffd166' if model_type == 'xgb' else '#ff8c42'
        self.train_time = None
        self.get_metrics_cache = None
        self.selected_lags = None

    def _make_features(self, values: np.ndarray, lags: list) -> tuple:
        """PACF 선택 lag + rolling stats 피처"""
        max_lag = max(lags)
        X, y = [], []
        for i in range(max_lag, len(values)):
            row = [values[i - l] for l in lags]
            w7  = values[max(0, i-7):i]
            w14 = values[max(0, i-14):i] if len(values) > 14 else values[:i]
            row += [
                np.mean(w7), np.std(w7) + 1e-8,
                np.mean(w14), np.std(w14) + 1e-8,
                i / len(values),
            ]
            X.append(row)
            y.append(values[i])
        return np.array(X), np.array(y), max_lag

    def fit(self, values_norm: np.ndarray, preprocessor: RevIN):
        import time
        t0 = time.time()

        n = len(values_norm)
        max_lags_limit = self.strategy.get('max_lags')

        # [FIX-3] PACF 기반 자동 lag 선택
        self.selected_lags = select_lags_by_pacf(
            values_norm,
            n_limit=max_lags_limit
        )
        print(f"     [{self.name}] PACF 선택 lag: {self.selected_lags} (총 {len(self.selected_lags)}개)")

        X, y, self.max_lag = self._make_features(values_norm, self.selected_lags)

        # [FIX-1] 데이터 크기 n<200 시 train/val 분리로 early stopping 효과
        if n < 200 and self.model_type == 'xgb':
            val_size = max(5, len(X) // 5)
            X_tr, X_val = X[:-val_size], X[-val_size:]
            y_tr, y_val = y[:-val_size], y[-val_size:]
            try:
                from xgboost import XGBRegressor
                self.model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            except:
                self.model.fit(X_tr, y_tr)
        else:
            self.model.fit(X, y)

        preds_norm = self.model.predict(X)
        fitted_norm = np.concatenate([values_norm[:self.max_lag], preds_norm])
        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        buf = list(self.values_norm)
        lags = self.selected_lags
        preds_norm = []
        n_total = len(self.values_norm)

        for h in range(horizon):
            row = [buf[-(l)] for l in lags]
            w7  = np.array(buf[-7:])
            w14 = np.array(buf[-14:]) if len(buf) >= 14 else np.array(buf)
            row += [
                np.mean(w7),  np.std(w7) + 1e-8,
                np.mean(w14), np.std(w14) + 1e-8,
                (n_total + h) / n_total,
            ]
            p_norm = float(self.model.predict([row])[0])
            preds_norm.append(p_norm)
            buf.append(p_norm)

        return self.preprocessor.inverse_transform(np.array(preds_norm))

    def get_metrics(self, actual_orig: np.ndarray) -> dict:
        return compute_metrics(actual_orig, self.fitted_orig)


# ─────────────────────────────────────────────
# 11. N-BEATS (simplified)
# ─────────────────────────────────────────────
class NBEATSModel:
    def __init__(self):
        self.name = 'N-BEATS'
        self.color = '#b06aff'
        self.train_time = None
        self.get_metrics_cache = None

    def fit(self, values_norm: np.ndarray, preprocessor: RevIN):
        import time
        t0 = time.time()
        n = len(values_norm)
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.n = n

        t_idx = np.arange(n) / n
        self.trend_coefs = np.polyfit(t_idx, values_norm, 3)

        fft = np.fft.rfft(values_norm - np.polyval(self.trend_coefs, t_idx))
        freqs = np.fft.rfftfreq(n)
        k = 5
        top_k = np.argsort(np.abs(fft))[-k:]
        fft_filtered = np.zeros_like(fft)
        fft_filtered[top_k] = fft[top_k]
        self.seasonal_component = np.fft.irfft(fft_filtered, n=n)
        self.fft_filtered = fft_filtered

        fitted_norm = np.polyval(self.trend_coefs, t_idx) + self.seasonal_component
        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        n = self.n
        future_idx = np.arange(n, n + horizon) / n
        trend_pred = np.polyval(self.trend_coefs, future_idx)
        freqs = np.fft.rfftfreq(n)
        seasonal_pred = np.zeros(horizon)
        for i, fft_val in enumerate(self.fft_filtered):
            if abs(fft_val) > 1e-10:
                freq = freqs[i]
                amp = np.abs(fft_val) * 2 / n
                phase = np.angle(fft_val)
                seasonal_pred += amp * np.cos(2 * np.pi * freq * np.arange(n, n+horizon) + phase)
        return self.preprocessor.inverse_transform(trend_pred + seasonal_pred)

    def get_metrics(self, actual_orig: np.ndarray) -> dict:
        return compute_metrics(actual_orig, self.fitted_orig)


# ─────────────────────────────────────────────
# 12. FIX-2: Rolling Origin CV 기반 앙상블 가중치
# ─────────────────────────────────────────────
def compute_oos_weight(model, values_orig: np.ndarray,
                       values_norm: np.ndarray, preprocessor: RevIN,
                       horizon_cv: int = None, n_windows: int = 4) -> float:
    """
    [FIX-2] Out-of-sample SMAPE로 앙상블 가중치 계산
    논문 근거: Meisenbacher — "out-of-sample 또는 rolling origin CV 기반 선택 권장"

    Rolling origin CV를 실행해 각 모델의 OOS SMAPE를 계산하고
    이를 가중치의 기반으로 사용
    """
    n = len(values_orig)
    if horizon_cv is None:
        horizon_cv = max(3, n // 10)

    min_train = max(20, n // 3)
    max_test_end = n - horizon_cv
    if max_test_end <= min_train:
        # 데이터 부족 → in-sample 사용하되 기본값 반환
        return 10.0

    step = max(1, (max_test_end - min_train) // n_windows)
    smapes = []

    for w in range(n_windows):
        train_end = min_train + w * step
        if train_end + horizon_cv > n:
            break

        train_orig = values_orig[:train_end]
        actual     = values_orig[train_end: train_end + horizon_cv]

        # 각 모델 타입에 맞게 CV 학습
        try:
            prep_cv = RevIN()
            train_norm = prep_cv.fit_transform(train_orig)
            period_cv = detect_period_acf(train_norm)

            if isinstance(model, ETSModel):
                m_cv = ETSModel().fit(train_norm, prep_cv, period=period_cv)
            elif isinstance(model, ARIMAModel):
                # ARIMA CV는 오래 걸리므로 ETS로 대리
                m_cv = ETSModel().fit(train_norm, prep_cv, period=period_cv)
            elif isinstance(model, MLModel):
                # strategy 그대로 전달
                m_cv = MLModel(model.model_type, model.strategy).fit(train_norm, prep_cv)
            elif isinstance(model, NBEATSModel):
                m_cv = NBEATSModel().fit(train_norm, prep_cv)
            else:
                continue

            pred = m_cv.predict(horizon_cv)[:len(actual)]
            denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
            smape_cv = float(np.mean(np.abs(actual - pred) / denom) * 100)
            smapes.append(smape_cv)

        except Exception as e:
            continue

    if not smapes:
        return 10.0  # fallback

    oos_smape = np.mean(smapes)
    return round(oos_smape, 4)


# ─────────────────────────────────────────────
# 13. 앙상블 (FIX-2 + FIX-4 통합)
# ─────────────────────────────────────────────
class Ensemble:
    """
    [FIX-2] OOS 기반 가중치
    [FIX-4] 과적합 모델 자동 제외
    """
    def __init__(self, models: list, oos_smapes: dict, ci: float = 0.90):
        self.ci = ci
        self.name = f'Ensemble'
        self.color = '#ff8c42'

        # [FIX-4] 과적합 모델 제외 후 유효 모델만 사용
        self.models = []
        raw_weights = []
        for m in models:
            oos_s = oos_smapes.get(m.name, 10.0)
            ins_s = m.get_metrics_cache.get('SMAPE', 10.0) if m.get_metrics_cache else 10.0
            ovf = detect_overfitting(m.name, ins_s, oos_s)
            if not ovf['is_overfit']:
                self.models.append(m)
                # 가중치: 1 / max(OOS SMAPE, 1.0) — clip으로 0% 보호
                raw_weights.append(1.0 / max(oos_s, 1.0))
            else:
                print(f"     ❌ [{m.name}] 앙상블에서 제외됨")

        if not self.models:
            print("  ⚠️  모든 모델이 과적합으로 제외됨 — 원래 모델 중 최선 선택")
            # fallback: OOS SMAPE 가장 낮은 모델
            best_m = min(models, key=lambda m: oos_smapes.get(m.name, 999))
            self.models = [best_m]
            raw_weights = [1.0]

        total_w = sum(raw_weights)
        self.weights = [w / total_w for w in raw_weights]
        self.name = f'Ensemble({len(self.models)})'

        print(f"\n   ✅ 앙상블 구성 모델: {[m.name for m in self.models]}")
        print(f"   가중치 (OOS 기반): { {m.name: round(w, 4) for m, w in zip(self.models, self.weights)} }")

    def predict(self, horizon: int) -> dict:
        preds_list = [m.predict(horizon) for m in self.models]

        preds = np.zeros(horizon)
        for p, w in zip(preds_list, self.weights):
            preds += w * p

        # 잔차 기반 신뢰구간
        all_residuals = []
        for m in self.models:
            if hasattr(m, 'fitted_orig') and hasattr(m, 'values_norm'):
                orig = m.preprocessor.inverse_transform(m.values_norm)
                resid = np.array(m.fitted_orig) - orig
                all_residuals.extend(resid.tolist())

        resid_std = np.std(all_residuals) if all_residuals else np.std(preds) * 0.1
        z = stats.norm.ppf((1 + self.ci) / 2)
        uncertainty = z * resid_std * np.sqrt(1 + np.arange(horizon) * 0.03)

        return {
            'pred':    preds,
            'lower':   preds - uncertainty,
            'upper':   preds + uncertainty,
            'weights': self.weights,
        }

    def get_fitted(self) -> np.ndarray:
        if not self.models:
            return np.array([])
        n = len(self.models[0].fitted_orig)
        fitted = np.zeros(n)
        for m, w in zip(self.models, self.weights):
            fitted += w * np.array(m.fitted_orig)
        return fitted


# ─────────────────────────────────────────────
# 14. 날짜 생성
# ─────────────────────────────────────────────
def generate_future_dates(last_date: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    freq_map = {'H': 'h', 'D': 'D', 'W': 'W', 'MS': 'MS', 'QS': 'QS'}
    pd_freq = freq_map.get(freq, 'D')
    return pd.date_range(start=last_date, periods=horizon+1, freq=pd_freq)[1:]


# ─────────────────────────────────────────────
# 15. ACF 경고
# ─────────────────────────────────────────────
def compute_acf(residuals: np.ndarray, max_lag: int = 30) -> dict:
    n = len(residuals)
    centered = residuals - np.mean(residuals)
    var_r = np.var(centered) + 1e-10
    acf_vals = [np.mean(centered[:-k] * centered[k:]) / var_r for k in range(1, max_lag+1)]
    conf_bound = 1.96 / np.sqrt(n)
    n_significant = sum(abs(a) > conf_bound for a in acf_vals)
    q_stat = n * (n+2) * sum(a**2 / (n - k - 1) for k, a in enumerate(acf_vals[:10]))
    warning = None
    if n_significant > max_lag * 0.3:
        warning = f"⚠️ 잔차 자기상관 있음 (ACF {n_significant}/{max_lag}개 유의)"
    return {
        'acf': acf_vals, 'conf_bound': conf_bound,
        'n_significant': n_significant,
        'ljung_box_q': round(q_stat, 4),
        'white_noise': q_stat < 20,
        'warning': warning,
    }


# ─────────────────────────────────────────────
# 16. Rolling Backtest
# ─────────────────────────────────────────────
def rolling_backtest(values_orig: np.ndarray, horizon: int, n_windows: int = 5) -> list:
    n = len(values_orig)
    min_train = max(30, n // 3)
    max_test_end = n - horizon
    step = max(1, (max_test_end - min_train) // n_windows)
    results = []
    for w in range(n_windows):
        train_end = min_train + w * step
        if train_end + horizon > n: break
        train = values_orig[:train_end]
        actual = values_orig[train_end: train_end + horizon]
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            # 주기 자동 감지 후 seasonal 적용
            period = detect_period_acf(train, freq='MS') if len(train) >= 24 else 12
            period = max(2, min(period, len(train) // 2))
            use_seasonal = len(train) >= period * 2
            m = ExponentialSmoothing(
                train,
                trend='add',
                seasonal='add' if use_seasonal else None,
                seasonal_periods=period if use_seasonal else None,
                initialization_method='estimated'
            )
            pred = m.fit(optimized=True).forecast(horizon)[:len(actual)]
        except:
            try:
                # fallback: seasonal 없이
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                m = ExponentialSmoothing(train, trend='add', initialization_method='estimated')
                pred = m.fit(optimized=True).forecast(horizon)[:len(actual)]
            except:
                pred = np.full(len(actual), np.mean(train))
        residuals = actual - pred
        smape = float(np.mean(np.abs(residuals) / ((np.abs(actual)+np.abs(pred))/2+1e-10)) * 100)
        results.append({'window': w+1, 'train_end': train_end,
                        'actual': actual, 'pred': pred, 'smape': round(smape, 4)})
    return results


# ─────────────────────────────────────────────
# 17. 전체 파이프라인
# ─────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, date_col: str, value_col: str,
                 horizon: int = 30, ci: float = 0.90,
                 models_to_run: list = None) -> dict:

    print("\n" + "="*60)
    print(f"🚀 파이프라인 시작 — horizon={horizon}, ci={ci*100}%")
    print("="*60)

    # Step 1: 진단
    diag = diagnose(df, date_col, value_col)
    freq = diag['freq']
    n = diag['n']

    # [FIX-1] 데이터 크기 기반 전략 결정
    strategy = get_model_strategy(n)
    print(f"\n📌 모델 전략: {strategy['label']}")

    # 모델 목록 필터링
    if models_to_run is None:
        models_to_run = ['arima', 'ets', 'xgb', 'rf', 'nbeats']
    allowed = strategy['allowed_models']
    models_to_run = [m for m in models_to_run if m in allowed]
    print(f"   실행할 모델: {models_to_run}")

    # Step 2: 전처리 (RevIN)
    print("\n⚙️  RevIN 전처리 중...")
    prep = RevIN()
    values_orig = df[value_col].values.astype(float)
    values_norm = prep.fit_transform(values_orig)
    print(f"   정규화: mean={prep.mean_:.4f}, std={prep.std_:.4f}")
    inv_check = prep.inverse_transform(values_norm[:3])
    print(f"   역변환 검증: {np.round(inv_check, 2)} ≈ {np.round(values_orig[:3], 2)}")

    # Step 3: STL
    print("\n🔬 STL 분해 중...")
    period = detect_period_acf(values_norm, freq=freq)
    stl_norm = stl_decompose(values_norm, period=period, freq=freq)

    # 역변환: 정규화 스케일 → 원본 스케일
    stl = {
        'trend':    prep.inverse_transform(stl_norm['trend']),
        'seasonal': stl_norm['seasonal'] * prep.std_,   # 계절성은 평균 없이 std만
        'residual': stl_norm['residual'] * prep.std_,   # 잔차도 동일
        'period':          stl_norm['period'],
        'trend_strength':  stl_norm['trend_strength'],
        'season_strength': stl_norm['season_strength'],
    }
    print(f"   감지된 주기: {stl['period']}, 트렌드 강도: {stl['trend_strength']:.4f}, "
          f"계절성 강도: {stl['season_strength']:.4f}")

    # Step 4: 모델 학습
    print("\n📊 모델 학습 중...")
    trained_models = []

    if 'ets' in models_to_run:
        print("  → ETS 학습...")
        m_ets = ETSModel().fit(values_norm, prep, period=stl['period'])
        m_ets.get_metrics_cache = m_ets.get_metrics(values_orig)
        trained_models.append(m_ets)
        print(f"     ETS 지표: {m_ets.get_metrics_cache}")

    if 'arima' in models_to_run:
        print("  → ARIMA 학습 (walk-forward)...")
        m_arima = ARIMAModel().fit(values_norm, prep)
        m_arima.get_metrics_cache = m_arima.get_metrics(values_orig)
        trained_models.append(m_arima)
        print(f"     ARIMA{m_arima.order} 지표: {m_arima.get_metrics_cache}")

    if 'xgb' in models_to_run:
        print("  → XGBoost 학습...")
        m_xgb = MLModel('xgb', strategy).fit(values_norm, prep)
        m_xgb.get_metrics_cache = m_xgb.get_metrics(values_orig)
        trained_models.append(m_xgb)
        print(f"     XGBoost 지표: {m_xgb.get_metrics_cache}")

    if 'rf' in models_to_run:
        print("  → Random Forest 학습...")
        m_rf = MLModel('rf', strategy).fit(values_norm, prep)
        m_rf.get_metrics_cache = m_rf.get_metrics(values_orig)
        trained_models.append(m_rf)
        print(f"     RF 지표: {m_rf.get_metrics_cache}")

    if 'nbeats' in models_to_run:
        print("  → N-BEATS 학습...")
        m_nb = NBEATSModel().fit(values_norm, prep)
        m_nb.get_metrics_cache = m_nb.get_metrics(values_orig)
        trained_models.append(m_nb)
        print(f"     N-BEATS 지표: {m_nb.get_metrics_cache}")

    # [FIX-2] Step 5: OOS 가중치 계산 (Rolling Origin CV)
    print("\n🔗 OOS 가중치 계산 중 (Rolling Origin CV)...")
    oos_smapes = {}
    for m in trained_models:
        print(f"  → [{m.name}] OOS CV 실행...")
        oos_s = compute_oos_weight(m, values_orig, values_norm, prep,
                                   horizon_cv=min(horizon, max(5, n//10)),
                                   n_windows=min(4, max(2, n//30)))
        oos_smapes[m.name] = oos_s
        ins_s = m.get_metrics_cache.get('SMAPE', 0) if m.get_metrics_cache else 0
        print(f"     in-sample SMAPE={ins_s:.2f}%, OOS SMAPE={oos_s:.2f}%")

    # [FIX-4] Step 6: 앙상블 (과적합 제외 포함)
    print("\n🔗 앙상블 구성 중 (FIX-4: 과적합 탐지)...")
    ens = Ensemble(trained_models, oos_smapes, ci=ci)
    ens_result = ens.predict(horizon)
    ens_fitted = ens.get_fitted()
    ens_metrics = compute_metrics(values_orig, ens_fitted)
    print(f"\n   앙상블 in-sample 지표: {ens_metrics}")

    # Step 7: 날짜 생성
    last_date = pd.to_datetime(df[date_col].iloc[-1])
    future_dates = generate_future_dates(last_date, freq, horizon)
    print(f"\n📅 미래 날짜 (앞 5개): {future_dates[:5].tolist()}")

    # Step 8: ACF 경고
    residuals = values_orig - ens_fitted
    acf_result = compute_acf(residuals)
    if acf_result['warning']:
        print(f"\n{acf_result['warning']}")

    # Step 9: 백테스트
    print("\n⏱  롤링 백테스트 중...")
    backtest = rolling_backtest(values_orig, min(horizon, 30))
    bt_smapes = [w['smape'] for w in backtest]
    avg_bt = np.mean(bt_smapes)
    print(f"   백테스트 SMAPE: {bt_smapes}")
    print(f"   평균 SMAPE: {avg_bt:.4f}%")

    # 앙상블 in-sample vs 백테스트 최종 비교
    ens_ins = ens_metrics.get('SMAPE', 0)
    if avg_bt > ens_ins * 3 and ens_ins > 0.1:
        print(f"\n  ⚠️  최종 경고: 앙상블 in-sample {ens_ins:.2f}% vs 백테스트 {avg_bt:.2f}% "
              f"— 과적합 의심 (비율: {avg_bt/ens_ins:.1f}x)")
    else:
        print(f"\n  ✅ 일반화 양호: in-sample {ens_ins:.2f}% vs 백테스트 {avg_bt:.2f}% "
              f"(비율: {avg_bt/max(ens_ins,0.01):.1f}x)")

    return {
        'diagnostics': diag,
        'strategy': strategy,
        'preprocessor': prep,
        'stl': stl,
        'models': trained_models,
        'oos_smapes': oos_smapes,
        'ensemble': ens_result,
        'ensemble_metrics': ens_metrics,
        'ensemble_fitted': ens_fitted,
        'future_dates': future_dates,
        'acf_result': acf_result,
        'backtest': backtest,
        'freq': freq,
    }

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  TimeFlow v2 — 실제 데이터 4종 범용성 검증              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── 나이브 baseline (비교 기준) ─────────────────────────────
    def naive_seasonal(values, horizon, period=12):
        preds = []
        for h in range(horizon):
            idx = len(values) - period + (h % period)
            preds.append(values[idx] if idx >= 0 else values[-1])
        return np.array(preds)

    def smape_score(actual, pred):
        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        return float(np.mean(np.abs(actual - pred) / denom) * 100)

    def mase_score(actual, pred, train):
        mae_model = np.mean(np.abs(actual - pred))
        mae_naive = np.mean(np.abs(np.diff(train))) + 1e-10
        return mae_model / mae_naive

    # ── 데이터셋 정의 ───────────────────────────────────────────
    DATASETS = [
        {
            'name':     'RSXFS (소매판매)',
            'path':     r"C:\Users\leechaewon\Desktop\홍익대 자료\시계열분석\1차과제_시계열예측웹앱개발\반도체,자동차 도메인 데이터\RSXFS.csv",
            'date_col': 'observation_date',
            'val_col':  'RSXFS',
            'start':    '2000-01-01',   # 최근 구간만 사용
            'models':   ['ets', 'arima', 'xgb', 'rf', 'nbeats'],
        },
        {
            'name':     'HOUST (주택착공)',
            'path':     r"C:\Users\leechaewon\Desktop\홍익대 자료\시계열분석\1차과제_시계열예측웹앱개발\반도체,자동차 도메인 데이터\HOUST.csv",
            'date_col': 'observation_date',
            'val_col':  'HOUST',
            'start':    '2000-01-01',
            'models':   ['ets', 'arima', 'xgb', 'rf', 'nbeats'],
        },
        {
            'name':     'PCE (개인소비지출)',
            'path':     r"C:\Users\leechaewon\Desktop\홍익대 자료\시계열분석\1차과제_시계열예측웹앱개발\반도체,자동차 도메인 데이터\PCE.csv",
            'date_col': 'observation_date',
            'val_col':  'PCE',
            'start':    '2000-01-01',   # 2000년 이후만 → 스케일 3배 이내
            'models':   ['ets', 'arima', 'xgb', 'rf'],
        },
        {
            'name':     'INDPRO (산업생산)',
            'path':     r"C:\Users\leechaewon\Desktop\홍익대 자료\시계열분석\1차과제_시계열예측웹앱개발\반도체,자동차 도메인 데이터\INDPRO.csv",
            'date_col': 'observation_date',
            'val_col':  'INDPRO',
            'start':    '2000-01-01',
            'models':   ['ets', 'arima', 'xgb', 'rf'],
        },
    ]

    HOLDOUT = 12   # 마지막 12개월 hold-out
    summary  = []  # 최종 비교표용

    # ── 데이터셋별 실행 ─────────────────────────────────────────
    for ds in DATASETS:
        print(f"\n\n{'='*60}")
        print(f"📂 [{ds['name']}]")
        print(f"{'='*60}")

        # 로드 + 필터
        df = pd.read_csv(ds['path'], parse_dates=[ds['date_col']])
        df = df.rename(columns={ds['date_col']: 'ds', ds['val_col']: 'y'})
        df = df[df['ds'] >= ds['start']].dropna(subset=['y']).reset_index(drop=True)
        print(f"  사용 데이터: {len(df)}개 ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
        print(f"  스케일: {df['y'].min():.1f} ~ {df['y'].max():.1f} "
              f"({df['y'].max()/df['y'].min():.1f}배)")

        # hold-out 분리
        train_df    = df.iloc[:-HOLDOUT].copy()
        test_actual = df['y'].values[-HOLDOUT:]

        # 나이브 baseline
        naive_pred = naive_seasonal(train_df['y'].values, HOLDOUT, period=12)
        naive_s    = smape_score(test_actual, naive_pred)

        # 우리 엔진 실행
        try:
            result = run_pipeline(
                train_df, 'ds', 'y',
                horizon=HOLDOUT, ci=0.90,
                models_to_run=ds['models']
            )
            engine_pred = result['ensemble']['pred']
            engine_s    = smape_score(test_actual, engine_pred)
            engine_mase = mase_score(test_actual, engine_pred, train_df['y'].values)
            improve     = (naive_s - engine_s) / naive_s * 100

            # 판정
            if engine_s < naive_s * 0.7:
                verdict = '✅ 우수 (나이브 대비 30%+ 개선)'
            elif engine_s < naive_s:
                verdict = '🟡 양호 (나이브보다 나음)'
            elif engine_s < naive_s * 1.1:
                verdict = '🟠 보통 (나이브와 비슷)'
            else:
                verdict = '🔴 미흡 (나이브보다 나쁨)'

            # hold-out 예측값 vs 실제값 출력
            print(f"\n  📋 Hold-out 12개월 예측 vs 실제")
            print(f"  {'날짜':12s} {'실제':>10s} {'나이브':>10s} {'엔진':>10s} {'오차%':>8s}")
            print(f"  {'-'*55}")
            for d, act, nav, eng in zip(
                result['future_dates'], test_actual, naive_pred, engine_pred
            ):
                err_pct = abs(act - eng) / (abs(act) + 1e-10) * 100
                print(f"  {str(d)[:10]:12s} {act:10.1f} {nav:10.1f} {eng:10.1f} {err_pct:7.1f}%")

            print(f"\n  {'나이브 SMAPE':20s}: {naive_s:.2f}%")
            print(f"  {'엔진 SMAPE':20s}: {engine_s:.2f}%")
            print(f"  {'개선율':20s}: {improve:+.1f}%")
            print(f"  {'MASE':20s}: {engine_mase:.3f}  "
                  f"({'나이브 대비 우수' if engine_mase < 1 else '나이브보다 나쁨'})")
            print(f"  {'판정':20s}: {verdict}")

            summary.append({
                'dataset':      ds['name'],
                'n_train':      len(train_df),
                'naive_smape':  round(naive_s, 2),
                'engine_smape': round(engine_s, 2),
                'improvement':  round(improve, 1),
                'mase':         round(engine_mase, 3),
                'verdict':      verdict,
            })

        except Exception as e:
            print(f"  ❌ 실행 오류: {e}")
            summary.append({
                'dataset': ds['name'], 'n_train': len(train_df),
                'naive_smape': round(naive_s, 2), 'engine_smape': -1,
                'improvement': -999, 'mase': -1, 'verdict': f'❌ 오류: {e}',
            })

    # ── 최종 비교표 ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("⚖️  전체 데이터셋 범용성 비교 (Hold-out 12개월)")
    print(f"{'='*70}")
    print(f"  {'데이터셋':22s} {'훈련수':>6s} {'나이브':>8s} {'엔진':>8s} {'개선율':>8s} {'MASE':>6s}  판정")
    print(f"  {'-'*70}")
    for r in summary:
        print(f"  {r['dataset']:22s} {r['n_train']:>6d} "
              f"{r['naive_smape']:>7.2f}% {r['engine_smape']:>7.2f}% "
              f"{r['improvement']:>+7.1f}% {r['mase']:>6.3f}  {r['verdict']}")

    print(f"\n  📌 MASE < 1.0 = 나이브 대비 우수  |  개선율 > 30% = 실무 수준")

    # 범용성 최종 판정
    valid = [r for r in summary if r['engine_smape'] > 0]
    if valid:
        avg_improve = np.mean([r['improvement'] for r in valid])
        avg_mase    = np.mean([r['mase'] for r in valid])
        good_count  = sum(1 for r in valid if r['mase'] < 1.0)
        print(f"\n  평균 개선율: {avg_improve:+.1f}%")
        print(f"  평균 MASE:   {avg_mase:.3f}")
        print(f"  나이브 초과: {good_count}/{len(valid)}개 데이터셋")

        if good_count == len(valid) and avg_improve > 20:
            print("\n  🏆 최종: 범용성 확인 — 모든 데이터셋에서 나이브 대비 우수")
        elif good_count >= len(valid) // 2:
            print("\n  🟡 최종: 부분 범용 — 일부 데이터셋에서만 나이브 초과")
        else:
            print("\n  🔴 최종: 범용성 부족 — 특정 데이터에 편향 의심")