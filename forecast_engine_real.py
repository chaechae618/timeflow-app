"""
TimeFlow 예측 엔진 — 경량 배포 버전
모델: ETS + ARIMA + RandomForest (3개)
최적화: Render 무료 플랜 (512MB RAM, 저사양 CPU) 대응
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 주파수 감지 + 진단
# ─────────────────────────────────────────────
def detect_frequency(date_series: pd.Series) -> str:
    if len(date_series) < 2:
        return 'unknown'
    dates = pd.to_datetime(date_series).sort_values()
    hours = dates.diff().dropna().median().total_seconds() / 3600
    if hours <= 1:     return 'H'
    elif hours <= 25:  return 'D'
    elif hours <= 170: return 'W'
    elif hours <= 800: return 'MS'
    else:              return 'QS'


def diagnose(df, date_col, value_col):
    values = df[value_col].values.astype(float)
    n = len(values)
    null_mask = np.isnan(values)
    q1, q3 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
    iqr = q3 - q1
    outlier_count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    freq = detect_frequency(df[date_col])

    is_stationary, adf_pvalue = None, None
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = values[~null_mask]
        if len(clean) >= 20:
            res = adfuller(clean, autolag='AIC')
            is_stationary = res[1] < 0.05
            adf_pvalue = round(res[1], 4)
    except Exception:
        pass

    return {
        'n': n,
        'null_count': int(null_mask.sum()),
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


# ─────────────────────────────────────────────
# 2. 전처리 (RevIN 경량화)
# ─────────────────────────────────────────────
class RevIN:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.fitted = False
        self.log_transform = False

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        v = values.copy().astype(float)
        # 결측치 보간
        nan_idx = np.where(np.isnan(v))[0]
        for i in nan_idx:
            left  = v[:i][~np.isnan(v[:i])]
            right = v[i+1:][~np.isnan(v[i+1:])]
            if len(left) and len(right):  v[i] = (left[-1] + right[0]) / 2
            elif len(left):               v[i] = left[-1]
            elif len(right):              v[i] = right[0]
        # 이상치 클리핑
        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        v = np.clip(v, q1 - 2.5 * (q3 - q1), q3 + 2.5 * (q3 - q1))
        # 로그 변환 (스케일 100배 이상일 때)
        v_pos = v[v > 0]
        if len(v_pos) > 0 and v.max() / (v_pos.min() + 1e-10) > 100:
            self.log_transform = True
            v = np.log1p(v)
        self.mean_ = np.mean(v)
        self.std_  = np.std(v) + self.eps
        self.fitted = True
        return (v - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        result = x * self.std_ + self.mean_
        if self.log_transform:
            result = np.expm1(result)
        return result


Preprocessor = RevIN


# ─────────────────────────────────────────────
# 3. STL 분해
# ─────────────────────────────────────────────
def detect_period(values: np.ndarray, freq: str) -> int:
    defaults = {'MS': 12, 'QS': 4, 'W': 52, 'D': 7, 'H': 24}
    return defaults.get(freq, 7)


def stl_decompose(values: np.ndarray, period: int, freq: str) -> dict:
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(values, period=max(2, period), robust=True).fit()
        trend, seasonal, residual = res.trend, res.seasonal, res.resid
    except Exception:
        n = len(values)
        w = min(15, n // 4)
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
# 4. 평가 지표
# ─────────────────────────────────────────────
def compute_metrics(actual, predicted):
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    res = a - p
    mae  = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res**2)))
    denom = (np.abs(a) + np.abs(p)) / 2 + 1e-10
    smape = float(np.mean(np.abs(res) / denom) * 100)
    nz = np.abs(a) > 1e-6
    mape = float(np.mean(np.abs(res[nz] / a[nz])) * 100) if nz.sum() > 0 else float('nan')
    ss_res = np.sum(res**2)
    ss_tot = np.sum((a - np.mean(a))**2) + 1e-10
    r2 = float(1 - ss_res / ss_tot)
    naive_mae = np.mean(np.abs(np.diff(a))) + 1e-10
    return {
        'MAE':   round(mae, 4),
        'RMSE':  round(rmse, 4),
        'SMAPE': round(smape, 4),
        'MAPE':  round(mape, 4),
        'R2':    round(r2, 4),
        'MASE':  round(mae / naive_mae, 4),
    }


# ─────────────────────────────────────────────
# 5. ETS 모델
# ─────────────────────────────────────────────
class ETSModel:
    def __init__(self):
        self.name = 'ETS'
        self.color = '#00e5a0'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor, period=12):
        import time
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        t0 = time.time()
        n = len(values_norm)
        try:
            use_seasonal = period >= 2 and n >= period * 2
            m = ExponentialSmoothing(
                values_norm, trend='add',
                seasonal='add' if use_seasonal else None,
                seasonal_periods=period if use_seasonal else None,
                initialization_method='estimated'
            )
            self.model_fit = m.fit(optimized=True)
        except Exception:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(values_norm, trend='add',
                                     initialization_method='estimated')
            self.model_fit = m.fit(optimized=True)

        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.fitted_orig = preprocessor.inverse_transform(
            np.array(self.model_fit.fittedvalues))
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        return self.preprocessor.inverse_transform(
            np.array(self.model_fit.forecast(horizon)))

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


# ─────────────────────────────────────────────
# 6. ARIMA 모델 (경량화: 탐색 범위 축소)
# ─────────────────────────────────────────────
class ARIMAModel:
    def __init__(self):
        self.name = 'ARIMA'
        self.color = '#00d4ff'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor):
        import time
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        t0 = time.time()
        n = len(values_norm)

        # 차분 차수 결정
        d = 0
        try:
            if adfuller(values_norm)[1] > 0.05:
                d = 1
        except Exception:
            d = 1

        # ★ 경량화: p 0~2, q 0~1 만 탐색 (기존 p0~4, q0~2 → 대폭 축소)
        best_aic, best_order = np.inf, (1, d, 1)
        for p in range(0, 3):
            for q in range(0, 2):
                try:
                    fit = ARIMA(values_norm, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p, d, q)
                except Exception:
                    continue

        self.order = best_order
        self.name = f'ARIMA{best_order}'
        self.model_fit = ARIMA(values_norm, order=best_order).fit()
        self.preprocessor = preprocessor
        self.values_norm = values_norm

        # ★ 경량화: walk-forward 제거 → in-sample fitted 사용
        fitted_norm = np.array(self.model_fit.fittedvalues)
        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        fc = self.model_fit.forecast(steps=horizon)
        return self.preprocessor.inverse_transform(
            fc.values if hasattr(fc, 'values') else np.array(fc))

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


# ─────────────────────────────────────────────
# 7. RandomForest 모델 (경량화: 트리 수 축소)
# ─────────────────────────────────────────────
class RFModel:
    def __init__(self):
        self.name = 'RandomForest'
        self.color = '#ff8c42'
        self.train_time = 0
        self.get_metrics_cache = None

    def _make_features(self, values, lags):
        max_lag = max(lags)
        X, y = [], []
        for i in range(max_lag, len(values)):
            row = [values[i - l] for l in lags]
            w = values[max(0, i-7):i]
            row += [np.mean(w), np.std(w) + 1e-8, i / len(values)]
            X.append(row)
            y.append(values[i])
        return np.array(X), np.array(y), max_lag

    def fit(self, values_norm, preprocessor):
        import time
        t0 = time.time()
        n = len(values_norm)

        # ★ 경량화: 고정 lag [1,2,3] 사용 (PACF 탐색 제거)
        lags = [1, 2, 3]
        if n >= 14: lags.append(7)
        if n >= 24: lags.append(12)

        X, y, self.max_lag = self._make_features(values_norm, lags)
        self.lags = lags

        # ★ 경량화: n_estimators=30, max_depth=4 (기존 100, 6)
        self.model = RandomForestRegressor(
            n_estimators=30, max_depth=4,
            min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
        self.model.fit(X, y)

        preds_norm = self.model.predict(X)
        fitted_norm = np.concatenate([values_norm[:self.max_lag], preds_norm])
        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        buf = list(self.values_norm)
        n_total = len(self.values_norm)
        preds = []
        for h in range(horizon):
            row = [buf[-l] for l in self.lags]
            w = np.array(buf[-7:])
            row += [np.mean(w), np.std(w) + 1e-8, (n_total + h) / n_total]
            p = float(self.model.predict([row])[0])
            preds.append(p)
            buf.append(p)
        return self.preprocessor.inverse_transform(np.array(preds))

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


# ─────────────────────────────────────────────
# 8. OOS 가중치 (윈도우 1개 — 경량화)
# ─────────────────────────────────────────────
def compute_oos_weight(model, values_orig, preprocessor, horizon_cv):
    """윈도우 1개만 사용하는 경량 OOS 가중치 계산"""
    n = len(values_orig)
    train_end = int(n * 0.8)
    if train_end + horizon_cv > n:
        return 10.0

    train = values_orig[:train_end]
    actual = values_orig[train_end:train_end + horizon_cv]

    try:
        prep_cv = RevIN()
        train_norm = prep_cv.fit_transform(train)
        period_cv = detect_period(train_norm, 'MS')

        if isinstance(model, ETSModel):
            m_cv = ETSModel().fit(train_norm, prep_cv, period=period_cv)
        elif isinstance(model, ARIMAModel):
            m_cv = ETSModel().fit(train_norm, prep_cv, period=period_cv)
        elif isinstance(model, RFModel):
            m_cv = RFModel().fit(train_norm, prep_cv)
        else:
            return 10.0

        pred = m_cv.predict(horizon_cv)[:len(actual)]
        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        return float(round(np.mean(np.abs(actual - pred) / denom) * 100, 4))
    except Exception:
        return 10.0


# ─────────────────────────────────────────────
# 9. 앙상블
# ─────────────────────────────────────────────
class Ensemble:
    def __init__(self, models, oos_smapes, ci=0.90):
        self.ci = ci
        self.models = models
        raw_weights = [1.0 / max(oos_smapes.get(m.name, 10.0), 1.0) for m in models]
        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights]
        self.name = f'Ensemble({len(models)})'

    def predict(self, horizon):
        preds = np.zeros(horizon)
        for m, w in zip(self.models, self.weights):
            preds += w * m.predict(horizon)

        all_residuals = []
        for m in self.models:
            if hasattr(m, 'fitted_orig') and hasattr(m, 'values_norm'):
                orig = m.preprocessor.inverse_transform(m.values_norm)
                all_residuals.extend((np.array(m.fitted_orig) - orig).tolist())

        resid_std = np.std(all_residuals) if all_residuals else np.std(preds) * 0.1
        z = stats.norm.ppf((1 + self.ci) / 2)
        uncertainty = z * resid_std * np.sqrt(1 + np.arange(horizon) * 0.03)
        return {
            'pred':  preds,
            'lower': preds - uncertainty,
            'upper': preds + uncertainty,
            'weights': self.weights,
        }

    def get_fitted(self):
        n = len(self.models[0].fitted_orig)
        fitted = np.zeros(n)
        for m, w in zip(self.models, self.weights):
            fitted += w * np.array(m.fitted_orig)
        return fitted


# ─────────────────────────────────────────────
# 10. ACF 진단
# ─────────────────────────────────────────────
def compute_acf(residuals, max_lag=20):
    n = len(residuals)
    centered = residuals - np.mean(residuals)
    var_r = np.var(centered) + 1e-10
    acf_vals = [float(np.mean(centered[:-k] * centered[k:]) / var_r)
                for k in range(1, max_lag + 1)]
    conf_bound = 1.96 / np.sqrt(n)
    n_sig = sum(abs(a) > conf_bound for a in acf_vals)
    q = n * (n+2) * sum(a**2 / (n - k - 1) for k, a in enumerate(acf_vals[:10]))
    return {
        'acf': acf_vals,
        'conf_bound': conf_bound,
        'n_significant': n_sig,
        'ljung_box_q': round(q, 4),
        'white_noise': q < 20,
        'warning': f'잔차 자기상관 있음 ({n_sig}/{max_lag})' if n_sig > max_lag * 0.3 else None,
    }


# ─────────────────────────────────────────────
# 11. 롤링 백테스트 (윈도우 3개 — 경량화)
# ─────────────────────────────────────────────
def rolling_backtest(values_orig, horizon, n_windows=3):
    n = len(values_orig)
    min_train = max(30, n // 3)
    results = []
    step = max(1, (n - horizon - min_train) // n_windows)

    for w in range(n_windows):
        train_end = min_train + w * step
        if train_end + horizon > n:
            break
        train  = values_orig[:train_end]
        actual = values_orig[train_end:train_end + horizon]
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(train, trend='add',
                                     initialization_method='estimated')
            pred = np.array(m.fit(optimized=True).forecast(horizon))[:len(actual)]
        except Exception:
            pred = np.full(len(actual), np.mean(train))

        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        smape = float(np.mean(np.abs(actual - pred) / denom) * 100)
        results.append({
            'window': w + 1, 'train_end': train_end,
            'actual': actual, 'pred': pred, 'smape': round(smape, 4),
        })
    return results


# ─────────────────────────────────────────────
# 12. 날짜 생성
# ─────────────────────────────────────────────
def generate_future_dates(last_date, freq, horizon):
    freq_map = {'H': 'h', 'D': 'D', 'W': 'W', 'MS': 'MS', 'QS': 'QS'}
    return pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq=freq_map.get(freq, 'D')
    )[1:]


# ─────────────────────────────────────────────
# 13. 메인 파이프라인
# ─────────────────────────────────────────────
def run_pipeline(df, date_col, value_col,
                 horizon=12, ci=0.90, models_to_run=None):

    if models_to_run is None:
        models_to_run = ['ets', 'arima', 'rf']

    # 통계 모델만 허용하는 모델명 정규화
    # (xgb, nbeats 요청이 와도 무시)
    allowed = ['ets', 'arima', 'rf']
    models_to_run = [m for m in models_to_run if m in allowed]
    if not models_to_run:
        models_to_run = ['ets', 'arima', 'rf']

    # Step 1: 진단
    diag = diagnose(df, date_col, value_col)
    freq = diag['freq']
    n    = diag['n']

    strategy = {
        'label': f'경량 배포 버전 (n={n})',
        'allowed_models': allowed,
        'max_lags': 3,
        'xgb_params': None,
        'rf_params': {'n_estimators': 30, 'max_depth': 4},
    }

    # Step 2: 전처리
    prep = RevIN()
    values_orig = df[value_col].values.astype(float)
    values_norm = prep.fit_transform(values_orig)

    # Step 3: STL
    period = detect_period(values_norm, freq)
    stl_norm = stl_decompose(values_norm, period=period, freq=freq)
    stl = {
        'trend':          prep.inverse_transform(stl_norm['trend']),
        'seasonal':       stl_norm['seasonal'] * prep.std_,
        'residual':       stl_norm['residual'] * prep.std_,
        'period':         stl_norm['period'],
        'trend_strength': stl_norm['trend_strength'],
        'season_strength':stl_norm['season_strength'],
    }

    # Step 4: 모델 학습
    trained_models = []

    if 'ets' in models_to_run:
        m = ETSModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = m.get_metrics(values_orig)
        trained_models.append(m)

    if 'arima' in models_to_run:
        m = ARIMAModel().fit(values_norm, prep)
        m.get_metrics_cache = m.get_metrics(values_orig)
        trained_models.append(m)

    if 'rf' in models_to_run:
        m = RFModel().fit(values_norm, prep)
        m.get_metrics_cache = m.get_metrics(values_orig)
        trained_models.append(m)

    # Step 5: OOS 가중치 (윈도우 1개)
    horizon_cv = min(horizon, max(3, n // 10))
    oos_smapes = {}
    for m in trained_models:
        oos_smapes[m.name] = compute_oos_weight(m, values_orig, prep, horizon_cv)

    # Step 6: 앙상블
    ens = Ensemble(trained_models, oos_smapes, ci=ci)
    ens_result  = ens.predict(horizon)
    ens_fitted  = ens.get_fitted()
    ens_metrics = compute_metrics(values_orig, ens_fitted)

    # Step 7: 날짜 생성
    last_date    = pd.to_datetime(df[date_col].iloc[-1])
    future_dates = generate_future_dates(last_date, freq, horizon)

    # Step 8: ACF
    residuals  = values_orig - ens_fitted
    acf_result = compute_acf(residuals)

    # Step 9: 백테스트 (윈도우 3개)
    backtest = rolling_backtest(values_orig, min(horizon, 12), n_windows=3)

    return {
        'diagnostics':      diag,
        'strategy':         strategy,
        'preprocessor':     prep,
        'stl':              stl,
        'models':           trained_models,
        'oos_smapes':       oos_smapes,
        'ensemble':         ens_result,
        'ensemble_metrics': ens_metrics,
        'ensemble_fitted':  ens_fitted,
        'future_dates':     future_dates,
        'acf_result':       acf_result,
        'backtest':         backtest,
        'freq':             freq,
    }
