"""
TimeFlow — FastAPI 백엔드
실행: python main.py
접속: http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io, sys, os, uvicorn, traceback

# forecast_engine_real.py 동일 폴더에 있어야 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from forecast_engine_real import run_pipeline
    ENGINE_OK = True
except ImportError:
    try:
        from forecast_engine_v2 import run_pipeline
        ENGINE_OK = True
    except ImportError:
        ENGINE_OK = False

app = FastAPI(title="TimeFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HTML 서빙 ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_redesigned.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ── 상태 확인 ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "engine": ENGINE_OK}

# ── 메인 예측 API ──────────────────────────────────────────
@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    date_col: str    = Form(...),
    val_col:  str    = Form(...),
    horizon:  int    = Form(12),
    ci:       int    = Form(90),
    models:   str    = Form("ets,arima,xgb,rf"),
):
    if not ENGINE_OK:
        return JSONResponse({"error": "forecast_engine_real.py를 같은 폴더에 두세요."}, status_code=500)

    try:
        # CSV 파싱
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # 컬럼 처리
        df = df.rename(columns={date_col: "ds", val_col: "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"]  = pd.to_numeric(df["y"], errors="coerce")
        df = df[["ds", "y"]].dropna().reset_index(drop=True)

        if len(df) < 20:
            return JSONResponse({"error": "데이터가 20개 미만입니다."}, status_code=400)

        # 모델 선택
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if not model_list:
            model_list = ["ets", "arima"]

        # 파이프라인 실행
        result = run_pipeline(
            df, "ds", "y",
            horizon=horizon,
            ci=ci / 100,
            models_to_run=model_list,
        )

        # JSON 직렬화 가능한 형태로 변환
        values_orig = df["y"].values.tolist()
        dates_orig  = [str(d)[:10] for d in df["ds"]]

        # 모델별 결과
        model_results = []
        model_colors  = {
            "ETS": "#00e5a0", "ARIMA": "#00d4ff",
            "XGBoost": "#ffd166", "GBM": "#ffd166",
            "RandomForest": "#ff8c42", "N-BEATS": "#b06aff",
        }
        for m in result["models"]:
            mc = m.get_metrics_cache or {}
            model_results.append({
                "name":     m.name,
                "smape":    round(float(mc.get("SMAPE", 0)), 4),
                "mae":      round(float(mc.get("MAE",   0)), 4),
                "rmse":     round(float(mc.get("RMSE",  0)), 4),
                "r2":       round(float(mc.get("R2",    0)), 4),
                "mase":     round(float(mc.get("MASE",  0)), 4),
                "color":    model_colors.get(m.name, "#8a9bb8"),
                "fitted":   _safe_list(getattr(m, "fitted_orig", [])),
                "trainTime": getattr(m, "train_time", 0) or 0,
            })

        # 앙상블 결과
        ens_metrics = result["ensemble_metrics"]
        ens_fitted  = _safe_list(result["ensemble_fitted"])
        ens_pred    = _safe_list(result["ensemble"]["pred"])
        ens_lower   = _safe_list(result["ensemble"]["lower"])
        ens_upper   = _safe_list(result["ensemble"]["upper"])
        ens_residuals = [float(v - f) for v, f in zip(values_orig, ens_fitted)]

        # STL
        stl = result["stl"]

        # 백테스트
        backtest_windows = []
        for w in result["backtest"]:
            backtest_windows.append({
                "window":     w["window"],
                "smape":      round(float(w["smape"]), 4),
                "cutoffDate": dates_orig[min(w["train_end"], len(dates_orig)-1)],
                "actual":     _safe_list(w["actual"]),
                "pred":       _safe_list(w["pred"]),
            })

        # ACF
        acf_result = result.get("acf_result", {})

        # OOS
        oos_smapes = {k: round(float(v), 4) for k, v in result.get("oos_smapes", {}).items()}

        return {
            "ok": True,
            "dates":  dates_orig,
            "values": values_orig,
            "futureDates": [str(d)[:10] for d in result["future_dates"]],
            "ensemble": {
                "pred":      ens_pred,
                "lower":     ens_lower,
                "upper":     ens_upper,
                "fitted":    ens_fitted,
                "residuals": ens_residuals,
                "mae":    round(float(ens_metrics.get("MAE",   0)), 4),
                "rmse":   round(float(ens_metrics.get("RMSE",  0)), 4),
                "smape":  round(float(ens_metrics.get("SMAPE", 0)), 4),
                "mape":   round(float(ens_metrics.get("MAPE",  0)), 4),
                "r2":     round(float(ens_metrics.get("R2",    0)), 4),
                "mase":   round(float(ens_metrics.get("MASE",  0)), 4),
            },
            "modelResults": model_results,
            "stl": {
                "trend":          _safe_list(stl["trend"]),
                "seasonal":       _safe_list(stl["seasonal"]),
                "residual":       _safe_list(stl["residual"]),
                "period":         int(stl["period"]),
                "trendStrength":  round(float(stl["trend_strength"]),  4),
                "seasonStrength": round(float(stl["season_strength"]), 4),
            },
            "backtest":   backtest_windows,
            "acf": {
                "vals":        [round(float(v), 4) for v in acf_result.get("acf", [])],
                "confBound":   round(float(acf_result.get("conf_bound", 0.196)), 4),
                "ljungBoxQ":   round(float(acf_result.get("ljung_box_q", 0)), 4),
                "whiteNoise":  bool(acf_result.get("white_noise", True)),
                "nSignificant": int(acf_result.get("n_significant", 0)),
            },
            "oosSMAPEs":    oos_smapes,
            "diagnostics":  {k: str(v) for k, v in result["diagnostics"].items()},
            "freq":         result["freq"],
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


def _safe_list(arr):
    """numpy array → JSON 직렬화 가능한 Python list"""
    if arr is None: return []
    try:
        result = []
        for v in arr:
            f = float(v)
            if np.isnan(f) or np.isinf(f):
                result.append(None)
            else:
                result.append(round(f, 6))
        return result
    except:
        return []


if __name__ == "__main__":
    print("=" * 55)
    print("  TimeFlow API 서버 시작")
    print("  접속: http://localhost:8002")
    print("  엔진:", "✅ 준비됨" if ENGINE_OK else "❌ forecast_engine_real.py 없음")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)