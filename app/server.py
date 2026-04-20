from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from app.api import fetch_target_history, get_runtime_config
from app.config import get_settings
from app.forecast import build_plotly_figure, run_changepoint_detection, run_probabilistic_forecast

app = FastAPI(title="5511 혼잡도 모니터링", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, object]:
    settings = get_settings()
    runtime = get_runtime_config(settings)
    runtime["api_key"] = "***masked***" if runtime.get("api_key") else ""
    return runtime


@app.get("/series")
def series(
    days: int = Query(default=45, ge=14, le=180),
    full_scan: bool = Query(default=False),
    refresh: bool = Query(default=False),
) -> JSONResponse:
    settings = get_settings()
    try:
        df = fetch_target_history(
            settings=settings,
            days=days,
            full_scan=full_scan,
            refresh=refresh,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"데이터 수집 실패: {exc}") from exc

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "캐시 데이터가 없습니다. 먼저 /sync로 수집하거나 refresh=true로 조회하세요."
            ),
        )
    payload = df.copy()
    payload["ds"] = payload["ds"].dt.strftime("%Y-%m-%d")
    return JSONResponse(payload.to_dict(orient="records"))


@app.get("/", response_class=HTMLResponse)
def dashboard(
    days: int = Query(default=60, ge=20, le=180),
    horizon: int = Query(default=7, ge=3, le=30),
    full_scan: bool = Query(default=False),
    refresh: bool = Query(default=False),
) -> HTMLResponse:
    settings = get_settings()

    try:
        df = fetch_target_history(
            settings=settings,
            days=days,
            full_scan=full_scan,
            refresh=refresh,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"데이터 수집 실패: {exc}") from exc

    if len(df) < 20:
        raise HTTPException(
            status_code=404,
            detail=(
                f"학습 가능한 캐시 데이터가 부족합니다(len={len(df)}). "
                "먼저 /sync?days=90 호출 후 다시 시도하세요."
            ),
        )

    try:
        forecast = run_probabilistic_forecast(df=df, horizon=horizon, alpha=0.1)
        cpd = run_changepoint_detection(df["y"])
        fig = build_plotly_figure(source_df=df, forecast=forecast, cpd=cpd)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"모델/시각화 생성 실패: {exc}") from exc

    warnings = "".join(f"<li>{w}</li>" for w in cpd.get("warnings", [])) or "<li>없음</li>"
    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>5511 혼잡도 대시보드</title>
      </head>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px;">
        <h2>5511 · 서울대입구역3번출구 · 간이 실시간 혼잡 시계열</h2>
        <p>
          데이터 포인트: <b>{len(df)}</b> /
          예측 지평: <b>{horizon}일</b> /
          full_scan: <b>{str(full_scan).lower()}</b> /
          refresh: <b>{str(refresh).lower()}</b>
        </p>
        <p>
          모델: <b>Quantile Regression + CP + CQR</b>,
          변화점: <b>PELT + SDAR(ChangeFinder)</b>
        </p>
        <ul>{warnings}</ul>
        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/sync")
def sync(
    days: int = Query(default=90, ge=14, le=365),
    full_scan: bool = Query(default=False),
) -> dict[str, object]:
    settings = get_settings()
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="SEOUL_TRAFFIC_API_KEY가 비어 있습니다.")
    try:
        df = fetch_target_history(
            settings=settings,
            days=days,
            full_scan=full_scan,
            refresh=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"동기화 실패: {exc}") from exc
    return {
        "synced_rows": len(df),
        "days": days,
        "full_scan": full_scan,
        "message": "database/target_history.csv 갱신 완료",
    }
