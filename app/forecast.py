from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots


@dataclass
class ForecastResult:
    history: pd.DataFrame
    future: pd.DataFrame
    alpha: float
    cp_interval: float
    cqr_slack: float


def _make_features(ds: pd.Series) -> pd.DataFrame:
    idx = np.arange(len(ds), dtype=float)
    dow = ds.dt.dayofweek.astype(float).to_numpy()
    return pd.DataFrame(
        {
            "t": idx,
            "sin_week": np.sin(2.0 * np.pi * dow / 7.0),
            "cos_week": np.cos(2.0 * np.pi * dow / 7.0),
        }
    )


def _fit_quantile(
    y: pd.Series,
    x: pd.DataFrame,
    quantile: float,
) -> Any:
    x_const = sm.add_constant(x, has_constant="add")
    model = sm.QuantReg(y, x_const)
    return model.fit(q=quantile)


def run_probabilistic_forecast(
    df: pd.DataFrame,
    horizon: int = 7,
    alpha: float = 0.1,
) -> ForecastResult:
    if len(df) < 20:
        raise ValueError("시계열 포인트가 너무 적습니다. 최소 20개 이상 필요합니다.")

    data = df[["ds", "y"]].copy().sort_values("ds").reset_index(drop=True)
    n = len(data)
    train_end = max(int(n * 0.7), 12)
    cal_end = max(int(n * 0.85), train_end + 4)

    train = data.iloc[:train_end].copy()
    cal = data.iloc[train_end:cal_end].copy()
    full = data.iloc[:cal_end].copy()

    x_train = _make_features(train["ds"])
    x_cal = _make_features(cal["ds"])
    x_full = _make_features(full["ds"])

    fit_median = _fit_quantile(train["y"], x_train, 0.5)
    fit_low = _fit_quantile(train["y"], x_train, alpha / 2.0)
    fit_high = _fit_quantile(train["y"], x_train, 1.0 - alpha / 2.0)

    pred_cal_median = fit_median.predict(sm.add_constant(x_cal, has_constant="add"))
    pred_cal_low = fit_low.predict(sm.add_constant(x_cal, has_constant="add"))
    pred_cal_high = fit_high.predict(sm.add_constant(x_cal, has_constant="add"))

    # Naive CP: |y - y_hat50| 오차 분위수로 고정폭 구간 생성
    abs_residual = np.abs(cal["y"].to_numpy() - pred_cal_median.to_numpy())
    cp_interval = float(np.quantile(abs_residual, 1.0 - alpha)) if len(abs_residual) else 0.0

    # CQR: max(q_low - y, y - q_high)의 분위수로 구간 보정
    cqr_scores = np.maximum(
        pred_cal_low.to_numpy() - cal["y"].to_numpy(),
        cal["y"].to_numpy() - pred_cal_high.to_numpy(),
    )
    cqr_slack = float(np.quantile(cqr_scores, 1.0 - alpha)) if len(cqr_scores) else 0.0

    last_day = data["ds"].iloc[-1]
    future_ds = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_all_ds = pd.concat([full["ds"], pd.Series(future_ds)], ignore_index=True)
    x_future_all = _make_features(future_all_ds)

    fit_median_full = _fit_quantile(full["y"], x_full, 0.5)
    fit_low_full = _fit_quantile(full["y"], x_full, alpha / 2.0)
    fit_high_full = _fit_quantile(full["y"], x_full, 1.0 - alpha / 2.0)

    pred_median_all = fit_median_full.predict(sm.add_constant(x_future_all, has_constant="add"))
    pred_low_all = fit_low_full.predict(sm.add_constant(x_future_all, has_constant="add"))
    pred_high_all = fit_high_full.predict(sm.add_constant(x_future_all, has_constant="add"))

    output = pd.DataFrame(
        {
            "ds": future_all_ds,
            "yhat50": pred_median_all,
            "qr_low": pred_low_all,
            "qr_high": pred_high_all,
        }
    )
    output["cp_low"] = output["yhat50"] - cp_interval
    output["cp_high"] = output["yhat50"] + cp_interval
    output["cqr_low"] = output["qr_low"] - cqr_slack
    output["cqr_high"] = output["qr_high"] + cqr_slack

    history = output.iloc[: len(full)].copy().reset_index(drop=True)
    future = output.iloc[len(full) :].copy().reset_index(drop=True)
    return ForecastResult(
        history=history,
        future=future,
        alpha=alpha,
        cp_interval=cp_interval,
        cqr_slack=cqr_slack,
    )


def run_changepoint_detection(
    series: pd.Series,
    pelt_penalty: float = 8.0,
    sdar_r: float = 0.01,
    sdar_order: int = 3,
    sdar_smooth: int = 5,
) -> dict[str, Any]:
    y = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    result: dict[str, Any] = {
        "pelt_idx": [],
        "sdar_scores": [],
        "sdar_idx": [],
        "warnings": [],
    }
    if len(y) < 8:
        result["warnings"].append("CPD 실행을 위한 포인트가 부족합니다.")
        return result

    try:
        import ruptures as rpt

        model = rpt.Pelt(model="rbf").fit(y)
        cp = model.predict(pen=pelt_penalty)
        result["pelt_idx"] = [int(i) for i in cp if i < len(y)]
    except Exception as exc:
        result["warnings"].append(f"PELT 실행 실패: {exc}")

    try:
        import changefinder

        cf = changefinder.ChangeFinder(r=sdar_r, order=sdar_order, smooth=sdar_smooth)
        scores = [float(cf.update(float(v))) for v in y]
        result["sdar_scores"] = scores
        if scores:
            threshold = float(np.quantile(scores, 0.95))
            result["sdar_idx"] = [i for i, s in enumerate(scores) if s >= threshold]
    except Exception as exc:
        result["warnings"].append(f"SDAR(ChangeFinder) 실행 실패: {exc}")

    return result


def build_plotly_figure(
    source_df: pd.DataFrame,
    forecast: ForecastResult,
    cpd: dict[str, Any],
) -> go.Figure:
    df = source_df.sort_values("ds").reset_index(drop=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines+markers",
            name="실측 혼잡도(재차인원)",
            line={"color": "#1f77b4"},
        ),
        secondary_y=False,
    )

    fut = forecast.future
    fig.add_trace(
        go.Scatter(
            x=fut["ds"],
            y=fut["yhat50"],
            mode="lines+markers",
            name="QR 중앙예측(50%)",
            line={"color": "#ff7f0e"},
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=fut["ds"],
            y=fut["cqr_high"],
            mode="lines",
            name="CQR 상한",
            line={"width": 0},
            showlegend=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=fut["ds"],
            y=fut["cqr_low"],
            mode="lines",
            name="CQR 하한",
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.2)",
            line={"width": 0},
            showlegend=True,
        ),
        secondary_y=False,
    )

    if cpd.get("sdar_scores"):
        fig.add_trace(
            go.Scatter(
                x=df["ds"],
                y=cpd["sdar_scores"],
                mode="lines",
                name="SDAR 점수",
                line={"color": "#9467bd", "dash": "dot"},
            ),
            secondary_y=True,
        )

    for idx in cpd.get("pelt_idx", []):
        if 0 <= idx < len(df):
            x = df["ds"].iloc[idx]
            fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#d62728")

    fig.update_layout(
        title="5511 서울대입구역3번출구 혼잡도 시계열 + CQR 예측구간 + CPD",
        template="plotly_white",
        legend={"orientation": "h"},
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
    )
    fig.update_yaxes(title_text="혼잡도(재차인원)", secondary_y=False)
    fig.update_yaxes(title_text="SDAR 점수", secondary_y=True)
    return fig
