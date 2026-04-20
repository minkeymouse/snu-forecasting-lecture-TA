from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from app.config import Settings


def _ensure_database_paths(settings: Settings) -> tuple[Path, Path]:
    db_dir = settings.database_dir
    raw_dir = db_dir / settings.raw_cache_dirname
    db_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return db_dir, raw_dir


def _daily_cache_file(settings: Settings, stdr_de: str) -> Path:
    _, raw_dir = _ensure_database_paths(settings)
    return raw_dir / f"{stdr_de}.json"


def _history_cache_file(settings: Settings) -> Path:
    db_dir, _ = _ensure_database_paths(settings)
    return db_dir / settings.target_cache_file


def _load_daily_match_from_cache(settings: Settings, stdr_de: str) -> dict[str, Any] | None | bool:
    cache_path = _daily_cache_file(settings, stdr_de)
    if not cache_path.exists():
        return False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    # 캐시 파일이 존재하면 네트워크 재호출을 피하기 위해 None(미발견)도 유효 결과로 취급.
    return payload.get("matched_row")


def _save_daily_match_to_cache(
    settings: Settings,
    stdr_de: str,
    matched_row: dict[str, Any] | None,
) -> None:
    cache_path = _daily_cache_file(settings, stdr_de)
    payload = {"matched_row": matched_row}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _request_rows(
    settings: Settings,
    stdr_de: str,
    start_row: int,
    row_cnt: int,
) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "apikey": settings.api_key,
            "stdrDe": stdr_de,
            "startRow": start_row,
            "rowCnt": row_cnt,
        }
    )
    url = f"{settings.api_base_url}?{query}"
    try:
        with urlopen(url, timeout=settings.request_timeout_sec) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return payload
    return []


def _target_match(row: dict[str, Any], settings: Settings) -> bool:
    return (
        str(row.get("nodeNm", "")).strip() == settings.target_node_name
        and str(row.get("routeNm", "")).strip() == settings.target_route_name
    )


def find_target_row_for_day(
    settings: Settings,
    stdr_de: str,
    full_scan: bool = False,
) -> dict[str, Any] | None:
    cached_match = _load_daily_match_from_cache(settings, stdr_de)
    if cached_match is not False:
        if isinstance(cached_match, dict):
            return cached_match
        return None

    quick_rows = _request_rows(
        settings=settings,
        stdr_de=stdr_de,
        start_row=settings.quick_start_row,
        row_cnt=settings.quick_row_cnt,
    )
    for row in quick_rows:
        if _target_match(row, settings):
            _save_daily_match_to_cache(settings, stdr_de, row)
            return row

    if not full_scan:
        _save_daily_match_to_cache(settings, stdr_de, None)
        return None

    for page in range(1, settings.max_pages_per_day + 1):
        start_row = (page - 1) * settings.rows_per_page + 1
        rows = _request_rows(
            settings=settings,
            stdr_de=stdr_de,
            start_row=start_row,
            row_cnt=settings.rows_per_page,
        )
        for row in rows:
            if _target_match(row, settings):
                _save_daily_match_to_cache(settings, stdr_de, row)
                return row
    _save_daily_match_to_cache(settings, stdr_de, None)
    return None


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    for col in ("nownmprNmpr", "tkcarNmpr", "gffNmpr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "stdrDe" in df.columns:
        df["ds"] = pd.to_datetime(df["stdrDe"], format="%Y%m%d", errors="coerce")
    if "y" not in df.columns and "nownmprNmpr" in df.columns:
        df["y"] = df["nownmprNmpr"]
    keep_cols = [
        "ds",
        "y",
        "tkcarNmpr",
        "gffNmpr",
        "routeNm",
        "nodeNm",
        "sttnNo",
        "nodeId",
        "tourTime",
        "avrgVe",
        "stdrDe",
    ]
    cols = [c for c in keep_cols if c in df.columns]
    return df[cols].dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)


def _load_history_cache(settings: Settings) -> pd.DataFrame:
    path = _history_cache_file(settings)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return _normalize_history_df(df)


def _save_history_cache(settings: Settings, df: pd.DataFrame) -> None:
    path = _history_cache_file(settings)
    payload = df.copy()
    payload["ds"] = payload["ds"].dt.strftime("%Y-%m-%d")
    payload.to_csv(path, index=False)


def fetch_target_history(
    settings: Settings,
    days: int = 60,
    full_scan: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    history_cache = _load_history_cache(settings)
    if not refresh:
        if history_cache.empty:
            return pd.DataFrame(columns=["ds", "y", "tkcarNmpr", "gffNmpr", "routeNm", "nodeNm"])
        cutoff = pd.Timestamp(date.today() - timedelta(days=max(days - 1, 0)))
        cached = history_cache[history_cache["ds"] >= cutoff].copy()
        return cached.sort_values("ds").reset_index(drop=True)

    if not settings.api_key:
        raise ValueError("SEOUL_TRAFFIC_API_KEY is missing.")

    history_by_date: dict[str, dict[str, Any]] = {}
    if not history_cache.empty:
        for _, row in history_cache.iterrows():
            key = row["ds"].strftime("%Y%m%d")
            history_by_date[key] = row.to_dict()
    rows: list[dict[str, Any]] = []
    today = date.today()
    for offset in range(days):
        current_date = today - timedelta(days=offset)
        stdr_de = current_date.strftime("%Y%m%d")
        cached = history_by_date.get(stdr_de)
        if cached:
            rows.append(cached)
            continue
        row = find_target_row_for_day(settings=settings, stdr_de=stdr_de, full_scan=full_scan)
        if row is None:
            continue
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ds", "y", "tkcarNmpr", "gffNmpr", "routeNm", "nodeNm"])

    df = _normalize_history_df(pd.DataFrame(rows))
    _save_history_cache(settings, df)
    return df


def get_runtime_config(settings: Settings) -> dict[str, Any]:
    runtime = asdict(settings)
    runtime["database_dir"] = str(settings.database_dir)
    return runtime
