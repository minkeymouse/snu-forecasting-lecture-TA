from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_base_url: str = "https://t-data.seoul.go.kr/apig/apiman-gateway/tapi/BisTbisStSttnOprat/1.0"
    target_node_name: str = "서울대입구역3번출구"
    target_route_name: str = "5511"
    quick_start_row: int = 36000
    quick_row_cnt: int = 8000
    rows_per_page: int = 3000
    max_pages_per_day: int = 30
    request_timeout_sec: int = 5
    database_dir: Path = Path("database")
    raw_cache_dirname: str = "raw_daily"
    target_cache_file: str = "target_history.csv"


def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("SEOUL_TRAFFIC_API_KEY", "").strip().strip('"').strip("'"),
        api_base_url=os.getenv(
            "SEOUL_TRAFFIC_BASE_URL",
            "https://t-data.seoul.go.kr/apig/apiman-gateway/tapi/BisTbisStSttnOprat/1.0",
        ),
        target_node_name=os.getenv("TARGET_NODE_NAME", "서울대입구역3번출구"),
        target_route_name=os.getenv("TARGET_ROUTE_NAME", "5511"),
        quick_start_row=int(os.getenv("SEOUL_TRAFFIC_QUICK_START_ROW", "36000")),
        quick_row_cnt=int(os.getenv("SEOUL_TRAFFIC_QUICK_ROW_CNT", "8000")),
        rows_per_page=int(os.getenv("SEOUL_TRAFFIC_ROWS_PER_PAGE", "3000")),
        max_pages_per_day=int(os.getenv("SEOUL_TRAFFIC_MAX_PAGES_PER_DAY", "30")),
        request_timeout_sec=int(os.getenv("SEOUL_TRAFFIC_TIMEOUT_SEC", "5")),
        database_dir=Path(os.getenv("DATABASE_DIR", "database")),
        raw_cache_dirname=os.getenv("RAW_CACHE_DIRNAME", "raw_daily"),
        target_cache_file=os.getenv("TARGET_CACHE_FILE", "target_history.csv"),
    )
