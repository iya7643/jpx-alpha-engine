from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    universe_path: Path
    output_csv_path: Path
    snapshot_dir: Path
    top_k: int
    schedule_hour: int
    schedule_minute: int
    run_on_start: bool
    timezone: str
    qlib_provider_uri: str | None
    database_url: str
    prediction_horizon_minutes: int
    learning_rate: float
    backfill_years: int
    failure_state_path: Path
    failure_threshold: int
    suppress_yfinance_warnings: bool
    tdnet_cache_path: Path
    tdnet_lookback_days: int
    tdnet_max_items: int
    learning_max_predictions_per_run: int


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]

    universe_rel = os.getenv('JPX_UNIVERSE_PATH', 'data/universe_jp.csv')
    output_rel = os.getenv('JPX_OUTPUT_CSV', 'output/recommendations.csv')
    snapshot_rel = os.getenv('JPX_SNAPSHOT_DIR', 'data/snapshots')
    failure_state_rel = os.getenv('JPX_FAILURE_STATE_PATH', 'data/universe_failures.json')
    tdnet_cache_rel = os.getenv('JPX_TDNET_CACHE_PATH', 'data/tdnet_cache.json')

    provider = os.getenv('QLIB_PROVIDER_URI', '').strip() or None
    db_url = os.getenv('JPX_DATABASE_URL', 'postgresql://jpx:jpxpass@localhost:5432/jpx_alpha')

    return Settings(
        project_root=project_root,
        universe_path=(project_root / universe_rel).resolve(),
        output_csv_path=(project_root / output_rel).resolve(),
        snapshot_dir=(project_root / snapshot_rel).resolve(),
        top_k=_env_int('JPX_TOP_K', 10),
        schedule_hour=_env_int('JPX_SCHEDULE_HOUR', 16),
        schedule_minute=_env_int('JPX_SCHEDULE_MINUTE', 30),
        run_on_start=_env_bool('JPX_RUN_ON_START', True),
        timezone=os.getenv('JPX_TIMEZONE', 'Asia/Tokyo'),
        qlib_provider_uri=provider,
        database_url=db_url,
        prediction_horizon_minutes=_env_int('JPX_PREDICTION_HORIZON_MINUTES', 1440),
        learning_rate=_env_float('JPX_LEARNING_RATE', 0.01),
        backfill_years=_env_int('JPX_BACKFILL_YEARS', 5),
        failure_state_path=(project_root / failure_state_rel).resolve(),
        failure_threshold=max(1, _env_int('JPX_FAILURE_THRESHOLD', 3)),
        suppress_yfinance_warnings=_env_bool('JPX_SUPPRESS_YF_WARNINGS', True),
        tdnet_cache_path=(project_root / tdnet_cache_rel).resolve(),
        tdnet_lookback_days=max(1, _env_int('JPX_TDNET_LOOKBACK_DAYS', 7)),
        tdnet_max_items=max(100, _env_int('JPX_TDNET_MAX_ITEMS', 1200)),
        learning_max_predictions_per_run=max(50, _env_int('JPX_LEARNING_MAX_PREDICTIONS_PER_RUN', 500)),
    )
