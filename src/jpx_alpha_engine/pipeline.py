from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd

from .collector import collect_market_snapshot, load_universe
from .config import Settings
from .db import (
    bootstrap_factor_weights,
    estimate_reliability_percent,
    init_db,
    load_latest_trade_dates,
    load_factor_weights,
    load_score_reliability_stats,
    save_snapshot_and_predictions,
)
from .learning import apply_learning_feedback
from .qlib_context import init_qlib
from .scorer import build_recommendation_csv, score_snapshot


LOGGER = logging.getLogger(__name__)


def _load_failure_counts(path: str) -> dict[str, int]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_failure_counts(path: str, counts: dict[str, int]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)


def _prune_universe_csv(universe_path: str, excluded_codes: set[str]) -> int:
    if not excluded_codes:
        return 0

    df = pd.read_csv(universe_path, dtype=str)
    if 'code' not in df.columns:
        return 0

    before = len(df)
    df = df[~df['code'].isin(excluded_codes)].copy()
    removed = before - len(df)
    if removed > 0:
        df.to_csv(universe_path, index=False, encoding='utf-8-sig')
    return removed



def _attach_confidence(recommendation: pd.DataFrame, db_url: str) -> pd.DataFrame:
    if recommendation.empty:
        recommendation['信頼度'] = pd.Series(dtype='object')
        return recommendation

    stats = load_score_reliability_stats(db_url)
    with_conf = recommendation.copy()
    with_conf['信頼度'] = with_conf['スコア'].apply(
        lambda x: f"{estimate_reliability_percent(x, stats):.1f}%"
    )
    return with_conf

def run_pipeline(settings: Settings) -> pd.DataFrame:
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    settings.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    settings.failure_state_path.parent.mkdir(parents=True, exist_ok=True)

    init_db(settings.database_url)
    bootstrap_factor_weights(settings.database_url)

    learned_count = apply_learning_feedback(
        settings.database_url,
        settings.prediction_horizon_minutes,
        settings.learning_rate,
        max_predictions_per_run=settings.learning_max_predictions_per_run,
        suppress_yfinance_warnings=settings.suppress_yfinance_warnings,
    )
    if learned_count:
        LOGGER.info('Applied learning feedback for %s predictions', learned_count)

    current_weights = load_factor_weights(settings.database_url)

    init_qlib(settings.qlib_provider_uri)

    failure_counts = _load_failure_counts(str(settings.failure_state_path))
    universe = load_universe(str(settings.universe_path))

    active_universe = [
        u for u in universe if failure_counts.get(u.code, 0) < settings.failure_threshold
    ]
    if len(active_universe) < len(universe):
        LOGGER.info(
            'Skipped %s symbols already excluded by failure threshold',
            len(universe) - len(active_universe),
        )

    latest_daily_dates = load_latest_trade_dates(
        settings.database_url,
        [u.code for u in active_universe],
    )

    snapshot, success_codes, failed_codes = collect_market_snapshot(
        active_universe,
        suppress_yfinance_warnings=settings.suppress_yfinance_warnings,
        tdnet_cache_path=str(settings.tdnet_cache_path),
        tdnet_lookback_days=settings.tdnet_lookback_days,
        tdnet_max_items=settings.tdnet_max_items,
        db_url=settings.database_url,
        latest_daily_dates=latest_daily_dates,
    )

    for u in active_universe:
        if u.code in success_codes:
            failure_counts[u.code] = 0
        elif u.code in failed_codes:
            failure_counts[u.code] = failure_counts.get(u.code, 0) + 1

    _save_failure_counts(str(settings.failure_state_path), failure_counts)

    auto_excluded = {
        code for code, cnt in failure_counts.items() if cnt >= settings.failure_threshold
    }
    removed = _prune_universe_csv(str(settings.universe_path), auto_excluded)
    if removed > 0:
        LOGGER.warning(
            'Auto-excluded %s symbols from universe after %s consecutive failures',
            removed,
            settings.failure_threshold,
        )

    if snapshot.empty:
        LOGGER.warning('No snapshot rows collected. CSV update skipped.')
        return pd.DataFrame()

    scored = score_snapshot(snapshot, weight_overrides=current_weights)
    recommendation = build_recommendation_csv(scored, settings.top_k)
    recommendation = _attach_confidence(recommendation, settings.database_url)

    recommendation.to_csv(settings.output_csv_path, index=False, encoding='utf-8-sig')
    save_snapshot_and_predictions(
        settings.database_url,
        snapshot=snapshot,
        scored=scored,
        horizon_minutes=settings.prediction_horizon_minutes,
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = settings.snapshot_dir / f'snapshot_{ts}.csv'
    scored_path = settings.snapshot_dir / f'scored_{ts}.csv'
    snapshot.to_csv(snapshot_path, index=False, encoding='utf-8-sig')
    scored.to_csv(scored_path, index=False, encoding='utf-8-sig')

    LOGGER.info('Wrote recommendation CSV: %s', settings.output_csv_path)
    return recommendation




