from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import psycopg

from .scorer import get_default_weights


def _connect(db_url: str) -> psycopg.Connection:
    return psycopg.connect(db_url)


def _coerce_scalar_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, pd.Series):
        cleaned = value.dropna()
        if cleaned.empty:
            return default
        value = cleaned.iloc[0]

    if value is None or pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)

    if isinstance(value, (float, int)):
        if pd.isna(value) or value == float('inf') or value == float('-inf'):
            return None
        return value

    if value is None or (isinstance(value, str) and value.lower() == 'nan'):
        return None

    return value


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key in row.index:
        return _coerce_scalar_float(row[key], default)

    tuple_matches = [idx for idx in row.index if isinstance(idx, tuple) and len(idx) > 0 and idx[0] == key]
    if tuple_matches:
        return _coerce_scalar_float(row[tuple_matches[0]], default)

    return default


def init_db(db_url: str) -> None:
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_history_daily (
                    code TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    open_price DOUBLE PRECISION,
                    high_price DOUBLE PRECISION,
                    low_price DOUBLE PRECISION,
                    close_price DOUBLE PRECISION,
                    adj_close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (code, trade_date)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    captured_at TIMESTAMPTZ NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    last_price DOUBLE PRECISION NOT NULL,
                    feature_json JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id BIGSERIAL PRIMARY KEY,
                    predicted_at TIMESTAMPTZ NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    predicted_price DOUBLE PRECISION NOT NULL,
                    score DOUBLE PRECISION NOT NULL,
                    signal TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    factor_json JSONB NOT NULL,
                    evaluated_at TIMESTAMPTZ,
                    realized_return DOUBLE PRECISION,
                    is_correct BOOLEAN
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_weights (
                    factor_name TEXT PRIMARY KEY,
                    weight DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()


def bootstrap_factor_weights(db_url: str) -> None:
    defaults = get_default_weights()
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            for factor, weight in defaults.items():
                cur.execute(
                    """
                    INSERT INTO factor_weights (factor_name, weight)
                    VALUES (%s, %s)
                    ON CONFLICT (factor_name) DO NOTHING
                    """,
                    (factor, weight),
                )
        conn.commit()


def load_factor_weights(db_url: str) -> dict[str, float]:
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT factor_name, weight FROM factor_weights')
            rows = cur.fetchall()
    return {str(name): float(weight) for name, weight in rows}


def upsert_factor_weights(db_url: str, weights: dict[str, float]) -> None:
    now = datetime.utcnow()
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            for factor, weight in weights.items():
                cur.execute(
                    """
                    INSERT INTO factor_weights (factor_name, weight, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (factor_name)
                    DO UPDATE SET weight = EXCLUDED.weight, updated_at = EXCLUDED.updated_at
                    """,
                    (factor, float(weight), now),
                )
        conn.commit()


def save_snapshot_and_predictions(
    db_url: str,
    snapshot: pd.DataFrame,
    scored: pd.DataFrame,
    horizon_minutes: int,
) -> None:
    merged = scored.copy()

    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            for _, row in merged.iterrows():
                captured_at = pd.Timestamp(row['timestamp']).to_pydatetime()

                feature_payload = {
                    'ret_5m': _coerce_scalar_float(row.get('ret_5m')),
                    'ret_1d': _coerce_scalar_float(row.get('ret_1d')),
                    'volume_ratio': _coerce_scalar_float(row.get('volume_ratio')),
                    'volatility_2h': _coerce_scalar_float(row.get('volatility_2h')),
                    'rsi_14': _coerce_scalar_float(row.get('rsi_14')),
                    'roe': _coerce_scalar_float(row.get('roe')),
                    'trailing_pe': _coerce_scalar_float(row.get('trailing_pe')),
                    'price_to_book': _coerce_scalar_float(row.get('price_to_book')),
                    'dividend_yield': _coerce_scalar_float(row.get('dividend_yield')),
                    'news_sentiment': _coerce_scalar_float(row.get('news_sentiment')),
                    'news_article_count': _coerce_scalar_float(row.get('news_article_count')),
                }

                cur.execute(
                    """
                    INSERT INTO technical_snapshots (captured_at, code, name, last_price, feature_json)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        captured_at,
                        str(row['code']),
                        str(row['name']),
                        _coerce_scalar_float(row.get('last_price')),
                        json.dumps(_json_safe(feature_payload), ensure_ascii=False, allow_nan=False),
                    ),
                )

                factor_payload: dict[str, dict[str, float]] = {}
                for factor in get_default_weights().keys():
                    factor_payload[factor] = {
                        'raw': _coerce_scalar_float(row.get(factor)),
                        'z': _coerce_scalar_float(row.get(f'z__{factor}')),
                        'contrib': _coerce_scalar_float(row.get(f'contrib__{factor}')),
                    }

                cur.execute(
                    """
                    INSERT INTO model_predictions (
                        predicted_at,
                        code,
                        name,
                        predicted_price,
                        score,
                        signal,
                        horizon_minutes,
                        factor_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        captured_at,
                        str(row['code']),
                        str(row['name']),
                        _coerce_scalar_float(row.get('last_price')),
                        _coerce_scalar_float(row.get('score')),
                        str(row['signal']),
                        int(horizon_minutes),
                        json.dumps(_json_safe(factor_payload), ensure_ascii=False, allow_nan=False),
                    ),
                )
        conn.commit()


def fetch_unevaluated_predictions(db_url: str, horizon_minutes: int) -> list[dict[str, object]]:
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, predicted_at, code, signal, predicted_price, factor_json
                FROM model_predictions
                WHERE evaluated_at IS NULL
                  AND predicted_at <= NOW() - (%s || ' minutes')::interval
                  AND horizon_minutes = %s
                ORDER BY predicted_at ASC
                """,
                (horizon_minutes, horizon_minutes),
            )
            rows = cur.fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        results.append(
            {
                'id': int(row[0]),
                'predicted_at': row[1],
                'code': str(row[2]),
                'signal': str(row[3]),
                'predicted_price': float(row[4]),
                'factor_json': row[5],
            }
        )
    return results


def mark_prediction_evaluated(
    db_url: str,
    prediction_id: int,
    realized_return: float,
    is_correct: bool,
) -> None:
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE model_predictions
                SET evaluated_at = NOW(),
                    realized_return = %s,
                    is_correct = %s
                WHERE id = %s
                """,
                (float(realized_return), bool(is_correct), int(prediction_id)),
            )
        conn.commit()


def save_daily_technical_history(db_url: str, code: str, history_df: pd.DataFrame) -> int:
    if history_df.empty:
        return 0

    inserted = 0
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            for idx, row in history_df.iterrows():
                trade_date = pd.Timestamp(idx).date()
                cur.execute(
                    """
                    INSERT INTO technical_history_daily (
                        code,
                        trade_date,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        adj_close,
                        volume
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (code, trade_date)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume
                    """,
                    (
                        code,
                        trade_date,
                        _row_float(row, 'Open'),
                        _row_float(row, 'High'),
                        _row_float(row, 'Low'),
                        _row_float(row, 'Close'),
                        _row_float(row, 'Adj Close', _row_float(row, 'Close')),
                        _row_float(row, 'Volume'),
                    ),
                )
                inserted += 1
        conn.commit()
    return inserted
