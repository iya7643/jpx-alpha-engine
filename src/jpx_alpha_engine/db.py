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
                        str(row['code']),
                        pd.Timestamp(row.get('daily_trade_date', row['timestamp'])).date(),
                        _coerce_scalar_float(row.get('daily_open')),
                        _coerce_scalar_float(row.get('daily_high')),
                        _coerce_scalar_float(row.get('daily_low')),
                        _coerce_scalar_float(row.get('daily_close', row.get('last_price'))),
                        _coerce_scalar_float(row.get('daily_adj_close', row.get('daily_close', row.get('last_price')))),
                        _coerce_scalar_float(row.get('daily_volume')),
                    ),
                )
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
                    'tdnet_sentiment': _coerce_scalar_float(row.get('tdnet_sentiment')),
                    'tdnet_disclosure_count': _coerce_scalar_float(row.get('tdnet_disclosure_count')),
                    'tdnet_positive_count': _coerce_scalar_float(row.get('tdnet_positive_count')),
                    'tdnet_negative_count': _coerce_scalar_float(row.get('tdnet_negative_count')),
                    'tdnet_earnings_revision_score': _coerce_scalar_float(row.get('tdnet_earnings_revision_score')),
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
                SELECT
                    mp.id,
                    mp.predicted_at,
                    mp.code,
                    mp.signal,
                    mp.predicted_price,
                    mp.factor_json,
                    ((mp.predicted_at + (%s || ' minutes')::interval) AT TIME ZONE 'Asia/Tokyo')::date AS target_trade_date,
                    th.trade_date AS eval_trade_date,
                    th.close_price AS eval_close_price
                FROM model_predictions mp
                LEFT JOIN LATERAL (
                    SELECT trade_date, close_price
                    FROM technical_history_daily
                    WHERE code = mp.code
                      AND trade_date >= ((mp.predicted_at + (%s || ' minutes')::interval) AT TIME ZONE 'Asia/Tokyo')::date
                    ORDER BY trade_date ASC
                    LIMIT 1
                ) th ON TRUE
                WHERE mp.evaluated_at IS NULL
                  AND mp.predicted_at <= NOW() - (%s || ' minutes')::interval
                  AND mp.horizon_minutes = %s
                ORDER BY mp.predicted_at ASC
                """,
                (horizon_minutes, horizon_minutes, horizon_minutes, horizon_minutes),
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
                'target_trade_date': row[6],
                'eval_trade_date': row[7],
                'eval_close_price': _coerce_scalar_float(row[8], default=float('nan')),
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


def load_score_reliability_stats(
    db_url: str,
    bucket_size: int = 50,
    min_samples: int = 10,
) -> dict[str, object]:
    bucket_size = max(1, int(bucket_size))
    min_samples = max(1, int(min_samples))

    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    CASE WHEN score >= 0 THEN 'buy' ELSE 'sell' END AS side,
                    FLOOR(ABS(score) / %s) * %s AS bucket_start,
                    COUNT(*) AS n,
                    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) AS win_rate
                FROM model_predictions
                WHERE is_correct IS NOT NULL
                GROUP BY 1, 2
                HAVING COUNT(*) >= %s
                """,
                (bucket_size, bucket_size, min_samples),
            )
            bucket_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    CASE WHEN score >= 0 THEN 'buy' ELSE 'sell' END AS side,
                    COUNT(*) AS n,
                    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) AS win_rate
                FROM model_predictions
                WHERE is_correct IS NOT NULL
                GROUP BY 1
                """
            )
            global_rows = cur.fetchall()

    buckets: dict[str, dict[int, float]] = {'buy': {}, 'sell': {}}
    for side, bucket_start, _, win_rate in bucket_rows:
        side_key = str(side)
        if side_key not in buckets:
            continue
        buckets[side_key][int(float(bucket_start))] = float(win_rate)

    global_rates: dict[str, float] = {'buy': 0.5, 'sell': 0.5}
    for side, _, win_rate in global_rows:
        side_key = str(side)
        if side_key in global_rates and win_rate is not None:
            global_rates[side_key] = float(win_rate)

    return {
        'bucket_size': bucket_size,
        'global_rates': global_rates,
        'buckets': buckets,
    }


def estimate_reliability_percent(score: object, stats: dict[str, object]) -> float:
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0

    side = 'buy' if score_f >= 0 else 'sell'
    bucket_size = int(stats.get('bucket_size', 50) or 50)
    bucket_start = int(abs(score_f) // bucket_size * bucket_size)

    buckets = stats.get('buckets', {})
    side_buckets = buckets.get(side, {}) if isinstance(buckets, dict) else {}
    if isinstance(side_buckets, dict) and bucket_start in side_buckets:
        return float(side_buckets[bucket_start]) * 100.0

    global_rates = stats.get('global_rates', {})
    if isinstance(global_rates, dict):
        return float(global_rates.get(side, 0.5)) * 100.0

    return 50.0

def load_latest_trade_dates(db_url: str, codes: list[str]) -> dict[str, object]:
    if not codes:
        return {}

    normalized = [str(c) for c in codes if str(c)]
    if not normalized:
        return {}

    sql = """
        SELECT DISTINCT ON (code)
            code,
            trade_date
        FROM technical_history_daily
        WHERE code = ANY(%s)
        ORDER BY code, trade_date DESC
    """

    latest_dates: dict[str, object] = {}
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (normalized,))
            for code, trade_date in cur.fetchall():
                if trade_date is None:
                    continue
                latest_dates[str(code)] = trade_date

    return latest_dates


def load_latest_close_prices(db_url: str, codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}

    normalized = [str(c) for c in codes if str(c)]
    if not normalized:
        return {}

    sql = """
        SELECT DISTINCT ON (code)
            code,
            close_price
        FROM technical_history_daily
        WHERE code = ANY(%s)
        ORDER BY code, trade_date DESC
    """

    prices: dict[str, float] = {}
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (normalized,))
            for code, close_price in cur.fetchall():
                if close_price is None:
                    continue
                prices[str(code)] = float(close_price)

    return prices




