from __future__ import annotations

from collections import defaultdict

import yfinance as yf

from .db import (
    fetch_unevaluated_predictions,
    load_factor_weights,
    mark_prediction_evaluated,
    upsert_factor_weights,
)
from .scorer import get_default_weights


def _get_latest_price(code: str) -> float | None:
    ticker = yf.Ticker(f'{code}.T')
    hist = ticker.history(period='5d', interval='1d', auto_adjust=False)
    if hist.empty:
        return None
    return float(hist['Close'].iloc[-1])


def apply_learning_feedback(db_url: str, horizon_minutes: int, learning_rate: float) -> int:
    pending = fetch_unevaluated_predictions(db_url, horizon_minutes)
    if not pending:
        return 0

    default_weights = get_default_weights()
    current_weights = load_factor_weights(db_url)
    if not current_weights:
        current_weights = default_weights.copy()

    delta_by_factor: dict[str, float] = defaultdict(float)
    evaluated = 0

    for pred in pending:
        latest_price = _get_latest_price(str(pred['code']))
        if latest_price is None or float(pred['predicted_price']) == 0:
            continue

        realized_return = latest_price / float(pred['predicted_price']) - 1
        true_sign = 1 if realized_return >= 0 else -1
        pred_sign = 1 if str(pred['signal']) == '買い候補' else -1

        is_correct = true_sign == pred_sign
        mark_prediction_evaluated(db_url, int(pred['id']), float(realized_return), is_correct)
        evaluated += 1

        error = true_sign - pred_sign
        factor_json = pred['factor_json']
        if isinstance(factor_json, str):
            continue

        for factor in default_weights.keys():
            payload = factor_json.get(factor, {})
            z = float(payload.get('z', 0.0) or 0.0)
            delta_by_factor[factor] += learning_rate * error * z

    if evaluated == 0:
        return 0

    updated = current_weights.copy()
    for factor, base_weight in default_weights.items():
        candidate = float(updated.get(factor, base_weight)) + delta_by_factor.get(factor, 0.0)
        updated[factor] = max(0.01, min(0.40, candidate))

    total = sum(updated.values())
    if total > 0:
        for factor in updated:
            updated[factor] = updated[factor] / total

    upsert_factor_weights(db_url, updated)
    return evaluated
