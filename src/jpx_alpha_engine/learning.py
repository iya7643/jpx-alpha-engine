from __future__ import annotations

import logging
from collections import defaultdict

import math

from .db import (
    fetch_unevaluated_predictions,
    load_factor_weights,
    mark_prediction_evaluated,
    upsert_factor_weights,
)
from .scorer import get_default_weights


LOGGER = logging.getLogger(__name__)


def apply_learning_feedback(
    db_url: str,
    horizon_minutes: int,
    learning_rate: float,
    max_predictions_per_run: int = 500,
    suppress_yfinance_warnings: bool = True,
) -> int:
    _ = suppress_yfinance_warnings  # kept for backward compatibility

    pending = fetch_unevaluated_predictions(db_url, horizon_minutes)
    if not pending:
        return 0

    if max_predictions_per_run > 0 and len(pending) > max_predictions_per_run:
        LOGGER.info(
            'Learning pending=%s; evaluate only first %s this run',
            len(pending),
            max_predictions_per_run,
        )
        pending = pending[:max_predictions_per_run]
    else:
        LOGGER.info('Learning pending=%s', len(pending))

    default_weights = get_default_weights()
    current_weights = load_factor_weights(db_url)
    if not current_weights:
        current_weights = default_weights.copy()

    delta_by_factor: dict[str, float] = defaultdict(float)
    evaluated = 0
    skipped_no_price = 0

    for idx, pred in enumerate(pending, start=1):
        eval_price = pred.get('eval_close_price')
        try:
            latest_price = float(eval_price)
        except (TypeError, ValueError):
            latest_price = float('nan')

        if math.isnan(latest_price) or float(pred['predicted_price']) == 0:
            skipped_no_price += 1
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

        if idx % 200 == 0 or idx == len(pending):
            LOGGER.info(
                'Learning progress: %s/%s (evaluated=%s, skipped_no_price=%s)',
                idx,
                len(pending),
                evaluated,
                skipped_no_price,
            )

    if evaluated == 0:
        LOGGER.warning('Learning finished with 0 evaluated (skipped_no_price=%s)', skipped_no_price)
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
    LOGGER.info('Learning completed: evaluated=%s, skipped_no_price=%s', evaluated, skipped_no_price)
    return evaluated
