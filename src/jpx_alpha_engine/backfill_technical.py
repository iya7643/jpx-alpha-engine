from __future__ import annotations

import logging

import yfinance as yf

from .collector import load_universe
from .config import load_settings
from .db import init_db, save_daily_technical_history


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    init_db(settings.database_url)

    universe = load_universe(str(settings.universe_path))
    total_rows = 0

    for item in universe:
        history = yf.download(
            tickers=item.yf_ticker,
            period=f'{settings.backfill_years}y',
            interval='1d',
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if history.empty:
            LOGGER.warning('No historical data for %s', item.yf_ticker)
            continue

        inserted = save_daily_technical_history(settings.database_url, item.code, history)
        total_rows += inserted
        LOGGER.info('Saved %s rows for %s(%s)', inserted, item.name, item.code)

    LOGGER.info('Backfill completed. total_rows=%s', total_rows)


if __name__ == '__main__':
    main()
