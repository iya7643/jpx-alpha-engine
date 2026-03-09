from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import load_settings
from .pipeline import run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()

    def job() -> None:
        try:
            run_pipeline(settings)
        except Exception:
            LOGGER.exception('Pipeline execution failed')

    job()

    scheduler = BlockingScheduler(timezone=settings.timezone)
    scheduler.add_job(
        job,
        trigger=IntervalTrigger(minutes=settings.interval_minutes),
        id='jpx_alpha_engine_job',
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    LOGGER.info('Scheduler started: every %s minutes', settings.interval_minutes)
    scheduler.start()


if __name__ == '__main__':
    main()
