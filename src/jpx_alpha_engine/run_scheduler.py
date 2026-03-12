from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

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

    scheduler = BlockingScheduler(timezone=settings.timezone)

    cron_trigger = CronTrigger(
        hour=settings.schedule_hour,
        minute=settings.schedule_minute,
        timezone=settings.timezone,
    )
    scheduled_job = scheduler.add_job(
        job,
        trigger=cron_trigger,
        id='jpx_alpha_engine_daily_job',
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    if settings.run_on_start:
        scheduler.add_job(
            job,
            trigger=DateTrigger(run_date=datetime.now()),
            id='jpx_alpha_engine_startup_job',
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        LOGGER.info('Startup job is enabled')

    next_run_time = getattr(scheduled_job, 'next_run_time', None)
    LOGGER.info(
        'Scheduler started: daily at %02d:%02d (%s), next_run=%s',
        settings.schedule_hour,
        settings.schedule_minute,
        settings.timezone,
        next_run_time,
    )
    scheduler.start()


if __name__ == '__main__':
    main()
