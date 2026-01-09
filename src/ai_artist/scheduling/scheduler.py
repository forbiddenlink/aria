"""Automated creation scheduling using APScheduler."""

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CreationScheduler:
    """Schedule automated art creation."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        logger.info("scheduler_started")

    def schedule_daily(
        self, hour: int, minute: int, job_func, timezone: str = "UTC"
    ):
        """Schedule daily creation."""
        trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        self.scheduler.add_job(job_func, trigger)
        logger.info("daily_schedule_added", hour=hour, minute=minute, tz=timezone)

    def schedule_weekly(
        self,
        days: list[str],
        hour: int,
        minute: int,
        job_func,
        timezone: str = "UTC",
    ):
        """Schedule weekly creation on specific days."""
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        day_numbers = [day_map[day.lower()] for day in days]

        trigger = CronTrigger(
            day_of_week=",".join(map(str, day_numbers)),
            hour=hour,
            minute=minute,
            timezone=timezone,
        )
        self.scheduler.add_job(job_func, trigger)
        logger.info("weekly_schedule_added", days=days, hour=hour, minute=minute)

    def schedule_cron(self, cron_expression: str, job_func, timezone: str = "UTC"):
        """Schedule using cron expression."""
        trigger = CronTrigger.from_crontab(cron_expression, timezone=timezone)
        self.scheduler.add_job(job_func, trigger)
        logger.info("cron_schedule_added", cron=cron_expression)

    def run_once(self, job_func, delay_seconds: int = 0):
        """Run job once after delay."""
        run_date = datetime.now().timestamp() + delay_seconds
        self.scheduler.add_job(
            job_func, "date", run_date=datetime.fromtimestamp(run_date)
        )
        logger.info("one_time_job_scheduled", delay=delay_seconds)

    def shutdown(self):
        """Shutdown scheduler."""
        self.scheduler.shutdown()
        logger.info("scheduler_shutdown")

