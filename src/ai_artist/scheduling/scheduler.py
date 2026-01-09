"""Automated scheduling for artwork creation."""

import asyncio
from collections.abc import Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CreationScheduler:
    """Schedule automated artwork creation."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.topics = [
            "nature landscape",
            "abstract art",
            "urban architecture",
            "serene ocean",
            "mountain vista",
            "forest scene",
            "space nebula",
            "sunset sky",
        ]
        self.current_topic_index = 0
        logger.info("scheduler_initialized")

    def get_next_topic(self) -> str:
        """Get next topic from rotation."""
        topic = self.topics[self.current_topic_index]
        self.current_topic_index = (self.current_topic_index + 1) % len(self.topics)
        logger.info("topic_selected", topic=topic, index=self.current_topic_index)
        return str(topic)

    def add_daily_job(
        self,
        job_func: Callable,
        hour: int = 9,
        minute: int = 0,
        job_id: str = "daily_creation",
    ):
        """Schedule daily artwork creation."""
        trigger = CronTrigger(hour=hour, minute=minute)
        self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            name=f"Daily creation at {hour:02d}:{minute:02d}",
            replace_existing=True,
        )
        logger.info(
            "daily_job_scheduled",
            job_id=job_id,
            hour=hour,
            minute=minute,
        )

    def add_interval_job(
        self,
        job_func: Callable,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        job_id: str = "interval_creation",
    ):
        """Schedule artwork creation at intervals."""
        trigger = IntervalTrigger(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            name=f"Interval creation every {hours}h {minutes}m {seconds}s",
            replace_existing=True,
        )
        logger.info(
            "interval_job_scheduled",
            job_id=job_id,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )

    def add_weekly_job(
        self,
        job_func: Callable,
        day_of_week: str = "mon",
        hour: int = 9,
        minute: int = 0,
        job_id: str = "weekly_creation",
    ):
        """Schedule weekly artwork creation."""
        trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            name=f"Weekly creation on {day_of_week} at {hour:02d}:{minute:02d}",
            replace_existing=True,
        )
        logger.info(
            "weekly_job_scheduled",
            job_id=job_id,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
        )

    def add_custom_cron_job(
        self,
        job_func: Callable,
        cron_expression: str,
        job_id: str = "custom_creation",
    ):
        """Schedule artwork creation with custom cron expression."""
        # Parse cron expression (minute hour day month day_of_week)
        parts = cron_expression.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts")

        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )
        self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            name=f"Custom cron: {cron_expression}",
            replace_existing=True,
        )
        logger.info(
            "custom_job_scheduled",
            job_id=job_id,
            cron=cron_expression,
        )

    def remove_job(self, job_id: str):
        """Remove a scheduled job."""
        try:
            self.scheduler.remove_job(job_id)
            logger.info("job_removed", job_id=job_id)
        except Exception as e:
            logger.error("job_removal_failed", job_id=job_id, error=str(e))

    def list_jobs(self) -> list[dict]:
        """List all scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            if next_run is None:
                # Try alternative attribute names
                next_run = getattr(job, "next_run", None)

            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run.isoformat() if next_run else None,
                    "trigger": str(job.trigger),
                }
            )
        return jobs

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("scheduler_started")

    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("scheduler_shutdown")

    async def run_once(self, job_func: Callable):
        """Run a job immediately once."""
        logger.info("running_job_once")
        try:
            await job_func()
            logger.info("job_completed")
        except Exception as e:
            logger.error("job_failed", error=str(e))


class ScheduledArtist:
    """Wrapper for scheduled artwork creation."""

    def __init__(self, artist):
        self.artist = artist
        self.scheduler = CreationScheduler()
        logger.info("scheduled_artist_initialized")

    async def create_with_rotation(self):
        """Create artwork with topic rotation."""
        topic = self.scheduler.get_next_topic()
        logger.info("scheduled_creation_started", topic=topic)
        try:
            await self.artist.create_artwork(theme=topic)
            logger.info("scheduled_creation_complete", topic=topic)
        except Exception as e:
            logger.error("scheduled_creation_failed", topic=topic, error=str(e))

    async def create_batch(self, count: int = 3):
        """Create multiple artworks in a batch."""
        logger.info("batch_creation_started", count=count)
        for i in range(count):
            try:
                await self.create_with_rotation()
                if i < count - 1:
                    # Small delay between creations
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error("batch_creation_item_failed", index=i, error=str(e))
        logger.info("batch_creation_complete", count=count)

    def schedule_daily(self, hour: int = 9, minute: int = 0):
        """Schedule daily creation."""
        self.scheduler.add_daily_job(
            self.create_with_rotation,
            hour=hour,
            minute=minute,
        )

    def schedule_interval(self, hours: int = 6):
        """Schedule creation at intervals."""
        self.scheduler.add_interval_job(
            self.create_with_rotation,
            hours=hours,
        )

    def schedule_batch_daily(self, count: int = 3, hour: int = 9, minute: int = 0):
        """Schedule daily batch creation."""

        async def batch_job():
            await self.create_batch(count=count)

        self.scheduler.add_daily_job(
            batch_job,
            hour=hour,
            minute=minute,
            job_id=f"daily_batch_{count}",
        )

    def start(self):
        """Start the scheduler."""
        self.scheduler.start()
        logger.info("scheduled_artist_started")

    def shutdown(self):
        """Shutdown the scheduler."""
        self.scheduler.shutdown()
        logger.info("scheduled_artist_shutdown")

    def list_jobs(self) -> list[dict]:
        """List all scheduled jobs."""
        jobs: list[dict] = self.scheduler.list_jobs()
        return jobs
