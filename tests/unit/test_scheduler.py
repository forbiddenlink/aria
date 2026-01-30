"""Tests for scheduling system."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from ai_artist.scheduling.scheduler import CreationScheduler, ScheduledArtist


@pytest.fixture
def scheduler():
    """Create a test scheduler."""
    return CreationScheduler()


@pytest.fixture
def mock_artist():
    """Create a mock artist."""
    artist = MagicMock()
    artist.create_artwork = AsyncMock()
    return artist


@pytest.fixture
def scheduled_artist(mock_artist):
    """Create a scheduled artist."""
    return ScheduledArtist(mock_artist)


class TestCreationScheduler:
    """Test CreationScheduler class."""

    def test_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.scheduler is not None
        assert len(scheduler.topics) == 8
        assert scheduler.current_topic_index == 0

    def test_get_next_topic(self, scheduler):
        """Test topic rotation."""
        first_topic = scheduler.get_next_topic()
        assert first_topic == "nature landscape"
        assert scheduler.current_topic_index == 1

        second_topic = scheduler.get_next_topic()
        assert second_topic == "abstract art"
        assert scheduler.current_topic_index == 2

    def test_topic_wraps_around(self, scheduler):
        """Test topic rotation wraps around."""
        # Get all topics
        for _ in range(8):
            scheduler.get_next_topic()

        # Should wrap to first topic
        assert scheduler.current_topic_index == 0
        topic = scheduler.get_next_topic()
        assert topic == "nature landscape"

    def test_add_daily_job(self, scheduler):
        """Test adding daily job."""
        job_func = AsyncMock()
        scheduler.add_daily_job(job_func, hour=9, minute=30, job_id="test_daily")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "test_daily"
        assert "09:30" in jobs[0]["name"]

    def test_add_interval_job(self, scheduler):
        """Test adding interval job."""
        job_func = AsyncMock()
        scheduler.add_interval_job(
            job_func, hours=2, minutes=30, job_id="test_interval"
        )

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "test_interval"

    def test_add_weekly_job(self, scheduler):
        """Test adding weekly job."""
        job_func = AsyncMock()
        scheduler.add_weekly_job(
            job_func, day_of_week="mon", hour=10, minute=0, job_id="test_weekly"
        )

        jobs = scheduler.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "test_weekly"

    def test_remove_job(self, scheduler):
        """Test removing a job."""
        job_func = AsyncMock()
        scheduler.add_daily_job(job_func, job_id="removable")
        assert len(scheduler.list_jobs()) == 1

        scheduler.remove_job("removable")
        assert len(scheduler.list_jobs()) == 0

    def test_list_jobs_empty(self, scheduler):
        """Test listing jobs when none exist."""
        jobs = scheduler.list_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_run_once(self, scheduler):
        """Test running a job once."""
        job_func = AsyncMock()
        await scheduler.run_once(job_func)
        job_func.assert_called_once()


class TestScheduledArtist:
    """Test ScheduledArtist class."""

    def test_initialization(self, scheduled_artist, mock_artist):
        """Test scheduled artist initialization."""
        assert scheduled_artist.artist == mock_artist
        assert scheduled_artist.scheduler is not None

    @pytest.mark.asyncio
    async def test_create_with_rotation(self, scheduled_artist, mock_artist):
        """Test creation with topic rotation."""
        await scheduled_artist.create_with_rotation()

        mock_artist.create_artwork.assert_called_once()
        call_args = mock_artist.create_artwork.call_args
        assert call_args.kwargs["theme"] == "nature landscape"

    @pytest.mark.asyncio
    async def test_create_batch(self, scheduled_artist, mock_artist):
        """Test batch creation."""
        await scheduled_artist.create_batch(count=3)

        assert mock_artist.create_artwork.call_count == 3

    @pytest.mark.asyncio
    async def test_create_batch_with_error(self, scheduled_artist, mock_artist):
        """Test batch creation continues after error."""
        # First call succeeds, second fails, third succeeds
        mock_artist.create_artwork.side_effect = [
            None,
            Exception("Test error"),
            None,
        ]

        await scheduled_artist.create_batch(count=3)

        # Should attempt all 3 despite error
        assert mock_artist.create_artwork.call_count == 3

    def test_schedule_daily(self, scheduled_artist):
        """Test scheduling daily creation."""
        scheduled_artist.schedule_daily(hour=10, minute=30)

        jobs = scheduled_artist.list_jobs()
        assert len(jobs) == 1
        assert "10:30" in jobs[0]["name"]

    def test_schedule_interval(self, scheduled_artist):
        """Test scheduling interval creation."""
        scheduled_artist.schedule_interval(hours=6)

        jobs = scheduled_artist.list_jobs()
        assert len(jobs) == 1
        assert "6h" in jobs[0]["name"]

    def test_schedule_batch_daily(self, scheduled_artist):
        """Test scheduling daily batch creation."""
        scheduled_artist.schedule_batch_daily(count=3, hour=9, minute=0)

        jobs = scheduled_artist.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "daily_batch_3"

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, scheduled_artist):
        """Test starting and shutting down scheduler."""
        import asyncio

        # Start scheduler in background
        scheduled_artist.start()
        await asyncio.sleep(0.1)  # Let scheduler start
        assert scheduled_artist.scheduler.scheduler.running

        scheduled_artist.shutdown()
        await asyncio.sleep(0.2)  # Allow time for shutdown
        assert not scheduled_artist.scheduler.scheduler.running
