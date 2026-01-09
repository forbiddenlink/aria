#!/usr/bin/env python
"""CLI tool for managing scheduled artwork creation."""

import argparse
import asyncio
import sys
from pathlib import Path

from ai_artist.main import AIArtist
from ai_artist.scheduling.scheduler import ScheduledArtist
from ai_artist.utils.config import load_config
from ai_artist.utils.logging import get_logger

logger = get_logger(__name__)


async def start_scheduler(args):
    """Start the scheduler with configured jobs."""
    config = load_config(Path("config/config.yaml"))
    artist = AIArtist(config)
    scheduled_artist = ScheduledArtist(artist)

    # Set up schedules based on args
    if args.daily:
        hour, minute = map(int, args.daily.split(":"))
        scheduled_artist.schedule_daily(hour=hour, minute=minute)
        print(f"✓ Scheduled daily creation at {hour:02d}:{minute:02d}")

    if args.interval:
        scheduled_artist.schedule_interval(hours=args.interval)
        print(f"✓ Scheduled creation every {args.interval} hours")

    if args.batch:
        count, time_str = args.batch.split("@")
        count = int(count)
        hour, minute = map(int, time_str.split(":"))
        scheduled_artist.schedule_batch_daily(count=count, hour=hour, minute=minute)
        print(f"✓ Scheduled daily batch of {count} artworks at {hour:02d}:{minute:02d}")

    # Start the scheduler
    scheduled_artist.start()
    print("\n🎨 AI Artist scheduler is running...")
    print("Press Ctrl+C to stop\n")

    # List scheduled jobs
    jobs = scheduled_artist.list_jobs()
    if jobs:
        print("Scheduled jobs:")
        for job in jobs:
            print(f"  • {job['name']}")
            print(f"    Next run: {job['next_run']}")
            print()

    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down scheduler...")
        scheduled_artist.shutdown()
        print("✓ Scheduler stopped")


async def run_now(args):
    """Run artwork creation immediately."""
    config = load_config(Path("config/config.yaml"))
    artist = AIArtist(config)

    if args.batch_size:
        scheduled_artist = ScheduledArtist(artist)
        await scheduled_artist.create_batch(count=args.batch_size)
    else:
        await artist.create_artwork(theme=args.theme)


async def list_jobs(args):
    """List all scheduled jobs."""
    config = load_config(Path("config/config.yaml"))
    artist = AIArtist(config)
    scheduled_artist = ScheduledArtist(artist)

    jobs = scheduled_artist.list_jobs()
    if not jobs:
        print("No scheduled jobs")
        return

    print(f"\nScheduled jobs ({len(jobs)}):\n")
    for job in jobs:
        print(f"ID: {job['id']}")
        print(f"Name: {job['name']}")
        print(f"Next run: {job['next_run']}")
        print(f"Trigger: {job['trigger']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Artist Scheduler - Automated artwork creation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start scheduler
    start_parser = subparsers.add_parser("start", help="Start the scheduler")
    start_parser.add_argument(
        "--daily",
        type=str,
        metavar="HH:MM",
        help="Schedule daily creation at specified time (e.g., 09:00)",
    )
    start_parser.add_argument(
        "--interval",
        type=int,
        metavar="HOURS",
        help="Schedule creation every N hours",
    )
    start_parser.add_argument(
        "--batch",
        type=str,
        metavar="COUNT@HH:MM",
        help="Schedule daily batch (e.g., 3@09:00 for 3 artworks at 9am)",
    )

    # Run now
    run_parser = subparsers.add_parser("run", help="Create artwork now")
    run_parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Theme for artwork",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Create multiple artworks",
    )

    # List jobs
    subparsers.add_parser("list", help="List scheduled jobs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "start":
            asyncio.run(start_scheduler(args))
        elif args.command == "run":
            asyncio.run(run_now(args))
        elif args.command == "list":
            asyncio.run(list_jobs(args))
    except Exception as e:
        logger.error("command_failed", command=args.command, error=str(e))
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
