"""Monitoring and observability utilities."""

from .metrics import (
    active_generations,
    feedback_events_total,
    generation_duration_seconds,
    generation_requests_total,
    get_metrics,
    images_generated_total,
    is_metrics_available,
    record_feedback,
    record_quality_metrics,
    track_generation_time,
    track_generation_time_async,
    update_gpu_metrics,
    update_model_pool_metrics,
)
from .sentry import (
    capture_exception,
    capture_message,
    init_sentry,
    is_initialized,
    set_user,
)

__all__ = [
    # Sentry
    "init_sentry",
    "capture_exception",
    "capture_message",
    "set_user",
    "is_initialized",
    # Metrics
    "active_generations",
    "feedback_events_total",
    "generation_duration_seconds",
    "generation_requests_total",
    "get_metrics",
    "images_generated_total",
    "is_metrics_available",
    "record_feedback",
    "record_quality_metrics",
    "track_generation_time",
    "track_generation_time_async",
    "update_gpu_metrics",
    "update_model_pool_metrics",
]
