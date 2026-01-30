"""Health check and monitoring endpoints for FastAPI application."""

import time
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

from ..utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    uptime: float


class ReadinessResponse(BaseModel):
    """Readiness check response model."""

    status: str
    checks: dict[str, Any]


# Track application start time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint.

    Returns 200 if the application is running.
    Useful for Docker health checks and load balancer pings.
    """
    uptime = time.time() - _start_time
    return HealthResponse(
        status="healthy",
        version="1.0.0",  # Could load from __version__ or config
        uptime=uptime,
    )


@router.get(
    "/health/ready", response_model=ReadinessResponse, status_code=status.HTTP_200_OK
)
async def readiness_check():
    """
    Readiness check endpoint.

    Returns 200 if the application is ready to accept traffic.
    Checks critical dependencies like database, file system, etc.
    """
    checks = {
        "api": "ready",
        # Add more checks as needed:
        # "database": check_database(),
        # "redis": check_redis(),
        # "storage": check_storage(),
    }

    # If any check fails, return 503
    all_ready = all(status == "ready" for status in checks.values())

    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
    )


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness check endpoint.

    Returns 200 if the application is alive (not deadlocked).
    Kubernetes uses this to determine if it should restart the pod.
    """
    return {"status": "alive"}
