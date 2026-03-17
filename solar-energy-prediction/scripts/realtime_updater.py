"""PM2-managed realtime updater for TimescaleDB forecast refreshes."""

from __future__ import annotations

import logging
import signal
import sys
import time

from app.services.prediction_service import generate_and_store_forecast
from app.services.timescaledb_service import timescaledb_service
from ml.utils.config import settings
from ml.utils.logging import setup_logging

setup_logging(settings.log_level)
LOGGER = logging.getLogger(__name__)
KEEP_RUNNING = True


def _handle_shutdown(signum: int, _: object) -> None:
    """Stop the updater gracefully when PM2 sends a termination signal."""

    global KEEP_RUNNING
    LOGGER.info("Received signal %s, stopping realtime updater", signum)
    KEEP_RUNNING = False


def main() -> int:
    """Continuously refresh forecasts and write them into TimescaleDB."""

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    LOGGER.info(
        "Starting PM2 realtime updater for lat=%s lon=%s every %s minutes",
        settings.default_latitude,
        settings.default_longitude,
        settings.scheduler_interval_minutes,
    )

    timescaledb_service.initialise()
    interval_seconds = settings.scheduler_interval_minutes * 60

    try:
        while KEEP_RUNNING:
            cycle_started_at = time.monotonic()
            try:
                result = generate_and_store_forecast(
                    latitude=settings.default_latitude,
                    longitude=settings.default_longitude,
                )
                LOGGER.info(
                    "Realtime update complete: rows_written=%s latest_prediction_kw=%.4f model=%s",
                    result["rows_written"],
                    result["latest_prediction_kw"],
                    result["model_name"],
                )
            except Exception:
                LOGGER.exception("Realtime update failed")

            elapsed = time.monotonic() - cycle_started_at
            sleep_for = max(interval_seconds - elapsed, 5)
            deadline = time.monotonic() + sleep_for
            while KEEP_RUNNING and time.monotonic() < deadline:
                time.sleep(1)
    finally:
        timescaledb_service.close()

    LOGGER.info("Realtime updater stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
