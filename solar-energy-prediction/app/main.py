"""Application entrypoint for the solar forecasting system."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.prediction_service import generate_and_store_forecast
from app.services.timescaledb_service import timescaledb_service
from ml.utils.config import settings
from ml.utils.logging import setup_logging

setup_logging(settings.log_level)
LOGGER = logging.getLogger(__name__)
SCHEDULER = BackgroundScheduler(timezone="UTC")


def _run_default_location_forecast() -> None:
    """Execute the scheduled forecast update for the default coordinate."""

    generate_and_store_forecast(
        latitude=settings.default_latitude,
        longitude=settings.default_longitude,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialise storage and background jobs during app startup."""

    timescaledb_service.initialise()
    if settings.enable_internal_scheduler:
        try:
            _run_default_location_forecast()
        except Exception:
            LOGGER.exception("Initial forecast generation failed during startup")

        SCHEDULER.add_job(
            _run_default_location_forecast,
            trigger="interval",
            minutes=settings.scheduler_interval_minutes,
            id="default-location-forecast",
            replace_existing=True,
        )
        SCHEDULER.start()
        LOGGER.info("Scheduler started with %s-minute interval", settings.scheduler_interval_minutes)
    else:
        LOGGER.info("Internal scheduler disabled; expecting an external worker such as PM2")

    yield

    if SCHEDULER.running:
        SCHEDULER.shutdown(wait=False)
    timescaledb_service.close()


app = FastAPI(
    title="Solar Energy Output Forecasting System",
    description="15-minute solar power forecasting with Open-Meteo, optional VEDAS enrichment, TimescaleDB, and Grafana.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root() -> FileResponse:
    """Serve the built-in solar forecasting dashboard."""

    return FileResponse("app/static/index.html")
