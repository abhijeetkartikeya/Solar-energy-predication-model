"""FastAPI routes for training, forecasting, and querying stored results."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    HealthResponse,
    PanelParameters,
    PredictRequest,
    PredictResponse,
    StoredForecastResponse,
    TrainRequest,
    TrainResponse,
)
from app.services.prediction_service import generate_and_store_forecast, list_stored_predictions
from app.services.timescaledb_service import timescaledb_service
from app.services.training_service import train_location_model
from ml.utils.config import settings

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Solar Forecasting"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the health of the API and database connection."""

    return HealthResponse(
        status="healthy",
        database_ok=timescaledb_service.health_check(),
        scheduler_interval_minutes=settings.scheduler_interval_minutes,
    )


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest) -> TrainResponse:
    """Train a model for the supplied coordinate using the last seven days of data."""

    try:
        result = train_location_model(
            latitude=request.latitude,
            longitude=request.longitude,
            panel_overrides=request.panel,
        )
        return TrainResponse(**result)
    except Exception as exc:
        LOGGER.exception("Training failed for lat=%s lon=%s", request.latitude, request.longitude)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate and persist a 7-day history plus 3-day ahead forecast."""

    try:
        result = generate_and_store_forecast(
            latitude=request.latitude,
            longitude=request.longitude,
            panel_overrides=request.panel,
            train_model=request.train_model,
        )
        return PredictResponse(**result)
    except Exception as exc:
        LOGGER.exception("Forecast generation failed for lat=%s lon=%s", request.latitude, request.longitude)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/predict", response_model=PredictResponse)
async def predict_get(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    train_model: bool = Query(default=False),
) -> PredictResponse:
    """GET shortcut for `/predict?lat=&lon=` as requested in the brief."""

    return await predict(
        PredictRequest(
            latitude=lat,
            longitude=lon,
            train_model=train_model,
            panel=PanelParameters(),
        )
    )


@router.get("/predictions", response_model=StoredForecastResponse)
async def get_predictions(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hours: int = Query(default=240, ge=1, le=24 * 30),
) -> StoredForecastResponse:
    """Read forecast rows for a coordinate from TimescaleDB."""

    rows = list_stored_predictions(latitude=lat, longitude=lon, hours=hours)
    return StoredForecastResponse(latitude=lat, longitude=lon, rows=rows)
