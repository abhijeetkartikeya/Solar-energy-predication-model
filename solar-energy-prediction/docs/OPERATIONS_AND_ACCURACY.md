# Operations And Accuracy Guide

## What is accurate today

The system currently has two forecasting modes:

1. Formula baseline
   - Uses irradiance, panel area, panel efficiency, inverter efficiency, and temperature derating.
   - This is the production-safe fallback and is always available.

2. Optional ML model
   - Trained per coordinate using the latest historical weather features.
   - Now fine-tuned by selecting the best model from:
     - Gradient Boosting
     - HistGradientBoosting
     - Random Forest
   - Selection is done with time-series cross-validation.

## Important limitation

If you do not provide real inverter or meter output as `actual_power_kw`, the model is trained against baseline-derived targets. That improves consistency and feature fitting, but it does **not** prove real-world accuracy against actual plant generation.

For real accuracy measurement, add plant output and evaluate:

- MAE
- RMSE
- R²
- MAPE or NMAE for plant reporting

## How to check current training quality

Train the model:

```bash
curl -X POST http://127.0.0.1:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"latitude": 22.25, "longitude": 72.74}'
```

The response now includes:

- `selected_model`
- `cross_validation`
- `metrics`

## Why some Grafana panels were blank

The cloud cover, wind speed, and pressure panels were failing when Grafana temporarily had an empty longitude variable. The dashboard SQL now uses safe fallbacks for empty variables and casts numeric columns explicitly.

## 15-minute interval guarantee

Weather ingestion and stored forecasts are aligned to 15-minute intervals.

You can verify this from the database:

```bash
docker exec solar_timescaledb \
  psql -U solar -d solar \
  -c "select timestamp from solar_power_predictions order by timestamp desc limit 8;"
```

## Manual start

Start Docker services:

```bash
docker compose up -d timescaledb grafana api
```

Run the API locally instead of Docker:

```bash
python3.11 -m venv .venvrun
./.venvrun/bin/python -m pip install -r requirements.txt
POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=55432 POSTGRES_DB=solar POSTGRES_USER=solar POSTGRES_PASSWORD=solar123 \
./.venvrun/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Manual stop

Stop Docker services:

```bash
docker compose down
```

Stop local API:

```bash
pkill -f "uvicorn app.main:app"
```

## PM2 start

Install dependencies first, then run:

```bash
pm2 start ecosystem.config.js
pm2 save
pm2 status
```

This runs:

- `solar-api`
- `solar-realtime-updater`

The updater refreshes real-time forecast data every 15 minutes.

## PM2 stop

```bash
pm2 stop solar-realtime-updater
pm2 stop solar-api
```

Stop everything:

```bash
pm2 delete ecosystem.config.js
```

## Recommended path for higher real accuracy

1. Add actual inverter or meter output as `actual_power_kw`.
2. Keep 15-minute aligned timestamps.
3. Retrain daily or weekly.
4. Compare baseline vs ML on held-out data.
5. Tune panel parameters:
   - area
   - efficiency
   - inverter efficiency
   - temperature coefficient
   - tilt
   - azimuth
