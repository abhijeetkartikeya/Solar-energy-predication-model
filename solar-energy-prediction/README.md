# Solar Energy Output Forecasting System

This project forecasts solar panel power output for a given latitude and longitude at 15-minute resolution. It keeps the existing Python project structure, adds a formula-based baseline, supports an optional ML model, stores results in TimescaleDB, and visualizes them in Grafana.

## Architecture

```text
Open-Meteo + optional VEDAS
          |
          v
ml/data/fetch_weather.py
ml/data/fetch_solar_data.py
          |
          v
ml/features/feature_engineering.py
          |
          +--> formula baseline power estimate
          |
          +--> optional ML training/inference
                 ml/training/train_model.py
                 ml/inference/predict_power.py
          |
          v
app/services/prediction_service.py
          |
          v
TimescaleDB / PostgreSQL
          |
          v
Grafana dashboard
```

## Project Structure

```text
solar-energy-prediction/
├── app/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   └── config.py
│   ├── services/
│   │   ├── prediction_service.py
│   │   ├── timescaledb_service.py
│   │   └── training_service.py
│   └── main.py
├── ml/
│   ├── data/
│   │   ├── fetch_weather.py
│   │   └── fetch_solar_data.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── inference/
│   │   └── predict_power.py
│   ├── training/
│   │   └── train_model.py
│   └── utils/
│       ├── config.py
│       └── logging.py
├── docker/
│   ├── postgres/init.sql
│   └── grafana/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Data Flow

1. Weather data is fetched from Open-Meteo for the last 7 days and next 3 days.
2. Optional VEDAS solar radiation data is merged when `VEDAS_API_BASE_URL` and `VEDAS_API_KEY` are configured.
3. The pipeline aligns all observations to a unified 15-minute time series.
4. Feature engineering computes irradiance, temperature-loss, daylight, and seasonal features.
5. A formula baseline estimates panel output from irradiance, panel area, efficiency, temperature losses, and inverter efficiency.
6. An optional Gradient Boosting model can be trained per coordinate and blended with the baseline.
7. Predictions are upserted into TimescaleDB and shown in Grafana.

## Storage Schema

Primary table: `solar_power_predictions`

Important columns:

- `timestamp`
- `latitude`
- `longitude`
- `predicted_power_kw`
- `baseline_power_kw`
- `model_power_kw`
- `global_horizontal_irradiance`
- `global_tilted_irradiance`
- `temperature_2m`
- `wind_speed_10m`
- `cloud_cover`
- `humidity`
- `pressure`
- `generation_type`

## Run Locally

```bash
cd "/Users/kartikeya/Desktop/solar energy predication model/solar-energy-prediction"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Environment variables you will usually want:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=solar
export POSTGRES_USER=solar
export POSTGRES_PASSWORD=solar123
export DEFAULT_LATITUDE=22.5726
export DEFAULT_LONGITUDE=88.3639
```

Optional VEDAS configuration:

```bash
export VEDAS_API_BASE_URL="https://your-vedas-endpoint"
export VEDAS_API_KEY="your-api-key"
```

## Run With Docker

```bash
docker compose up --build
```

Services:

- API docs: `http://localhost:8000/docs`
- Grafana: `http://localhost:3000`
- TimescaleDB: `localhost:5432`

Default Grafana credentials:

- username: `admin`
- password: `admin`

## API

Generate a forecast:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 22.5726,
    "longitude": 88.3639,
    "train_model": true
  }'
```

Shortcut GET endpoint:

```bash
curl "http://localhost:8000/api/predict?lat=22.5726&lon=88.3639"
```

Train only:

```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"latitude": 22.5726, "longitude": 88.3639}'
```

Read stored predictions:

```bash
curl "http://localhost:8000/api/predictions?lat=22.5726&lon=88.3639&hours=240"
```

## Continuous Updates

The API service runs an APScheduler job every 15 minutes. By default it refreshes forecasts for the configured `DEFAULT_LATITUDE` and `DEFAULT_LONGITUDE`, fetches the latest weather inputs, recomputes power forecasts, and upserts them into TimescaleDB.

For additional coordinates, call the API on demand or add your own scheduler wrapper around `/api/predict`.

## PM2 Realtime Updates

If you want PM2 to keep the graph updated continuously, use the included PM2 ecosystem file. In this mode PM2 runs:

- the FastAPI server
- a dedicated Python worker that refreshes TimescaleDB on the configured interval

Install dependencies first, then start PM2:

```bash
cd "/Users/kartikeya/Desktop/solar energy predication model/solar-energy-prediction"
pip install -r requirements.txt
pm2 start ecosystem.config.js
pm2 save
pm2 status
```

Useful PM2 commands:

```bash
pm2 logs solar-realtime-updater
pm2 restart solar-realtime-updater
pm2 restart solar-api
pm2 stop ecosystem.config.js
```

Important:

- `ENABLE_INTERNAL_SCHEDULER=false` is set in the PM2 config so only the PM2 worker performs periodic updates.
- The worker writes fresh 15-minute forecast rows into TimescaleDB, which Grafana reads directly.
- Adjust `SCHEDULER_INTERVAL_MINUTES`, `DEFAULT_LATITUDE`, and `DEFAULT_LONGITUDE` in your environment before starting PM2 if needed.

## Manual Start And Stop

Start all Docker services:

```bash
docker compose up -d timescaledb grafana api
```

Stop them:

```bash
docker compose down
```

Run the API manually:

```bash
POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=55432 POSTGRES_DB=solar POSTGRES_USER=solar POSTGRES_PASSWORD=solar123 \
./.venvrun/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Stop the manual API process:

```bash
pkill -f "uvicorn app.main:app"
```

## Accuracy Notes

- The forecast output is stored at 15-minute resolution.
- The baseline model is the reliable fallback.
- The training pipeline now performs time-series cross-validation and selects the best model among multiple tree-based regressors.
- Real accuracy against plant generation requires actual inverter output. Without `actual_power_kw`, the ML model is learning from the baseline-derived target rather than ground-truth plant power.

Detailed operations and accuracy guidance is available in [`docs/OPERATIONS_AND_ACCURACY.md`](/Users/kartikeya/Desktop/solar%20energy%20predication%20model/solar-energy-prediction/docs/OPERATIONS_AND_ACCURACY.md).

## Grafana Dashboard

The provisioned dashboard shows:

- historical plus forecasted power output
- GHI and GTI
- temperature
- cloud cover
- wind speed
- pressure

Grafana uses the provisioned TimescaleDB datasource and lets you switch coordinates using dashboard variables for latitude and longitude.

## Notes

- The formula baseline is the main production-safe fallback.
- The ML model is optional and is trained using the latest weather-derived target when no real meter data is available.
- If you later add actual inverter output, place it into `actual_power_kw` before training and the ML pipeline will use it automatically.
