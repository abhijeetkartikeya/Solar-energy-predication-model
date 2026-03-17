CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS solar_power_predictions (
    timestamp TIMESTAMPTZ NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    predicted_power_kw DOUBLE PRECISION NOT NULL,
    baseline_power_kw DOUBLE PRECISION NOT NULL,
    model_power_kw DOUBLE PRECISION NOT NULL,
    global_horizontal_irradiance DOUBLE PRECISION NOT NULL,
    global_tilted_irradiance DOUBLE PRECISION NOT NULL,
    temperature_2m DOUBLE PRECISION NOT NULL,
    wind_speed_10m DOUBLE PRECISION NOT NULL,
    cloud_cover DOUBLE PRECISION NOT NULL,
    humidity DOUBLE PRECISION NOT NULL,
    pressure DOUBLE PRECISION NOT NULL,
    generation_type TEXT NOT NULL,
    solar_data_source TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp, latitude, longitude)
);

SELECT create_hypertable(
    'solar_power_predictions',
    'timestamp',
    if_not_exists => TRUE
);
