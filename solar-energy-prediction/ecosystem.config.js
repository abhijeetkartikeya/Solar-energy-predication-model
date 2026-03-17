module.exports = {
  apps: [
    {
      name: "solar-api",
      script: "uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      cwd: "/Users/kartikeya/Desktop/solar energy predication model/solar-energy-prediction",
      env: {
        ENABLE_INTERNAL_SCHEDULER: "false",
        POSTGRES_HOST: "127.0.0.1",
        POSTGRES_PORT: "55432",
        POSTGRES_DB: "solar",
        POSTGRES_USER: "solar",
        POSTGRES_PASSWORD: "solar123"
      }
    },
    {
      name: "solar-realtime-updater",
      script: "scripts/realtime_updater.py",
      interpreter: "python3",
      cwd: "/Users/kartikeya/Desktop/solar energy predication model/solar-energy-prediction",
      env: {
        ENABLE_INTERNAL_SCHEDULER: "false",
        POSTGRES_HOST: "127.0.0.1",
        POSTGRES_PORT: "55432",
        POSTGRES_DB: "solar",
        POSTGRES_USER: "solar",
        POSTGRES_PASSWORD: "solar123"
      }
    }
  ]
};
