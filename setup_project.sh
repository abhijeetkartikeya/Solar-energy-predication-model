#!/bin/bash

PROJECT_NAME="solar-energy-prediction"

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Core folders
mkdir -p app/api
mkdir -p app/services
mkdir -p app/models
mkdir -p app/core

mkdir -p ml/data/raw
mkdir -p ml/data/processed
mkdir -p ml/features
mkdir -p ml/training
mkdir -p ml/utils

mkdir -p tests
mkdir -p docker

# Root files
touch requirements.txt
touch Dockerfile
touch docker-compose.yml
touch .dockerignore
touch README.md
touch .env

# App files
touch app/main.py
touch app/api/routes.py
touch app/api/schemas.py
touch app/services/prediction_service.py
touch app/services/training_service.py
touch app/core/config.py

# ML files
touch ml/features/build_features.py
touch ml/training/train.py
touch ml/training/evaluate.py
touch ml/utils/metrics.py

echo "Project structure created successfully!"#!/bin/bash

PROJECT_NAME="solar-energy-prediction"

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Core folders
mkdir -p app/api
mkdir -p app/services
mkdir -p app/models
mkdir -p app/core

mkdir -p ml/data/raw
mkdir -p ml/data/processed
mkdir -p ml/features
mkdir -p ml/training
mkdir -p ml/utils

mkdir -p tests
mkdir -p docker

# Root files
touch requirements.txt
touch Dockerfile
touch docker-compose.yml
touch .dockerignore
touch README.md
touch .env

# App files
touch app/main.py
touch app/api/routes.py
touch app/api/schemas.py
touch app/services/prediction_service.py
touch app/services/training_service.py
touch app/core/config.py

# ML files
touch ml/features/build_features.py
touch ml/training/train.py
touch ml/training/evaluate.py
touch ml/utils/metrics.py

echo "Project structure created successfully!"