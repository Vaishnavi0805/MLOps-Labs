#!/bin/bash
set -e

# Remove existing .env file if it exists
rm -f .env

# Stop and remove containers, networks, and volumes
docker compose down -v

# Create required Airflow directories
mkdir -p ./logs ./plugins ./config ./model

# Write the current user's UID into .env
echo "AIRFLOW_UID=$(id -u)" > .env

# Initialize Airflow database
echo "Initializing Airflow database..."
docker compose up airflow-init

echo "Setup completed successfully!"
echo "Run 'docker compose up' to start Airflow"