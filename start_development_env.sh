#!/bin/bash

# Exit on error
set -e

# Write UID/GID into .env so Compose can pick them up
echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env

if [ -e /dev/kfd ]; then
    echo "Using ROCm GPU environment"
    docker compose --profile gpu run --build --rm dev-gpu
else
    echo "Using CPU environment"
    docker compose --profile cpu run --build --rm dev-cpu
fi