#!/bin/bash

# Set user ID so file saving is fixed
export UID=$(id -u)
export GID=$(id -g)

if [ -e /dev/kfd ]; then
    echo "Using ROCm GPU environment"
    docker compose --profile gpu run --build --rm dev-gpu
else
    echo "Using CPU environment"
    docker compose --profile cpu run --build --rm dev-cpu
fi

