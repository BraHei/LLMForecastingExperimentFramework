#!/bin/bash

if [ -e /dev/kfd ]; then
    echo "Using ROCm GPU environment"
    docker compose --profile gpu run --build --rm dev-gpu
else
    echo "Using CPU environment"
    docker compose --profile cpu run --build --rm dev-cpu
fi

