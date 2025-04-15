#!/bin/bash
set -e

echo "Generating .env with UID and GID..."

echo "UID=$(id -u)" > /workspace/.env
echo "GID=$(id -g)" >> /workspace/.env

echo ".env file created inside container."