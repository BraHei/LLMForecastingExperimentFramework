#!/bin/bash
set -e

echo "Generating .env with UID and GID..."

echo "UID=$(id -u)" > ./.env
echo "GID=$(id -g)" >> ./.env
