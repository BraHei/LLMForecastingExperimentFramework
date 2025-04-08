#!/bin/bash

# URL of the dataset
URL="https://nhits-experiments.s3.amazonaws.com/datasets.zip"
ZIP_FILE="datasets.zip"

# Download the zip file
echo "Downloading dataset..."
curl -O "$URL"

# Unzip the dataset using the password
echo "Unzipping dataset..."
unzip "$ZIP_FILE" -d "./nixtla_datasets"

rm "$ZIP_FILE"

echo "Done!"

