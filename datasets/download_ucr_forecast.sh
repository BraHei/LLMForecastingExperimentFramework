#!/bin/bash

# URL of the dataset
URL="https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
ZIP_FILE="UCRArchive_2018.zip"
PASSWORD="someone"

# Download the zip file
echo "Downloading dataset..."
curl -O "$URL"

# Unzip the dataset using the password
echo "Unzipping dataset..."
unzip -P "$PASSWORD" "$ZIP_FILE"

rm "$ZIP_FILE"

echo "Done!"

