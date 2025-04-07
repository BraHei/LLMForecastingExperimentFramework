#!/bin/bash

# URL of the dataset
URL="https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
ZIP_FILE="UCR_TimeSeriesAnomalyDatasets2021.zip"

# Download the zip file
echo "Downloading dataset..."
curl -O "$URL"

# Unzip the dataset
echo "Unzipping dataset..."
unzip "$ZIP_FILE"

rm "$ZIP_FILE"

echo "Done!"

