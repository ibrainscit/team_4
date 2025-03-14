#!/bin/bash

# Define the URL of the zip file
ZIP_URL="https://brainclinics.com/images/downloads/TDBRAIN-dataset.zip"
ZIP_NAME="TDBRAIN-dataset.zip"
EXTRACT_FOLDER="TDBRAIN-dataset"
# PASSWORD='!Bra1n$Rgr3at:)'

# Run wget in the background using nohup and capture its PID
nohup wget -O "$ZIP_NAME" "$ZIP_URL" > download.log 2>&1 &
wget_pid=$!
wait "$wget_pid"

# Create the extraction folder if it doesn't exist
mkdir -p "$EXTRACT_FOLDER"

# Extract the zip file using the password and capture its PID
nohup unzip -P '!Bra1n$Rgr3at:)' "$ZIP_NAME" -d "$EXTRACT_FOLDER" > extract.log 2>&1 &
unzip_pid=$!
wait "$unzip_pid"

# Remove the zip file after extraction (optional)
# rm -f "$ZIP_NAME"
