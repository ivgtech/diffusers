#!/bin/bash

# Number of files to download, passed as an argument, -1 for all files
NUM_FILES=$1
NUM_FILES=-1

# Identify the worker file list automatically by searching for the worker file pattern
FILE_LIST=$(ls worker_*_files.txt 2> /dev/null | head -n 1)

if [[ -z "$FILE_LIST" ]]; then
    echo "No worker file list found."
    exit 1
fi

# Extract the worker ID from the filename
WORKER_ID=$(echo "$FILE_LIST" | grep -oP 'worker_\K[0-9]+(?=_files.txt)')

if [[ -z "$WORKER_ID" ]]; then
    echo "Failed to extract worker ID from file list."
    exit 1
fi

# Create the download directory if it doesn't exist
DOWNLOAD_DIR="data/"
mkdir -p $DOWNLOAD_DIR

# Check free disk space
FREE_SPACE_GB=$(df -BG "$DOWNLOAD_DIR" | tail -n 1 | awk '{print $4}' | sed 's/G//')

if [[ $FREE_SPACE_GB -lt 10 ]]; then
    echo "Error: Not enough disk space. Only $FREE_SPACE_GB GB available, need at least 10 GB."
    exit 1
fi

# Counter to track number of files downloaded
count=0

# Read each URL from the file list and download using aria2c
while IFS= read -r url
do
    filename=$(basename $url)
    full_path="$DOWNLOAD_DIR/$filename"

    # Check if the file already exists
    if [[ -f "$full_path" ]]; then
        echo "File $filename already exists, skipping download."
        continue
    fi

    # Proceed with download if file does not exist
    if [[ $NUM_FILES -eq -1 ]] || (( count < NUM_FILES )); then
        aria2c -x 16 -s 16 -k 1M "$url" -d $DOWNLOAD_DIR -o $filename
        ((count++))  # Increment the counter after each download
    else
        break  # Exit the loop after downloading the specified number of files
    fi
done < "$FILE_LIST"

