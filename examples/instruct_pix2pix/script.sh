#!/bin/bash

# Define the directory where the files will be downloaded
DOWNLOAD_DIR="data/"

# Create the download directory if it doesn't exist
mkdir -p $DOWNLOAD_DIR

# File containing the list of URLs
URL_FILE="urls.txt"

# Check if the URL file exists
if [ ! -f $URL_FILE ]; then
    echo "URL file $URL_FILE does not exist."
    exit 1
fi

# Read URLs from the file and download each one
while read -r url; do
    echo "Downloading: $url"
    # Extract the filename from the URL
    filename=$(basename $url)
    # Use aria2c to download the file
    aria2c -x 16 -s 16 -k 1M "$url" -d $DOWNLOAD_DIR -o $filename
done < $URL_FILE

echo "All files have been downloaded to $DOWNLOAD_DIR."
