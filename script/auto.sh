#!/bin/bash

# Check if both GPU number and script name are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <GPU_NUMBER> <SCRIPT_NAME>"
    echo "Example: $0 0 myscript.sh"
    exit 1
fi

GPU_NUM=$1
SCRIPT_NAME=$2

# Check if the script exists and is executable
if [ ! -x "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME not found or not executable"
    echo "Make sure the script exists and has execute permissions (chmod +x $SCRIPT_NAME)"
    exit 1
fi

while true; do
    # Get process count for the specified GPU
    PROCESS_COUNT=$(nvidia-smi -i $GPU_NUM --query-compute-apps=pid --format=csv,noheader | wc -l)
    
    # If no processes are running (count is 0), run the script
    if [ $PROCESS_COUNT -eq 0 ]; then
        echo "GPU $GPU_NUM is free. Running $SCRIPT_NAME..."
        ./$SCRIPT_NAME
    fi
    
    # Wait for 20 minutes before checking again
    sleep 1800
done