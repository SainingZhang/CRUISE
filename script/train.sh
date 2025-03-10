#!/bin/bash

# Check if correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Error: Please provide GPU ID and version number"
    echo "Usage: $0 <gpu_id> <version>"
    echo "Example: $0 0 07"
    exit 1
fi

# Function to check if specific GPU is free (returns 1 if GPU is free, 0 if busy)
check_gpu() {
    local gpu_id=$1
    # Check if any process is using the GPU
    local gpu_usage=$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ $gpu_usage -eq 0 ]; then
        return 1  # GPU is free
    else
        return 0  # GPU is busy
    fi
}

# Get the GPU ID and version number from command line arguments
gpu_id=$1
version=$2

# Check if GPU exists
num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
if [ $gpu_id -ge $num_gpus ]; then
    echo "Error: GPU $gpu_id does not exist. System has $num_gpus GPUs (0-$((num_gpus-1)))"
    exit 1
fi

# Set the config file path based on version
config="configs/seqs/00${version}.yaml"

# Check if config file exists
if [ ! -f "$config" ]; then
    echo "Error: Config file $config not found"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p log

# Continuously check GPU until it's free
while true; do
    if check_gpu $gpu_id; then
        echo "$(date): GPU $gpu_id is busy, checking again in 5 minutes..."
        sleep 300  # Wait for 5 minutes
    else
        echo "$(date): GPU $gpu_id is free, starting training..."
        break
    fi
done

# Generate timestamp when starting the training
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="log/seq_00${version}_${timestamp}.log"

# Run the training script with the specified config on the specified GPU
echo "Starting training with config: $config on GPU $gpu_id"
echo "Logs will be saved to: $logfile"
python train.py --config "$config" 2>&1 | tee "$logfile"