#!/bin/bash

# Check if an argument is provided
if [ $# -ne 2 ]; then
    echo "Error: Please provide a version number (07, 17, 22, 55, 79 or 82) and mode"
    echo "Usage: $0 <version> mode <mode>"
    echo "Example: $0 06 mode evaluate"
    exit 1
fi

# Get the version number from command line argument
version=$1
mode=$2

# Array of valid versions
valid_versions=("07" "17" "22" "55" "79" "82")

# Check if the provided version is valid
is_valid=0
for valid_version in "${valid_versions[@]}"; do
    if [ "$version" == "$valid_version" ]; then
        is_valid=1
        break
    fi
done

if [ $is_valid -eq 0 ]; then
    echo "Error: Invalid version number. Please use one of: ${valid_versions[*]}"
    exit 1
fi

# Set the config file path based on version
config="configs/seqs/00${version}_comp.yaml"

# Check if config file exists
if [ ! -f "$config" ]; then
    echo "Error: Config file $config not found"
    exit 1
fi

# Run the training script with the specified config
echo "Starting rendering with config: $config"
python render_without_actor_training.py --config $config mode $mode