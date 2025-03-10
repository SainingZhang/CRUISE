#!/bin/bash

# Check if an argument is provided
# if [ $# -ne 2 ]; then
#     echo "Error: Please provide a version number (07, 17, 22, 55, 79 or 82) and mode"
#     echo "Usage: $0 <version> mode <mode>"
#     echo "Example: $0 06 mode evaluate"
#     exit 1
# fi

# Get the version number from command line argument
version=$1
mode=$2

# Set the config file path based on version
# config="configs/seqs/00${version}.yaml"

# Check if config file exists
# if [ ! -f "$config" ]; then
#     echo "Error: Config file $config not found"
#     exit 1
# fi

# Run the training script with the specified config
# echo "Starting rendering with config: $config"
python render_without_actor_training.py --config /mnt/xuhr/street-gs/output/dair_seq_0089/exp_1/configs/config_000000.yaml mode edit_all