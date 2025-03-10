#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

logfile="log/test_$timestamp.log"

# python train.py --config configs/0022/0022.yaml 2>&1 | tee $logfile
python train_without_actor.py --config configs/0022/0022.yaml 2>&1 | tee $logfile