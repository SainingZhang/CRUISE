#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

logfile="log/test_09_$timestamp.log"

python train.py --config configs/by_transection/yizhuang09.yaml 2>&1 | tee $logfile