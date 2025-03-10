#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

logfile="log/test_06_$timestamp.log"

python train.py --config configs/by_transection/yizhuang06.yaml 2>&1 | tee $logfile