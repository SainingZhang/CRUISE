#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

logfile="log/output_$timestamp.log"

python train.py --config configs/0022/dair_train_0022_0.yaml 2>&1 | tee $logfile