#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

logfile="log/output_0022_0_all_cooperative_with_cooperative_pointcloud_$timestamp.log"

python train.py --config configs/0022/dair_train_0022_0_all_cooperative_with_cooperative_pointcloud.yaml 2>&1 | tee $logfile