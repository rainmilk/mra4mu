#!/bin/bash

# check log dir and create if not exists
if [ ! -d "logs" ]; then
    mkdir logs
fi

# 定义需要执行的脚本及其对应的日志文件
scripts=(
    "scripts/sh/save_predicts_cifar100_ailab.sh"
    "scripts/sh/save_predicts_cifar10_ailab.sh"
    "scripts/sh/save_predicts_fmnist_ailab.sh"
    "scripts/sh/save_predicts_tinyimgnet_ailab.sh"
)

# 定义对应的日志文件名
logs=(
    "logs/save_predicts_cifar100.log"
    "logs/save_predicts_cifar10.log"
    "logs/save_predicts_fmnist.log"
    "logs/save_predicts_tinyimgnet.log"
)

# 分配GPU
gpu_ids=(
    "0,1"
    "2,3"
    "4,5"
    "6,7"
)

# 遍历所有脚本，分配GPU并执行
for i in "${!scripts[@]}"; do
    script=${scripts[$i]}
    log=${logs[$i]}
    gpus=${gpu_ids[$i]}

    echo "Running $script on GPUs $gpus, logging to $log"
    CUDA_VISIBLE_DEVICES=$gpus nohup bash $script > $log 2>&1 &
done

echo "All scripts started."
