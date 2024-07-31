#!/bin/bash

# Directory paths
BASE_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data"
LIPS_SAVE_DATA_DIR="$BASE_SAVE_DATA_DIR/lipnet"
BASE_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs"
LIPS_OUTPUT_DIR="$BASE_OUTPUT_DIR/lipnet"

# Check directories
for dir in "$BASE_SAVE_DATA_DIR" "$LIPS_SAVE_DATA_DIR" "$BASE_OUTPUT_DIR" "$LIPS_OUTPUT_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Directory not found: $dir"
        exit 1
    fi
done

# Define tasks and assign GPUs
tasks=(
    "FT_prune cifar10 resnet18 0"
    "FT_prune cifar100 resnet18 1"
    "fisher tinyimgnet resnet18 2"
    "FT_prune tinyimgnet resnet18 3"
    "wfisher fmnist resnet18 4"
    "FT_prune fmnist resnet18 5"
    "wfisher cifar10 vgg16_bn_lth 6"
    "FT_prune cifar10 vgg16_bn_lth 7"
    "wfisher cifar100 vgg16_bn_lth 0"
    "FT_prune cifar100 vgg16_bn_lth 1"
    "wfisher tinyimgnet vgg16_bn_lth 2"
    "FT_prune tinyimgnet vgg16_bn_lth 3"
    "FT_prune fmnist vgg16_bn_lth 4"
)

# Function to run a task
run_task() {
    local unlearn_method=$1
    local dataset=$2
    local arch=$3
    local gpu=$4

    save_dir="$BASE_OUTPUT_DIR/${arch}_${dataset}/$unlearn_method"
    lip_save_dir="$LIPS_OUTPUT_DIR/$dataset"
    test_data_dir="$BASE_SAVE_DATA_DIR/$dataset"
    save_forget_dir="$LIPS_SAVE_DATA_DIR/${arch}/$dataset"
    log_file="./logs/${arch}_${dataset}_${unlearn_method}.log"

    CUDA_VISIBLE_DEVICES=$gpu nohup python -u ft_unlearn.py \
        --epoch 10 \
        --unlearn_lr 0.001 \
        --save_dir $save_dir \
        --unlearn $unlearn_method \
        --lip_save_dir $lip_save_dir \
        --test_data_dir $test_data_dir \
        --save_forget_dir $save_forget_dir \
        --dataset $dataset \
        --num_classes $(if [ "$dataset" = "cifar100" ] || [ "$dataset" = "tinyimgnet" ]; then echo 100; elif [ "$dataset" = "fashionMNIST" ]; then echo 10; else echo 10; fi) \
        --arch $arch \
        > $log_file 2>&1 &
}

# Run all tasks
for task in "${tasks[@]}"; do
    run_task $task
done

echo "All tasks started."
