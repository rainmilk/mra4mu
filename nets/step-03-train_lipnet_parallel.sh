#!/bin/bash

# Directory for saving machine unlearning results
BASE_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data"
LIPS_SAVE_DATA_DIR="$BASE_SAVE_DATA_DIR/lipnet"

# Check if required directories exist
if [ ! -d "$BASE_SAVE_DATA_DIR" ]; then
    echo "Directory for saving MU data directory not found: $BASE_SAVE_DATA_DIR"
    exit 1
fi

if [ ! -d "$LIPS_SAVE_DATA_DIR" ]; then
    echo "Directory for saving lipschitz data directory not found: $LIPS_SAVE_DATA_DIR"
    exit 1
fi

# Directory for saving the training outputs like Models and Logs
BASE_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs"

if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "Directory for saving outputs not found: $BASE_OUTPUT_DIR"
    exit 1
fi

# Directory for saving the training outputs like Models and Logs
LIPS_OUTPUT_DIR="$BASE_OUTPUT_DIR/lipnet"

if [ ! -d "$LIPS_OUTPUT_DIR" ]; then
    echo "Directory for saving lipschitz models not found: $LIPS_OUTPUT_DIR"
    exit 1
fi

# Execute under directory: lips-mu/nets

# Function to run training and testing tasks concurrently
run_task() {
    local gpu_id=$1
    local arch=$2
    local dataset=$3
    local num_classes=$4

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python train.py --epoch 200 --data $BASE_SAVE_DATA_DIR --lip_save_dir $LIPS_OUTPUT_DIR/$arch/$dataset --test_data_dir $BASE_SAVE_DATA_DIR/$dataset --save_forget_dir $LIPS_SAVE_DATA_DIR/$arch/$dataset --num_classes $num_classes > $arch-$dataset-train.log 2>&1 &

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python train.py --epoch 200 --data $BASE_SAVE_DATA_DIR --lip_save_dir $LIPS_OUTPUT_DIR/$arch/$dataset --test_data_dir $BASE_SAVE_DATA_DIR/$dataset --save_forget_dir $LIPS_SAVE_DATA_DIR/$arch/$dataset --num_classes $num_classes --resume_lipnet > $arch-$dataset-test.log 2>&1 &
}

# Define tasks
tasks=(
    "0 resnet18 cifar10 10"
    "1 resnet18 cifar100 100"
    "2 resnet18 tinyimgnet 200"
    "3 resnet18 fmnist 10"
)

# Run tasks concurrently
for task in "${tasks[@]}"; do
    run_task $task
done

echo "All tasks started."
