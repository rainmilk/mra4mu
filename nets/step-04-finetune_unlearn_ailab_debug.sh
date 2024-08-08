# Directory for saving machine unlearning results
BASE_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data"
LIPS_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data/lipnet"

if [ ! -d "$BASE_SAVE_DATA_DIR" ]; then
    echo "Directory for saving MU data directory not found: $BASE_SAVE_DATA_DIR"
    exit 1
fi

if [ ! -d "$LIPS_SAVE_DATA_DIR" ]; then
    echo "Directory for saving lipschitz data directory not found: $LIPS_SAVE_DATA_DIR"
    mkdir -p $LIPS_SAVE_DATA_DIR
    exit 1
fi

# Directory for saving the training outputs like Models and Logs
BASE_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs"

if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "Directory for saving outputs not found: $BASE_OUTPUT_DIR"
    exit 1
fi

# Directory for saving the training outputs
LIPS_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs/lipnet/resnet18"

if [ ! -d "$LIPS_OUTPUT_DIR" ]; then
    echo "Directory for saving lipschitz models not found: $LIPS_OUTPUT_DIR"
    exit 1
fi


# Directory for saving the logs
LOG_BASE_DIR="/nvme/szh/data/3ai/lips/logs"
STEP_NAME="step-04-finetune_unlearn"
STEP_LOG_DIR="$LOG_BASE_DIR/$STEP_NAME"

if [ ! -d "$LOG_BASE_DIR" ]; then
    echo "Directory for saving logs not found: $LOG_BASE_DIR"
    exit 1
fi

if [ ! -d "$STEP_LOG_DIR" ]; then
    echo "Directory for saving logs not found: $STEP_LOG_DIR"
    mkdir -p $STEP_LOG_DIR
fi

# Execute under directory: lips-mu/nets

# finetune unlearn models
CUDA_VISIBLE_DEVICES=3 nohup python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FF --unlearn fisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn > $STEP_LOG_DIR/resnet18_cifar100_FF.log 2>&1 &
