# Directory for saving machine unlearning results
BASE_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data"
LIPS_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data/lipnet"

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
LIPS_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs/lipnet"

if [ ! -d "$LIPS_OUTPUT_DIR" ]; then
    echo "Directory for saving lipschitz models not found: $LIPS_OUTPUT_DIR"
    exit 1
fi

# Execute under directory: lips-mu/nets

# train lipschitz network

# (1) resnet18 cifar10
# train
python train.py --epoch 200 --data $BASE_SAVE_DATA_DIR --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --num_classes 10
# test
python train.py --epoch 200 --data $BASE_SAVE_DATA_DIR --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --num_classes 10 --resume_lipnet

# (2) resnet18 cifar100
# train
python train.py --epoch 200 --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100
# test
python train.py --epoch 200 --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --resume_lipnet

# (3) resnet18 tinyimgnet
# train
python train.py --epoch 200 --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200
# test
python train.py --epoch 200 --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --resume_lipnet

# (4) resnet18 fMNIST
# train
python train.py --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10
# test
python train.py --epoch 200 --lip_save_dir $LIPS_OUTPUT_DIR/resnet18/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --resume_lipnet
