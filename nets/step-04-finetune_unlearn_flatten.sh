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
LIPS_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs/lipnet/resnet18"

if [ ! -d "$LIPS_OUTPUT_DIR" ]; then
    echo "Directory for saving lipschitz models not found: $LIPS_OUTPUT_DIR"
    exit 1
fi

# Execute under directory: lips-mu/nets

# finetune unlearn models

# 1. Resnet18 Cifar10
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn

# (4) FF
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/FF --unlearn fisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn

# (5) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn

# (6) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn

# 2. Resnet18 Cifar100
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn

# (4) FF
python ft_unlearn.py python -u ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FF --unlearn fisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn

# (5) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn

# (6) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn

# 3. Resnet18 tinyimgnet
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn

# (4) FF
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/FF --unlearn fisher --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn

# (5) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn

# (6) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn

# 4. Resnet18 fashionMNIST
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn

# (4) FF
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/FF --unlearn fisher --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn

# (5) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn

# (6) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn


# 5. VGG16 Cifar10
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn  --arch vgg16_bn_lth

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn  --arch vgg16_bn_lth

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn  --arch vgg16_bn_lth

# (4) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn  --arch vgg16_bn_lth

# (5) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/cifar10 --test_data_dir $BASE_SAVE_DATA_DIR/cifar10 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn  --arch vgg16_bn_lth

# 6. VGG16 Cifar100
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn  --arch vgg16_bn_lth

# (2) FT
python ft_unlearn.py  --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn  --arch vgg16_bn_lth

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn  --arch vgg16_bn_lth

# (4) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn  --arch vgg16_bn_lth

# (5) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/cifar100 --test_data_dir $BASE_SAVE_DATA_DIR/cifar100 --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn  --arch vgg16_bn_lth

# 7. VGG16 TinyImagenet
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth

# (4) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth

# (5) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/tinyimgnet --test_data_dir $BASE_SAVE_DATA_DIR/tinyimgnet  --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth

# 8. VGG16 fashionMNIST
# (1) retrain
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/retrain --unlearn retrain --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth

# (2) FT
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/FT --unlearn FT --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth

# (3) GA
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/GA --unlearn GA --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth

# (4) IU
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/IU --unlearn wfisher --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth

# (5) FT_prune
python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/FT_prune --unlearn FT_prune --lip_save_dir $LIPS_OUTPUT_DIR/fmnist --test_data_dir $BASE_SAVE_DATA_DIR/fmnist --save_forget_dir $LIPS_SAVE_DATA_DIR/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth
