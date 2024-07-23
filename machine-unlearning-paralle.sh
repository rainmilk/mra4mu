#!/bin/bash

# Define the GPUs to be used
GPUS=(0 1 2 3 4 5 6 7)

# Function to execute a training task on a specific GPU
run_training() {
    local gpu=$1
    local save_dir=$2
    local unlearn_method=$3
    local class_to_replace=$4
    local num_indexes_to_replace=$5
    local unlearn_epochs=$6
    local unlearn_lr=$7
    local arch=$8
    local dataset=$9
    local additional_params=${10}
    local log_file=${11}
    local data_dir=${12}

    CUDA_VISIBLE_DEVICES=$gpu nohup python -u main_forget.py \
        --save_dir $save_dir \
        --unlearn $unlearn_method \
        --class_to_replace $class_to_replace \
        --num_indexes_to_replace $num_indexes_to_replace \
        --unlearn_epochs $unlearn_epochs \
        --unlearn_lr $unlearn_lr \
        --arch $arch \
        --dataset $dataset \
        --data_dir $data_dir \
        $additional_params \
        > $log_file 2>&1 &
}

# Data directories
CIFAR10_DIR="../data/cifar-10"
CIFAR100_DIR="../data/cifar-100"
FASHIONMNIST_DIR="../data/fashion-mnist"
TINYIMAGENET_DIR="../data/tiny-imagenet-200"

# Index to keep track of GPU allocation
GPU_INDEX=0
NUM_GPUS=${#GPUS[@]}

# 1.1.1 Resnet18 Cifar10
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/finetune_backbone retrain 0 1 100 0.1 resnet18 cifar10 "" Resnet18-Cifar10-bbft.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/retrain retrain 0,1,3,5,6 2250 100 0.1 resnet18 cifar10 "" Resnet18-Cifar10-retrain.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/FT FT 0,1,3,5,6 2250 100 0.1 resnet18 cifar10 "--load_ff --resume" Resnet18-Cifar10-ft.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/GA GA 0,1,3,5,6 2250 4 0.0001 resnet18 cifar10 "--load_ff --resume" Resnet18-Cifar10-ga.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/FF fisher 0,1,3,5,6 2250 100 0.1 resnet18 cifar10 "--load_ff --resume --alpha 16.5" Resnet18-Cifar10-ff.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/IU wfisher 0,1,3,5,6 2250 100 0.1 resnet18 cifar10 "--load_ff --resume --alpha 16" Resnet18-Cifar10-iu.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar10/FT_prune FT_prune 0,1,3,5,6 2250 30 0.00001 resnet18 cifar10 "--load_ff --resume" Resnet18-Cifar10-ft-prune.log $CIFAR10_DIR
((GPU_INDEX++))

# 1.1.2 Resnet18 Cifar100
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/finetune_backbone retrain 0 1 100 0.1 resnet18 cifar100 "" Resnet18-Cifar100-bbft.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/retrain retrain 11,22,33,44,55 225 100 0.1 resnet18 cifar100 "" Resnet18-Cifar100-retrain.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/FT FT 11,22,33,44,55 225 100 0.1 resnet18 cifar100 "--load_ff --resume" Resnet18-Cifar100-ft.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/GA GA 11,22,33,44,55 225 4 0.001 resnet18 cifar100 "--load_ff --resume" Resnet18-Cifar100-ga.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/FF fisher 11,22,33,44,55 225 100 0.1 resnet18 cifar100 "--load_ff --resume --alpha 20" Resnet18-Cifar100-ff.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/IU wfisher 11,22,33,44,55 225 100 0.1 resnet18 cifar100 "--load_ff --resume --alpha 160" Resnet18-Cifar100-iu.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_cifar100/FT_prune FT_prune 11,22,33,44,55 225 20 0.00001 resnet18 cifar100 "--load_ff --resume" Resnet18-Cifar100-ft-prune.log $CIFAR100_DIR
((GPU_INDEX++))

# 1.1.3 Resnet18 TinyImagenet
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/finetune_backbone retrain 0 1 100 0.1 resnet18 TinyImagenet "" Resnet18-tinyim-bbft.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/retrain retrain 1,51,101,151,198 250 100 0.1 resnet18 TinyImagenet "" Resnet18-tinyim-retrain.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/FT FT 1,51,101,151,198 250 100 0.1 resnet18 TinyImagenet "--load_ff --resume" Resnet18-tinyim-ft.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/GA GA 1,51,101,151,198 250 5 0.00001 resnet18 TinyImagenet "--load_ff --resume" Resnet18-tinyim-ga.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/FF fisher 1,51,101,151,198 250 100 0.1 resnet18 TinyImagenet "--load_ff --resume --alpha 10.2" Resnet18-tinyim-ff.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/IU wfisher 1,51,101,151,198 250 100 0.1 resnet18 TinyImagenet "--load_ff --resume --alpha 160" Resnet18-tinyim-iu.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_tinyimg/FT_prune FT_prune 1,51,101,151,198 250 4 0.000005 resnet18 TinyImagenet "--load_ff --resume" Resnet18-tinyim-ft-prune.log $TINYIMAGENET_DIR
((GPU_INDEX++))

# 1.1.4 Resnet18 fashionMNIST
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/finetune_backbone retrain 0 1 100 0.1 resnet18 fashionMNIST "" Resnet18-fsmnist-bbft.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/retrain retrain 1,3,5,7,9 2700 100 0.1 resnet18 fashionMNIST "" Resnet18-fsmnist-retrain.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/FT FT 1,3,5,7,9 2700 100 0.1 resnet18 fashionMNIST "--load_ff --resume" Resnet18-fsmnist-ft.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/GA GA 1,3,5,7,9 2700 5 0.00001 resnet18 fashionMNIST "--load_ff --resume" Resnet18-fsmnist-ga.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/FF fisher 1,3,5,7,9 2700 100 0.1 resnet18 fashionMNIST "--load_ff --resume --alpha 16.5" Resnet18-fsmnist-ff.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/IU wfisher 1,3,5,7,9 2700 100 0.1 resnet18 fashionMNIST "--load_ff --resume --alpha 60" Resnet18-fsmnist-iu.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/resnet18_fmnist/FT_prune FT_prune 1,3,5,7,9 2700 20 0.00001 resnet18 fashionMNIST "--load_ff --resume" Resnet18-fsmnist-ft-prune.log $FASHIONMNIST_DIR
((GPU_INDEX++))

# 1.2.1 vgg16_bn_lth Cifar10
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/finetune_backbone retrain 0 1 100 0.1 vgg16_bn_lth cifar10 "" VGG16-Cifar10-bbft.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/retrain retrain 0,1,3,5,6 2250 100 0.1 vgg16_bn_lth cifar10 "" VGG16-Cifar10-retrain.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/FT FT 0,1,3,5,6 2250 100 0.1 vgg16_bn_lth cifar10 "--load_ff --resume" VGG16-Cifar10-ft.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/GA GA 0,1,3,5,6 2250 4 0.0001 vgg16_bn_lth cifar10 "--load_ff --resume" VGG16-Cifar10-ga.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/FF fisher 0,1,3,5,6 2250 100 0.1 vgg16_bn_lth cifar10 "--load_ff --resume --alpha 16.5" VGG16-Cifar10-ff.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/IU wfisher 0,1,3,5,6 2250 100 0.1 vgg16_bn_lth cifar10 "--load_ff --resume --alpha 1" VGG16-Cifar10-iu.log $CIFAR10_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar10/FT_prune FT_prune 0,1,3,5,6 2250 20 0.00001 vgg16_bn_lth cifar10 "--load_ff --resume" VGG16-Cifar10-ft-prune.log $CIFAR10_DIR
((GPU_INDEX++))

# 1.2.2 vgg16_bn_lth Cifar100
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/finetune_backbone retrain 0 1 100 0.1 vgg16_bn_lth cifar100 "" VGG16-Cifar100-bbft.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/retrain retrain 11,22,33,44,55 225 100 0.1 vgg16_bn_lth cifar100 "" VGG16-Cifar100-retrain.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/FT FT 11,22,33,44,55 225 100 0.1 vgg16_bn_lth cifar100 "--load_ff --resume" VGG16-Cifar100-ft.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/GA GA 11,22,33,44,55 225 4 0.001 vgg16_bn_lth cifar100 "--load_ff --resume" VGG16-Cifar100-ga.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/FF fisher 11,22,33,44,55 225 100 0.1 vgg16_bn_lth cifar100 "--load_ff --resume --alpha 16.5" VGG16-Cifar100-ff.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/IU wfisher 11,22,33,44,55 225 100 0.1 vgg16_bn_lth cifar100 "--load_ff --resume --alpha 6" VGG16-Cifar100-iu.log $CIFAR100_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_cifar100/FT_prune FT_prune 11,22,33,44,55 225 20 0.000005 vgg16_bn_lth cifar100 "--load_ff --resume" VGG16-Cifar100-ft-prune.log $CIFAR100_DIR
((GPU_INDEX++))

# 1.2.3 vgg16_bn_lth tiny image net
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/finetune_backbone retrain 0 1 100 0.1 vgg16_bn_lth TinyImagenet "" VGG16-tinyim-bbft.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/retrain retrain 1,51,101,151,198 250 100 0.1 vgg16_bn_lth TinyImagenet "" VGG16-tinyim-retrain.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/FT FT 1,51,101,151,198 250 100 0.1 vgg16_bn_lth TinyImagenet "--load_ff --resume" VGG16-tinyim-ft.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/GA GA 1,51,101,151,198 250 4 0.0001 vgg16_bn_lth TinyImagenet "--load_ff --resume" VGG16-tinyim-ga.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/FF fisher 1,51,101,151,198 250 100 0.1 vgg16_bn_lth TinyImagenet "--load_ff --resume --alpha 16.5" VGG16-tinyim-ff.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/IU wfisher 1,51,101,151,198 250 100 0.1 vgg16_bn_lth TinyImagenet "--load_ff --resume --alpha 10" VGG16-tinyim-iu.log $TINYIMAGENET_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_tinyimg/FT_prune FT_prune 1,51,101,151,198 250 7 0.000004 vgg16_bn_lth TinyImagenet "--load_ff --resume" VGG16-tinyim-ft-prune.log $TINYIMAGENET_DIR
((GPU_INDEX++))

# 1.2.4 vgg16_bn_lth fashionMNIST
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/finetune_backbone retrain 0 1 100 0.1 vgg16_bn_lth fashionMNIST "" VGG16-fashionmn-bbft.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/retrain retrain 1,3,5,7,9 2700 100 0.1 vgg16_bn_lth fashionMNIST "" VGG16-fashionmn-retrain.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/FT FT 1,3,5,7,9 2700 50 0.1 vgg16_bn_lth fashionMNIST "--load_ff --resume" VGG16-fashionmn-ft.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/GA GA 1,3,5,7,9 2700 4 0.0001 vgg16_bn_lth fashionMNIST "--load_ff --resume" VGG16-fashionmn-ga.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/FF fisher 1,3,5,7,9 2700 100 0.1 vgg16_bn_lth fashionMNIST "--load_ff --resume --alpha 16.5" VGG16-fashionmn-ff.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/IU wfisher 1,3,5,7,9 2700 100 0.1 vgg16_bn_lth fashionMNIST "--load_ff --resume --alpha 40" VGG16-fashionmn-iu.log $FASHIONMNIST_DIR
((GPU_INDEX++))
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} ./outputs/vgg16_fmnist/FT_prune FT_prune 1,3,5,7,9 2700 10 0.00001 vgg16_bn_lth fashionMNIST "--load_ff --resume" VGG16-fashionmn-ft-prune.log $FASHIONMNIST_DIR
((GPU_INDEX++))

echo "All tasks have been started."
