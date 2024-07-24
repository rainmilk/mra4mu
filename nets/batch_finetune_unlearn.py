import os
import multiprocessing

def run_command(command, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system(command)

commands = [
    # Resnet18 Cifar10
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar10/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar10/FT --unlearn FT --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar10/GA --unlearn GA --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar10/FF --unlearn fisher --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar10/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar10/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn",

    # Resnet18 Cifar100
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar100/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar100/FT --unlearn FT --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar100/GA --unlearn GA --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar100/FF --unlearn fisher --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar100/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_cifar100/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn",

    # Resnet18 TinyImagenet
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_tinyimg/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_tinyimg/FT --unlearn FT --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_tinyimg/GA --unlearn GA --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_tinyimg/FF --unlearn fisher --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_tinyimg/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_tinyimg/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn",

    # Resnet18 fashionMNIST
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_fmnist/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_fmnist/FT --unlearn FT --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_fmnist/GA --unlearn GA --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_fmnist/FF --unlearn fisher --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_fmnist/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/resnet18_fmnist/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn",

    # VGG16 Cifar10
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar10/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar10/FT --unlearn FT --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar10/GA --unlearn GA --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar10/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar10/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --dataset cifar10 --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",

    # VGG16 Cifar100
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar100/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar100/FT --unlearn FT --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar100/GA --unlearn GA --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar100/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_cifar100/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --finetune_unlearn --arch vgg16_bn_lth",

    # VGG16 TinyImagenet
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_tinyimg/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_tinyimg/FT --unlearn FT --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_tinyimg/GA --unlearn GA --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_tinyimg/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_tinyimg/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --finetune_unlearn --arch vgg16_bn_lth",

    # VGG16 fashionMNIST
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_fmnist/retrain --unlearn retrain --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_fmnist/FT --unlearn FT --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_fmnist/GA --unlearn GA --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_fmnist/IU --unlearn wfisher --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
    "python ft_unlearn.py --epoch 10 --unlearn_lr 0.001 --save_dir ../outputs/vgg16_fmnist/FT_prune --unlearn FT_prune --lip_save_dir ../outputs/lipnet/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --finetune_unlearn --arch vgg16_bn_lth",
]

if __name__ == "__main__":
    processes = []
    num_gpus = 8

    for i, command in enumerate(commands):
        gpu_id = i % num_gpus
        p = multiprocessing.Process(target=run_command, args=(command, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
