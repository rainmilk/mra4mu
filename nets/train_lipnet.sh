# Execute under directory: lips-mu/nets

# train lipschitz network

# (1) resnet18 cifar10
# train
python train.py --epoch 200 --data ../../data --lip_save_dir ../outputs/lipnet/resnet18/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --num_classes 10
# test
python train.py --epoch 200 --data ../../data --lip_save_dir ../outputs/lipnet/resnet18/cifar10 --test_data_dir ../../data/cifar10 --save_forget_dir ../../data/lipnet/resnet18/cifar10 --num_classes 10 --resume_lipnet

# (2) resnet18 cifar100
# train
python train.py --epoch 200 --lip_save_dir ../outputs/lipnet/resnet18/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100
# test
python train.py --epoch 200 --lip_save_dir ../outputs/lipnet/resnet18/cifar100 --test_data_dir ../../data/cifar100 --save_forget_dir ../../data/lipnet/resnet18/cifar100 --dataset cifar100 --num_classes 100 --resume_lipnet

# (3) resnet18 tinyimgnet
# train
python train.py --epoch 200 --lip_save_dir ../outputs/lipnet/resnet18/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200
# test
python train.py --epoch 200 --lip_save_dir ../outputs/lipnet/resnet18/tinyimgnet --test_data_dir ../../data/tinyimgnet --save_forget_dir ../../data/lipnet/resnet18/tinyimgnet --dataset TinyImagenet --num_classes 200 --resume_lipnet

# (4) resnet18 fMNIST
# train
python train.py --lip_save_dir ../outputs/lipnet/resnet18/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10
# test
python train.py --epoch 200 --lip_save_dir ../outputs/lipnet/resnet18/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --resume_lipnet
