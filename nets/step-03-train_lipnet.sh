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

# 此处需要注意FMNIST的通道问题在程序中可能存在被多次修改的问题。
# class CustomDataset(Dataset):
#     def __init__(self, data, label, dataset_name):
#         data = data.astype(np.float32)
#         if dataset_name in ['cifar10', 'cifar100']:
#             data = np.transpose(data, [0, 3, 1, 2])
#             self.data = data / 255
#         elif dataset_name == 'fashionMNIST':
#             '''
#             When training lipschitz network. Namely when executing train_lipnet.sh

#             fashionMNIST数据集是单通道（灰度）图像，只有1个通道。因此，在执行forward方法时，ResNet18的第一个卷积层（期望输入3个通道）与输入图像（1个通道）不匹配。因此将fashionMNIST数据集的单通道图像转换为三通道图像。

#             data[:, np.newaxis, ...] 将数据的形状从 [N, H, W] 变为 [N, 1, H, W]. 然后使用 np.repeat(data[:, np.newaxis, ...], 3, axis=1) 将数据的形状从 [N, 1, H, W] 变为 [N, 3, H, W]，即将单通道图像转换为三通道图像。
#             '''
#             data = np.repeat(data[:, np.newaxis, ...], 3, axis=1)
#             '''
#             - When training unlearning network.
#             - 将数据从二维（单通道）转换为三维，增加了一个通道维度，保持图像为单通道，适用于不需要RGB输入的网络。
#             data[:, np.newaxis, ...] 将数据的形状从 [N, H, W] 变为 [N, 1, H, W]。
#             - 通常用于不需要RGB输入的网络，或者网络结构可以处理单通道输入，如某些自定义的或特殊的卷积神经网络。
#             '''
#             # data = data[:, np.newaxis, ...]
#             self.data = data / 255
#         elif dataset_name == 'TinyImagenet':
#             self.data = data
#         self.label = label


# train
python train.py --lip_save_dir ../outputs/lipnet/resnet18/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10
# test
python train.py --epoch 200 --lip_save_dir ../outputs/lipnet/resnet18/fmnist --test_data_dir ../../data/fmnist --save_forget_dir ../../data/lipnet/resnet18/fmnist --dataset fashionMNIST --num_classes 10 --resume_lipnet
