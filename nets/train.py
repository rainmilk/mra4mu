import os

from torchvision import models
from torch import nn
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import utils
import arg_parser
from lip import SimpleLipNet


def train(train_loader, model, args):
    loss_fn = nn.CrossEntropyLoss()
    top1 = utils.AverageMeter()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # switch to train mode
    model.train()

    start = time.time()

    losses = utils.AverageMeter()

    for epoch in range(1, args.epochs + 1):
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda()
            target = target.long().cuda()

            # compute output
            _, output_clean = model(image)

            loss = loss_fn(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % 10 == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    state = {"state_dict": model.state_dict()}
    save_path = os.path.join(args.lip_save_dir, 'checkpoint.pth.tar')
    torch.save(state, save_path)

    return top1.avg


def test(test_loader, model):
    model.eval()
    lip_outs, outputs = [], []

    for i, (image, target) in enumerate(test_loader):
        image = image.cuda()
        # target = target.long().cuda()

        # lipnet embedding out [batch, 512]
        lip_out, output = model(image)

        lip_outs.append(lip_out.data.cpu().numpy())
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        outputs.append(output)

    return np.concatenate(lip_outs, axis=0), np.concatenate(outputs, axis=0)


class CustomDataset(Dataset):
    def __init__(self, data, label, dataset_name):
        data = data.astype(np.float32)
        if dataset_name in ['cifar10', 'cifar100']:
            data = np.transpose(data, [0, 3, 1, 2])
            self.data = data / 255
        elif dataset_name == 'fashionMNIST':
            '''
            When training lipschitz network. Namely when executing train_lipnet.sh

            fashionMNIST数据集是单通道（灰度）图像，只有1个通道。因此，在执行forward方法时，ResNet18的第一个卷积层（期望输入3个通道）与输入图像（1个通道）不匹配。因此将fashionMNIST数据集的单通道图像转换为三通道图像。

            data[:, np.newaxis, ...] 将数据的形状从 [N, H, W] 变为 [N, 1, H, W]. 然后使用 np.repeat(data[:, np.newaxis, ...], 3, axis=1) 将数据的形状从 [N, 1, H, W] 变为 [N, 3, H, W]，即将单通道图像转换为三通道图像。
            '''
            data = np.repeat(data[:, np.newaxis, ...], 3, axis=1)
            '''
            - When training unlearning network.
            - 将数据从二维（单通道）转换为三维，增加了一个通道维度，保持图像为单通道，适用于不需要RGB输入的网络。
            data[:, np.newaxis, ...] 将数据的形状从 [N, H, W] 变为 [N, 1, H, W]。
            - 通常用于不需要RGB输入的网络，或者网络结构可以处理单通道输入，如某些自定义的或特殊的卷积神经网络。
            '''
            # data = data[:, np.newaxis, ...]
            self.data = data / 255
        elif dataset_name == 'TinyImagenet':
            self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def get_loader(loader_name, data_dir, batch_size, dataset_name, inter_index=None, shuffle=False):
    dataset = None
    if loader_name == "test":
        test_data_path = os.path.join(data_dir, 'test_data.npy')
        test_label_path = os.path.join(data_dir, 'test_label.npy')
        test_data = np.load(test_data_path)
        test_label = np.load(test_label_path)

        dataset = CustomDataset(test_data, test_label, dataset_name)

    elif loader_name == "forget":
        forget_data_path = os.path.join(data_dir, 'forget_data.npy')
        forget_label_path = os.path.join(data_dir, 'forget_label.npy')
        forget_data = np.load(forget_data_path)
        forget_label = np.load(forget_label_path)

        dataset = CustomDataset(forget_data, forget_label, dataset_name)
    elif loader_name == "forget_inter":
        forget_data_path = os.path.join(data_dir, 'forget_data.npy')
        forget_label_path = os.path.join(data_dir, 'forget_label.npy')
        forget_data = np.load(forget_data_path)
        forget_label = np.load(forget_label_path)

        forget_inter_data = forget_data[inter_index]
        forget_inter_label = forget_label[inter_index]

        dataset = CustomDataset(forget_inter_data, forget_inter_label, dataset_name)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader


def main():
    shuffle_flg = False
    if not args.resume_lipnet:
        shuffle_flg = True
    test_loader = get_loader('test', args.test_data_dir, args.batch_size, args.dataset, shuffle_flg)
    forget_loader = get_loader('forget', args.test_data_dir, args.batch_size, args.dataset, False)

    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    model = SimpleLipNet(resnet, 512, args.num_classes, [512])
    model.cuda()

    os.makedirs(args.lip_save_dir, exist_ok=True)

    if args.resume_lipnet:
        ckpt_path = os.path.join(args.lip_save_dir, 'checkpoint.pth.tar')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        forget_lip_embeddings, forget_pred_lip = test(forget_loader, model)

        save_dir = args.save_forget_dir
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'forget_lip_pred.npy'), forget_pred_lip)

        forget_label = np.load(os.path.join(args.test_data_dir, 'forget_label.npy'))
        forget_acc_lip_all = np.sum(forget_pred_lip == forget_label) / len(forget_label)

        print(' forget_acc_all: %.2f' % (forget_acc_lip_all * 100))
    else:
        train(test_loader, model, args)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main()
