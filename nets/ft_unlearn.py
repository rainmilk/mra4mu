import os

from torchvision import models
from torch import nn
import torch
import time
import numpy as np
from torch.utils.data import Dataset

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import utils

import arg_parser
from train import get_loader
from models.VGG_LTH import vgg16_bn_lth


def train(train_loader, model, model_path, args):
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
            #target = target.cuda()
            target = target.long().cuda()

            # compute output
            output_clean = model(image)

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

            # if (i + 1) % 10 == 0:
            #     end = time.time()
            #     print(
            #         "Epoch: [{0}][{1}/{2}]\t"
            #         "Epoch: [{0}][{1}/{2}]\t"
            #         "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            #         "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
            #         "Time {3:.2f}".format(
            #             epoch, i, len(train_loader), end - start, loss=losses, top1=top1
            #         )
            #     )
            #     start = time.time()

        # print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    state = {"state_dict": model.state_dict()}
    torch.save(state, model_path)

    return top1.avg


def test(test_loader, model):
    model.eval()
    outputs = []

    for i, (image, target) in enumerate(test_loader):
        image = image.cuda()

        output = model(image)

        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        outputs.append(output)

    return np.concatenate(outputs, axis=0)


class CustomDataset(Dataset):
    def __init__(self, data_path, label_path):
        data = np.load(data_path).astype(np.float32)
        data = np.transpose(data, [0, 3, 1, 2])
        self.data = data / 255
        self.label = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def print_acc(test_preds, forget_preds):
    test_label = np.load(os.path.join(args.test_data_dir, 'test_label.npy'))
    test_acc_unlearn = np.mean(test_label == test_preds)

    forget_label = np.load(os.path.join(args.test_data_dir, 'forget_label.npy'))
    forget_acc_unlearn = np.mean(forget_preds == forget_label)

    print('test_acc: %.2f, forget_acc: %.2f' % (test_acc_unlearn * 100, forget_acc_unlearn * 100))

    label_list = sorted(list(set(forget_label)))
    for label in label_list:
        cls_index = forget_label == label
        forget_acc_unlearn_cls = np.mean(forget_preds[cls_index] == forget_label[cls_index])
        print('label: %s, forget_acc: %.2f' % (label, forget_acc_unlearn_cls*100))

    return forget_acc_unlearn


def main():
    test_loader = get_loader('test', args.test_data_dir, args.batch_size, args.dataset)
    forget_loader = get_loader('forget', args.test_data_dir, args.batch_size, args.dataset)

    # load unlearn model
    if args.arch == 'resnet18':
        unlearn_model = models.resnet18(pretrained=False, num_classes=args.num_classes)
        if args.dataset == 'fashionMNIST':
            unlearn_model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    elif args.arch == 'vgg16_bn_lth':
        unlearn_model = vgg16_bn_lth(num_classes=args.num_classes)
    model_path = os.path.join(args.save_dir, args.unlearn + 'checkpoint.pth.tar')
    checkpoint = torch.load(model_path)
    unlearn_model.load_state_dict(checkpoint["state_dict"], strict=False)
    unlearn_model.cuda()

    # load lip foreget predicts
    lip_forget_pred = np.load(os.path.join(args.save_forget_dir, 'forget_lip_pred.npy'))

    print('Before fine-tune:')
    print('unlearn model: %s, dataset: %s' % (args.unlearn, args.dataset))

    test_preds = test(test_loader, unlearn_model)
    forget_preds = test(forget_loader, unlearn_model)
    forget_acc_before = print_acc(test_preds, forget_preds)

    if args.finetune_unlearn:
        print('Finetuning...')
        top_forget_acc = forget_acc_before
        early_stop_num = 0

        for i in range(100):
            print('-----finetune iterate : %d -----' % (i+1))

            # forget lip predicts & unlearn predict
            rnd_idx = np.random.choice(len(forget_loader.dataset.label), 100)
            inter_index = lip_forget_pred == forget_preds
            if sum(inter_index) < 100:
                inter_index[rnd_idx] = True
            forget_inter_loader = get_loader('forget_inter', args.test_data_dir, args.batch_size, args.dataset, inter_index, True)

            train(forget_inter_loader, unlearn_model, model_path, args)

            # print('-----------------after train-----------------------')
            test_preds = test(test_loader, unlearn_model)
            forget_preds = test(forget_loader, unlearn_model)
            forget_acc = print_acc(test_preds, forget_preds)
            if top_forget_acc >= forget_acc:
                early_stop_num += 1
            else:
                top_forget_acc = forget_acc
                early_stop_num = 0

            if early_stop_num == 7:
                break


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main()
