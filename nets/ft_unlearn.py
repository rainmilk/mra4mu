import os

from torchvision import models
from torch import nn
import torch
import time
import numpy as np
from torch.utils.data import Dataset

import utils
import arg_parser
from train import get_loader


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

    # state = {"state_dict": model.state_dict()}
    # save_path = os.path.join(args.lip_save_dir, 'checkpoint.pth.tar')
    # torch.save(state, save_path)

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
    forget_acc_unlearn = np.sum(forget_preds == forget_label) / len(forget_label)

    print('test_acc: %.2f, forget_acc: %.2f' % (test_acc_unlearn * 100, forget_acc_unlearn * 100))


def main():
    test_loader = get_loader('test', args.test_data_dir, args.batch_size)
    forget_loader = get_loader('forget', args.test_data_dir, args.batch_size)

    # forget_loader_all = copy.deepcopy(forget_loader)

    # load unlearn model
    unlearn_model = models.resnet18(pretrained=False, num_classes=10)
    model_path = os.path.join(args.save_dir, args.unlearn + 'checkpoint.pth.tar')
    checkpoint = torch.load(model_path)
    unlearn_model.load_state_dict(checkpoint["state_dict"], strict=False)
    unlearn_model.cuda()

    # load lip foreget predicts
    lip_forget_pred = np.load(os.path.join(args.save_forget_dir, 'forget_lip_pred.npy'))

    test_preds = test(test_loader, unlearn_model)
    forget_preds = test(forget_loader, unlearn_model)
    print_acc(test_preds, forget_preds)

    for i in range(10):
        print('finetune iterate : %d' % (i+1))

        # forget lip predicts & unlearn predict
        rnd_idx = np.random.choice(len(forget_loader.dataset.label), 100)
        inter_index = lip_forget_pred == forget_preds
        if sum(inter_index) < 100:
            inter_index[rnd_idx] = True
        forget_inter_loader = get_loader('forget_inter', args.test_data_dir, args.batch_size, inter_index, True)

        train(forget_inter_loader, unlearn_model, args)

        print('-----------------after train-----------------------')
        test_preds = test(test_loader, unlearn_model)
        forget_preds = test(forget_loader, unlearn_model)
        print_acc(test_preds, forget_preds)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main()
