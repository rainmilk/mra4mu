import time

import torch

import utils

from .impl import iterative_unlearn
from .SAM import SAM


def train(train_loader, model, criterion, optimizer, epoch, args, l1=False):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)

        loss = criterion(output_clean, target)

        loss.backward()
        optimizer.first_step(zero_grad=True)

        output_clean = model(image)

        loss = criterion(output_clean, target)

        loss.backward()
        optimizer.second_step(zero_grad=True)

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
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

    return top1.avg


@iterative_unlearn
def retrain_sam(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    base_optimizer = torch.optim.SGD
    # from SAM import SAM

    new_optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=2.0,
        adaptive=True,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, new_optimizer, epoch, args)
