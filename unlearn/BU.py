import sys
import time

import torch

import utils
from train_test_utils import test_model

from .impl import iterative_unlearn

sys.path.append(".")
from utils import get_x_y_from_data_dict

import torch.nn as nn
from copy import deepcopy
import math
from unlearn.agents.adv import FGSM

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def find_adjacent_cls(adv_agent, model, x, y):
    x_adv = adv_agent.perturb(x, y)
    adv_logits = model(x_adv)
    adv_pred = torch.argmax(adv_logits.data, 1)
    return adv_pred, x_adv

@iterative_unlearn
def BU(data_loaders, model, criterion, optimizer, epoch, args):
    train_loader = data_loaders["forget"]
    print(len(train_loader))
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    adv_agent = FGSM(deepcopy(model), bound=0.5, norm=False, random_start=True, device='cuda')

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            x, y = get_x_y_from_data_dict(data, device)
            adv_pred, x_adv = find_adjacent_cls(adv_agent, model, x, y)
            adv_y = torch.argmax(model(x_adv), dim=1).detach().cuda()
            
            image, target = x, y
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)

            # loss = -criterion(output_clean, target)
            loss = criterion(output_clean, adv_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # image = image.cuda()
            # target = target.cuda()
            x = image.cuda()
            y = target.cuda()
            adv_pred, x_adv = find_adjacent_cls(adv_agent, model, x, y)
            adv_y = torch.argmax(model(x_adv), dim=1).detach().cuda()
            
            image, target = x, y

            # compute output
            output_clean = model(image)
            # loss = -args.alpha * criterion(output_clean, target)
            loss = args.alpha * criterion(output_clean, adv_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    test_loader = data_loaders["test"]
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    test_model(model, test_loader, criterion, device, epoch)

    return top1.avg