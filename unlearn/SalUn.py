import sys
import time

import torch

import utils
from train_test_utils import test_model

from .impl import iterative_unlearn

sys.path.append(".")
from utils import get_x_y_from_data_dict

import numpy as np
import os

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)



flag = False

# create saliency map
def save_gradient_ratio(unlearn_train_loader, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    gradients = {}
    model.eval()
    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, target) in enumerate(unlearn_train_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        print(i)
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        save_path = f'./save/{args.dataset}/mask_threshold_{i}.pt'
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(hard_dict, save_path)
        # torch.save(hard_dict, f'./save/{args.dataset}/mask_threshold_{i}.pt')

@iterative_unlearn
def SalUn(data_loaders, model, criterion, optimizer, epoch, args):
    global flag
    if flag == False:
        args.unlearn_lr=0.01
        args.momentum=0.9
        args.weight_decay=5e-4
        
        unlearn_train_loader = data_loaders["forget"]
        save_gradient_ratio(unlearn_train_loader, model, criterion, args)
        flag = True
        
    remain_class = np.setdiff1d(np.arange(args.num_classes), args.class_to_replace)
    
    threshold = 0.8
    mask = torch.load(f'./save/{args.dataset}/mask_threshold_{threshold}.pt')
    
    train_loader = data_loaders["forget"]
    print(len(train_loader))
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            # image, target = get_x_y_from_data_dict(data, device)
            x, _ = get_x_y_from_data_dict(data, device)
            y = torch.from_numpy(np.random.choice(remain_class, size=x.shape[0])).cuda()
            
            image, target = x, y
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)

            # loss = -criterion(output_clean, target)
            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
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

            image = image.cuda()
            # target = target.cuda()
            target = torch.from_numpy(np.random.choice(remain_class, size=image.shape[0])).cuda()

            # compute output
            output_clean = model(image)
            # loss = -args.alpha * criterion(output_clean, target)
            loss = args.alpha * criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
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