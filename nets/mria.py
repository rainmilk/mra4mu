import os

from torchvision import models
from torch import nn
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .dataset import MixupDataset, get_dataset_loader, NormalizeDataset
from .optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from configs import settings
import logging

import utils
import arg_parser
from nets.lip import SimpleLipNet


def sharpen(probs, temperature=1.0, axis=-1):
    probs = probs ** (1.0 / temperature)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def mixup_img(images, labels=None, nb_mixup=1, alpha=0.75):
    results_images = []
    results_labels = []

    nb_images = len(images)

    for i in range(nb_mixup):
        lbd = np.random.beta(alpha, alpha, size=nb_images)
        idx = lbd < 0.5
        lbd[idx] = 1 - lbd[idx]
        perm_idx = torch.randperm(nb_images)
        mixed_img = lbd * images + (1 - lbd) * images[perm_idx]
        results_images.append(mixed_img)
        if labels is not None:
            mixed_labels = lbd * labels + (1 - lbd) * labels[perm_idx]
            results_labels.append(mixed_labels)

    return results_images if labels is not None else (results_images, results_labels)


def mixup(images1, labels1, images2, labels2, alpha=0.75):
    nb_images = len(images1)
    lbd = np.random.beta(alpha, alpha, nb_images)
    rnd_idx = np.random.randint(len(images2), size=nb_images)
    mixed_img = lbd * images1 + (1 - lbd) * images2[rnd_idx]
    mixied_label = lbd * labels1 + (1 - lbd) * labels2[rnd_idx]

    return mixed_img, mixied_label


def mria_train(args):
    num_classes = settings.num_classes_dict[args.dataset]
    # kwargs = parse_kwargs(args.kwargs)
    case = settings.get_case(args.noise_ratio, args.noise_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(settings.root_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, "core_execution.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    learning_rate = getattr(args, "learning_rate", 0.001)
    weight_decay = getattr(args, "weight_decay", 5e-4)
    optimizer_type = getattr(args, "optimizer", "adam")
    num_epochs = getattr(args, "num_epochs", 50)
    uni_name = getattr(args, "uni_name", None)

    working_model_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="inc_train"
    )  # model_paths["working_model_path"]
    working_model_repair_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="restore", unique_name=uni_name
    )
    working_history_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="history", unique_name=uni_name
    )
    teacher_model_path = settings.get_ckpt_path(
        args.dataset, "pretrain", args.model, model_suffix="pretrain"
    )
    teacher_model_repair_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="teacher_restore", unique_name=uni_name,
    )
    teacher_history_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="teacher_history", unique_name=uni_name,
    )

    mean, std = None, None

    # 2. load model
    # (1) load working model
    model_infer = load_custom_model(args.model, num_classes, load_pretrained=False)

    model_infer = ClassifierWrapper(model_infer, num_classes)

    optimizer, lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        model_infer.parameters(),
        num_epochs,
        learning_rate,
        weight_decay,
    )

    working_criterion = nn.CrossEntropyLoss()

    backbone = load_custom_model(args.model, num_classes)
    model_ul = ClassifierWrapper(backbone, num_classes)
    top1 = utils.AverageMeter()

    # switch to train mode
    model_ul.eval()
    model_infer.train()

    losses = utils.AverageMeter()
    loss_fn = nn.CrossEntropyLoss()

    # Warmup Stage
    for epoch in range(args.warmup_epochs):
        for i, (image, target) in enumerate(data_loader):
            image = image.cuda()

            # predictions on data augmentation
            image_mixup = mixup_img(image)[0]
            pred_ul = model_ul(image_mixup)
            pred_ul = sharpen(pred_ul, temperature=args.temperature)

            pred_infer = model_infer(image_mixup)

            loss = loss_fn(pred_infer, pred_ul)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Alignment Stage
    nb_mixup = args.mixup_samples
    for epoch in range(args.warmup_epochs):
        for i, (image, target) in enumerate(data_loader):
            image = image.cuda()
            pred_ul_origin = model_ul(image)
            label_ul_origin = torch.argmax(pred_ul_origin, dim=-1)
            pred_infer_origin = model_infer(image)
            label_infer_origin = torch.argmax(pred_infer_origin, dim=-1)

            # predictions on data augmentation
            image_mixup = mixup_img(image, nb_mixup=nb_mixup)
            pred_list = []
            # compute output
            for im in image_mixup:
                pred_list.append(model_ul(im))

            pred_augs = torch.stack(pred_list)
            pred_aug_mean = torch.mean(pred_augs, dim=-1)
            label_pred_aug = torch.argmax(pred_aug_mean, dim=-1)

            agree_idx = label_pred_aug == label_infer_origin
            disagree_idx = ~agree_idx
            nb_agree, nb_disagree = torch.count_nonzero(agree_idx), torch.count_nonzero(disagree_idx)

            if nb_disagree > 0:
                image_disagree = image[disagree_idx]
                pred_disagree_mix = pred_infer_origin[disagree_idx]
                pred_disagree_mix = sharpen(pred_disagree_mix, temperature=args.temperature)

                if nb_agree > 0:
                    image_agree = image[agree_idx]
                    pred_agree_mix = (pred_infer_origin[agree_idx] + pred_ul_origin[agree_idx]) / 2
                    pred_agree_mix = sharpen(pred_agree_mix, temperature=args.temperature)
                else:
                    image_agree = image_disagree
                    pred_agree_mix = pred_disagree_mix

                img_mix, label_mix = mixup(image_disagree, pred_disagree_mix,
                                           image_agree, pred_agree_mix, args.temperature)

                pred_infer = model_infer(img_mix)
                loss = loss_fn(pred_infer, label_mix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # if (i + 1) % 10 == 0:
            #     end = time.time()
            #     print(
            #         "Epoch: [{0}][{1}/{2}]\t"
            #         "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            #         "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
            #         "Time {3:.2f}".format(
            #             epoch, i, len(train_loader), end - start, loss=losses, top1=top1
            #         )
            #     )
            #     start = time.time()

        # print("lip net train_accuracy {top1.avg:.3f}".format(top1=top1))

    state = {"state_dict": model.state_dict()}
    # save_path = os.path.join(args.lip_save_dir, ckpt_name)
    torch.save(state, model_path)

    return top1.avg


def mria_test(test_loader, model):
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



def get_loader_by_data(
    loader_name,
    batch_size,
    dataset_name,
    data,
    label,
    inter_index,
    one_channel=False,
    fit_embedding=None,
    query_embedding=None,
    label_true=None,
    add_data_all=None,
    add_label_all=None,
    add_inter_index=None,
    shuffle=False,
):
    dataset = None
    if loader_name == "inter":
        # inter data
        inter_data = data[inter_index]
        inter_label = label[inter_index]
        inter_label_true = label_true[inter_index]

        # add lip sample
        # len_label = len(label)
        # len_inter = sum(inter_index)
        # test_idx = np.random.choice(len_label, len_inter)
        # inter_index[test_idx] = True

        # check
        label_all_acc = np.mean(label == label_true)
        label_inter_acc = np.mean(inter_label == inter_label_true)
        print(
            "forget data unlearn acc: ",
            round(label_all_acc * 100, 2),
            "forget data alignment acc: ",
            round(label_inter_acc * 100, 2),
        )

        dataset = CustomDataset(inter_data, inter_label, dataset_name, one_channel)
    elif loader_name == "inter_and":
        # add random forget data from unlearn model
        random_idx = np.random.choice(len(data), sum(inter_index)//10)
        inter_index[random_idx] = True

        # inter data
        inter_data = data[inter_index]
        inter_label = label[inter_index]
        inter_label_true = label_true[inter_index]

        # check
        label_all_acc = np.mean(label == label_true)
        label_inter_acc = np.mean(inter_label == inter_label_true)
        print(
            "forget data unlearn acc: ",
            round(label_all_acc * 100, 2),
            "forget data alignment acc: ",
            round(label_inter_acc * 100, 2),
        )

        # add test data by knn
        # neigh = NearestNeighbors(n_neighbors=10)
        # neigh.fit(fit_embedding)
        # knn_index = neigh.kneighbors(query_embedding, return_distance=False)
        # knn_index = knn_index.reshape(-1)
        # print('knn index: ', len(set(knn_index)))
        # add_data = add_data_all[knn_index]
        # add_label = add_label_all[knn_index]

        # add test data by random
        # add_data_len = len(add_label_all)
        # inter_sum = sum(inter_index)
        # add_num = max(min(inter_sum, add_data_len), add_data_len // 3)
        # add_num = max((inter_sum + add_num) // batch_size, 1) * batch_size - inter_sum
        # add_idx = np.random.choice(add_data_len, add_num)
        # add_data = add_data_all[add_idx]
        # add_label = add_label_all[add_idx]

        # add test data by unlearn and lipnet inter
        # idx = np.where(add_inter_index)[0]
        # idx = np.random.choice(idx, len(inter_index) // 2)
        # add_inter_index[idx] = False
        add_data = add_data_all[add_inter_index]
        add_label = add_label_all[add_inter_index]

        inter_data = np.concatenate((inter_data, add_data), axis=0)
        inter_label = np.concatenate((inter_label, add_label), axis=0)

        data_len = inter_label.shape[0]
        data_len = data_len // batch_size * batch_size
        inter_data = inter_data[:data_len]
        inter_label = inter_label[:data_len]

        dataset = CustomDataset(inter_data, inter_label, dataset_name, one_channel)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def get_loader(
    loader_name, data_dir, batch_size, dataset_name, one_channel=False, shuffle=False
):
    dataset = None
    if loader_name == "test":
        test_data_path = os.path.join(data_dir, "test_data.npy")
        test_label_path = os.path.join(data_dir, "test_label.npy")
        test_data = np.load(test_data_path)
        test_label = np.load(test_label_path)

        dataset = CustomDataset(test_data, test_label, dataset_name, one_channel)

    elif loader_name == "forget":
        forget_data_path = os.path.join(data_dir, "forget_data.npy")
        forget_label_path = os.path.join(data_dir, "forget_label.npy")
        forget_data = np.load(forget_data_path)
        forget_label = np.load(forget_label_path)

        dataset = CustomDataset(forget_data, forget_label, dataset_name, one_channel)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def main():
    shuffle_flg = False
    if not args.resume_lipnet:
        shuffle_flg = True
    test_loader = get_loader(
        "test", args.test_data_dir, args.batch_size, args.dataset, shuffle=shuffle_flg
    )
    forget_loader = get_loader(
        "forget", args.test_data_dir, args.batch_size, args.dataset, shuffle=False
    )

    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    model = SimpleLipNet(resnet, 512, args.num_classes, [512])
    model.cuda()

    os.makedirs(args.lip_save_dir, exist_ok=True)

    if args.resume_lipnet:
        ckpt_path = os.path.join(args.lip_save_dir, "checkpoint.pth.tar")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        forget_lip_embeddings, forget_pred_lip = lip_test(forget_loader, model)

        save_dir = args.save_forget_dir
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "forget_lip_pred.npy"), forget_pred_lip)

        forget_label = np.load(os.path.join(args.test_data_dir, "forget_label.npy"))
        forget_acc_lip_all = np.sum(forget_pred_lip == forget_label) / len(forget_label)

        print(" forget_acc_all: %.2f" % (forget_acc_lip_all * 100))
    else:
        save_path = os.path.join(args.lip_save_dir, 'checkpoint.pth.tar')
        lip_train(test_loader, model, save_path, args)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main()
