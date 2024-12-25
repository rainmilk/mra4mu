import os
from torch import nn
import torch
import time
import numpy as np
from tqdm import tqdm

from datasetloader import get_dataset_loader
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from configs import settings
from train_test import model_test

import logging
import utils
import arg_parser


def sharpen(probs, temperature=1.0, axis=-1):
    probs = probs ** (1.0 / temperature)
    return probs / torch.sum(probs, axis=-1, keepdims=True)


def mixup_img(images, labels=None, nb_mixup=1, alpha=0.75, device='cpu'):
    results_images = []
    results_labels = []

    nb_images = len(images)

    for i in range(nb_mixup):
        beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        lbd = beta_dist.sample(sample_shape=(nb_images,1,1,1)).to(device)
        idx = lbd < 0.5
        lbd[idx] = 1 - lbd[idx]
        perm_idx = torch.randperm(nb_images)
        mixed_img = lbd * images + (1 - lbd) * images[perm_idx]
        results_images.append(mixed_img)
        if labels is not None:
            mixed_labels = lbd * labels + (1 - lbd) * labels[perm_idx]
            results_labels.append(mixed_labels)

    return results_images if labels is None else (results_images, results_labels)


def mixup(images1, labels1, images2, labels2, alpha=0.75, device='cpu'):
    nb_images = len(images1)
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    lbd = beta_dist.sample(sample_shape=(nb_images,)).to(device=device)
    rnd_idx = torch.randint(low=0, high=len(images2), size=(nb_images,))
    lbd_img = lbd.reshape((-1,1,1,1))
    mixed_img = lbd_img * images1 + (1 - lbd_img) * images2[rnd_idx]
    lbd_label = lbd.reshape((-1, 1))
    mixied_label = lbd_label * labels1 + (1 - lbd_label) * labels2[rnd_idx]

    return mixed_img, mixied_label


def mria_train(args):
    num_classes = settings.num_classes_dict[args.dataset]
    # kwargs = parse_kwargs(args.kwargs)
    case = settings.get_case(args.forget_ratio)
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

    ul_model_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="ul", unique_name=uni_name,
    )

    model_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="restore", unique_name=uni_name,
    )

    model_infer = load_custom_model(args.model, num_classes)
    model_infer = ClassifierWrapper(model_infer, num_classes)
    optimizer, lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        model_infer.parameters(),
        num_epochs,
        learning_rate,
        weight_decay,
    )

    backbone = load_custom_model(args.model, num_classes, load_pretrained=False)
    model_ul = ClassifierWrapper(backbone, num_classes)
    checkpoint = torch.load(ul_model_path)
    model_ul.load_state_dict(checkpoint, strict=False)

    model_ul.to(device=device)
    model_infer.to(device=device)
    # switch to train mode
    model_ul.eval()

    loss_fn = nn.CrossEntropyLoss()

    _, _, data_loader = get_dataset_loader(
        args.dataset,
        ["test", "forget"],
        [None, case],
        batch_size=args.batch_size,
        shuffle=True,
    )

    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        "forget",
        case,
        batch_size=args.batch_size,
        shuffle=True,
    )

    softmax = nn.Softmax(dim=1)

    # Warmup Stage
    for epoch in tqdm(range(args.warmup_epochs), desc="Warmup Stage"):
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            model_infer.train()
            for i, (image, target) in enumerate(data_loader):
                image = image.to(device)

                # predictions on data augmentation
                image_mixup = mixup_img(image, device=device)[0][0]
                pred_ul = model_ul(image_mixup)
                pred_ul = softmax(pred_ul)
                pred_ul = sharpen(pred_ul, temperature=args.temperature)

                pred_infer = model_infer(image_mixup)

                loss = loss_fn(pred_infer, pred_ul)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        model_test(forget_loader, model_infer, device)

    # Alignment Stage
    nb_mixup = args.mixup_samples

    for epoch in tqdm(range(args.epochs), desc="Alignment Stage"):
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for i, (image, target) in enumerate(data_loader):
                image = image.cuda()
                pred_ul_origin = model_ul(image)
                pred_ul_origin = softmax(pred_ul_origin)
                label_ul_origin = torch.argmax(pred_ul_origin, dim=-1)
                pred_infer_origin = model_infer(image)
                pred_infer_origin = softmax(pred_infer_origin)
                label_infer_origin = torch.argmax(pred_infer_origin, dim=-1)

                # predictions on data augmentation
                image_mixup = mixup_img(image, nb_mixup=nb_mixup, device=device)
                pred_list = []
                # compute output
                for im in image_mixup:
                    pred_prob = model_ul(im)
                    pred_prob = softmax(pred_prob)
                    pred_list.append(pred_prob)

                pred_augs = torch.stack(pred_list)
                pred_aug_mean = torch.mean(pred_augs, dim=0)
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
                                               image_agree, pred_agree_mix, args.temperature, device=device)

                    pred_infer = model_infer(img_mix)
                    loss = loss_fn(pred_infer, label_mix)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)

        model_test(forget_loader, model_infer, device)

    state = {"state_dict": model_infer.state_dict()}
    # save_path = os.path.join(args.lip_save_dir, ckpt_name)
    torch.save(state, model_save_path)

    return model_infer


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



if __name__ == "__main__":
    args = arg_parser.parse_args()
    mria_train(args)
