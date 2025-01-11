import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from unlearn.gen_mask import save_gradient_ratio

import arg_parser
import unlearn

from nets.datasetloader import get_dataset_loader
from configs import settings
from nets.custom_model import load_custom_model, ClassifierWrapperHooker
from nets.train_test import model_test



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = arg_parser.parse_args()

    case = settings.get_case(args.forget_ratio)

    uni_name = args.uni_name
    num_classes = settings.num_classes_dict[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _, _, train_loader = get_dataset_loader(
    #     args.dataset,
    #     "train",
    #     None,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    # )

    _, _, retain_loader = get_dataset_loader(
        args.dataset,
        "retain",
        case,
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

    _, _, test_loader = get_dataset_loader(
        args.dataset,
        "forget",
        case,
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, val_loader = get_dataset_loader(
        args.dataset,
        ["test"],
        [None],
        batch_size=args.batch_size,
        shuffle=False,
    )

    load_model_path = settings.get_ckpt_path(
        args.dataset,
        None,
        args.model,
        model_suffix="train"
    )

    save_model_path = settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        model_suffix="ul",
        unique_name=uni_name,
    )

    history_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="history", unique_name=uni_name
    )

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    unlearn_method = unlearn.get_unlearn_method(uni_name)

    loaded_model = load_custom_model(
        args.model, num_classes, load_pretrained=False
    )
    model = ClassifierWrapperHooker(loaded_model, num_classes)
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    mask = None
    if args.mask_thresh > 0:
        _, _, train_loader = get_dataset_loader(
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            shuffle=True,
        )
        mask = save_gradient_ratio(train_loader, model, criterion, args)
        mask = mask[args.mask_thresh]

    # model_test(val_loader, model, device)

    model_history = unlearn_method(unlearn_data_loaders, model, criterion, args, mask)

    model_test(val_loader, model, device)

    # save model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print("model saved to:", save_model_path)

    #save history
    torch.save(model_history, history_save_path)


if __name__ == "__main__":
    main()
