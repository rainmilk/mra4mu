import argparse
import os

import torch
import numpy as np

from arg_parser import parse_args
from configs import settings
from nets.custom_model import ClassifierWrapper, load_custom_model
from nets.datasetloader import get_dataset_loader
from nets.train_test import model_forward, model_test


def execute(args):
    case = settings.get_case(args.forget_ratio)
    uni_names = args.uni_name
    uni_names = [uni_names] if uni_names is None else uni_names.split(",")
    num_classes = settings.num_classes_dict[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # todo flowers-102 student_only
    if args.dataset == 'flower-102' and args.model_suffix == 'student_only':
        loaded_model = load_custom_model('resnet18', num_classes, load_pretrained=False)
    else:
        loaded_model = load_custom_model(args.model, num_classes, load_pretrained=False)

    model = ClassifierWrapper(loaded_model, num_classes)
    model.to(device)

    _, _, pred_loader = get_dataset_loader(
        args.dataset,
        ["test", "forget"],
        [None, case],
        batch_size=args.batch_size,
        shuffle=False,
    )

    # _, _, forget_cls_loader = get_dataset_loader(
    #     args.dataset,
    #     "forget_cls",
    #     case,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    # )

    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        "forget",
        case,
        batch_size=args.batch_size,
        shuffle=False,
    )

    for uni_name in uni_names:
        print(f"Evaluating {uni_name}:")
        model_case = None if args.model_suffix in ["train", "pretrain"] else settings.get_case(args.forget_ratio)
        model_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            model_case,
            args.model,
            model_suffix=args.model_suffix,
            unique_name=uni_name,
        )
        print(f"Loading model from {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        print(f"Evaluating prediction dataset:")
        results, embedding = evals(pred_loader, model, device, args)
        # print("Results: %.4f" % results)
        print("Results: ", results)
        print(f"Evaluating forget dataset:")
        n_results, n_embedding = evals(forget_loader, model, device, args)
        # print("Results: %.4f" % results)
        print("Results: ", n_results)
        # print(f"Evaluating forget class dataset:")
        # n_results, n_embedding = evals(forget_cls_loader, model)
        # # print("Results: %.4f" % results)
        # print("Results: ", n_results)

        # save results
        root_dir = settings.root_dir
        result_dir = os.path.join(root_dir, 'results', args.dataset, case, args.uni_name, )
        os.makedirs(result_dir, exist_ok=True)
        forget_file = args.model+'_'+args.model_suffix+'_forget.npy'
        global_file = args.model+'_'+args.model_suffix+'_global.npy'

        np.save(os.path.join(result_dir, forget_file), n_results)
        np.save(os.path.join(result_dir, global_file), results)


def evals(data_loader, model, device="cuda", args=None):
    eval_results = {}

    predicts, probs, embedding, labels = model_forward(
        data_loader, model, device, output_embedding=True, output_targets=True
    )

    # global acc
    global_acc = np.mean(predicts == labels)
    eval_results["global"] = global_acc.item()

    # class acc
    label_list = sorted(list(set(labels)))
    for label in label_list:
        cls_index = labels == label
        class_acc = np.mean(predicts[cls_index] == labels[cls_index])
        eval_results["label_" + str(label.item())] = class_acc.item()

    return eval_results, embedding


if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
