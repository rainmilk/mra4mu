import argparse

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

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

    forget_cls = settings.forget_classes_dict[args.dataset]

    loaded_model = load_custom_model(args.model, num_classes, load_pretrained=False)
    model = ClassifierWrapper(loaded_model, num_classes)
    model.to(device)


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
            model_suffix="restore",
            unique_name=uni_name,
        )
        print(f"Loading model from {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        predicts, probs, embedding, labels = model_forward(
            forget_loader, model, device, output_embedding=True, output_targets=True
        )

        show_tsne(embedding, labels)
        show_conf_mt(labels, predicts, forget_cls)
        acc, cls_acc = evals_classification(labels, predicts)

        model_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            model_case,
            args.model,
            model_suffix="ul",
            unique_name=uni_name,
        )
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        ul_predicts, ul_probs, ul_embedding, ul_labels = model_forward(
            forget_loader, model, device, output_embedding=True, output_targets=True
        )
        ul_acc, ul_cls_acc = evals_classification(ul_labels, ul_predicts)

        show_tsne(ul_embedding, ul_labels)
        show_conf_mt(ul_labels, ul_predicts, forget_cls)
        show_bars(ul_cls_acc[1], cls_acc[1], forget_cls)


def evals_classification(y_true, y_pred):
    eval_results = []

    # global acc
    global_acc = np.mean(y_true == y_pred)
    global_acc.item()

    # class acc
    label_list = sorted(list(set(y_true)))
    for label in label_list:
        cls_index = y_true == label
        class_acc = np.mean(y_pred[cls_index] == y_true[cls_index])
        eval_results.append(class_acc.item())

    return global_acc.item(), (label_list, eval_results)


def show_bars(bar_data_front, bar_data_back, forget_cls):
    x_labels = [f"C{y}" for y in forget_cls]
    df = pd.DataFrame({"Accuracy": bar_data_back, "Forget Classes": x_labels})
    sn.barplot(df, x="Forget Classes", y="Accuracy", color="blue")
    df = pd.DataFrame({"Accuracy": bar_data_front, "Forget Classes": x_labels})
    sn.barplot(df, x="Forget Classes", y="Accuracy", color="red")
    plt.show()


def show_conf_mt(y_true, y_pred, forget_cls):
    y_true = [y if y in forget_cls else -1 for y in y_true]
    y_pred = [y if y in forget_cls else -1 for y in y_pred]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    tick_labels = ["Other"] + [f"C{y}" for y in forget_cls]
    sn.heatmap(cm, xticklabels=tick_labels, yticklabels=tick_labels, annot=True, fmt='.2f', cbar=False)

    plt.show()


def show_tsne(embeddings, labels):
    # Apply t-SNE using MulticoreTSNE for speedup
    tsne_data = TSNE(n_components=2).fit_transform(embeddings)
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data,
                           columns=("x", "y", "label"))

    # Plotting the result of tsne
    sn.scatterplot(data=tsne_df, x='x', y='y',
                   hue='label', palette="bright")
    # get current axes
    ax = plt.gca()
    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # hide y-axis
    ax.get_yaxis().set_visible(False)

    plt.savefig("tsne.pdf")
    plt.show()


if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
