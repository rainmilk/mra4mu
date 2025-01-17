import argparse
import copy

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os

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
    # todo temp code
    if args.dataset == 'flower-102':
        loaded_model_ul = load_custom_model('swin_t', num_classes, load_pretrained=False)
        ul_model = ClassifierWrapper(loaded_model_ul, num_classes)
    else:
        ul_model = copy.deepcopy(model)
    model.to(device)
    ul_model.to(device)

    _, _, pred_loader = get_dataset_loader(
        args.dataset,
        ["forget", "test"],
        [case, None],
        batch_size=args.batch_size,
        shuffle=False
    )

    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        "forget_cls",
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

        # todo temp flower-102
        if args.dataset == 'flower-102':
            model_ckpt_path = settings.get_ckpt_path(
                args.dataset,
                model_case,
                'swin_t',
                model_suffix="ul",
                unique_name=uni_name,
            )
        else:
            model_ckpt_path = settings.get_ckpt_path(
                args.dataset,
                model_case,
                args.model,
                model_suffix="ul",
                unique_name=uni_name,
            )

        checkpoint = torch.load(model_ckpt_path)
        ul_model.load_state_dict(checkpoint, strict=False)
        ul_model.eval()

        predicts, probs, embedding, labels = model_forward(
            pred_loader, model, device, output_embedding=True, output_targets=True
        )

        ul_predicts, ul_probs, ul_embedding, ul_labels = model_forward(
            pred_loader, ul_model, device, output_embedding=True, output_targets=True
        )

        title = uni_name if args.fig_title is None else args.fig_title
        samples = 100

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, args.model_suffix, "tsne")
        subdir = os.path.dirname(save_path)
        os.makedirs(subdir, exist_ok=True)
        sample_idx = np.random.choice(len(embedding), size=num_classes * samples, replace=True)
        sample_idx = np.unique(sample_idx)
        show_tsne(embedding[sample_idx], labels[sample_idx], forget_cls, title=f"MRA-{title}", save_path=save_path)

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, "ul", "tsne")
        show_tsne(ul_embedding[sample_idx], ul_labels[sample_idx], forget_cls, title=f"ULM-{title}", save_path=save_path)

        predicts, probs, embedding, labels = model_forward(
            forget_loader, model, device, output_embedding=True, output_targets=True
        )

        ul_predicts, ul_probs, ul_embedding, ul_labels = model_forward(
            forget_loader, ul_model, device, output_embedding=True, output_targets=True
        )

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, args.model_suffix, "cmt")
        show_conf_mt(labels, predicts, forget_cls, title=f"MRA-{title}", save_path=save_path)
        acc, cls_acc = evals_classification(labels, predicts)
        ul_acc, ul_cls_acc = evals_classification(ul_labels, ul_predicts)

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, "ul", "cmt")
        show_conf_mt(ul_labels, ul_predicts, forget_cls, title=f"ULM-{title}", save_path=save_path)

        #MIA
        re_mia = evals_cls_acc(labels, predicts, forget_cls)
        ul_mia = evals_cls_acc(ul_labels, ul_predicts, forget_cls)

        save_path = settings.get_visual_result_path(args.dataset, case, uni_name, args.model, args.model_suffix, "bar")
        show_bars(ul_mia, re_mia, forget_cls, title=title, save_path=save_path)


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


def evals_cls_acc(y_true, y_pred, forget_cls):
    eval_results = []

    for label in forget_cls:
        cls_index = y_true == label
        mean_acc = np.mean(y_pred[cls_index] == label)
        if np.isnan(mean_acc):
            mean_acc = 0
        eval_results.append(mean_acc)

    return eval_results


def show_bars(bar_data_front, bar_data_back, forget_cls, size=(5, 5), title=None, save_path=None):
    x_labels = [f"C{y}" for y in forget_cls]
    df1 = pd.DataFrame({"Type": "ULM", "ACC": bar_data_front, "Forgetting Classes": x_labels})
    df2 = pd.DataFrame({"Type": "MRA", "ACC": bar_data_back, "Forgetting Classes": x_labels})
    df = pd.concat([df1, df2], axis=0)

    plt.clf()
    ax = sn.barplot(df, x="Forgetting Classes", y="ACC", hue="Type", palette="PuBuGn")
    ax.bar_label(ax.containers[0], fontsize=10, fmt="%.2f")
    ax.bar_label(ax.containers[1], fontsize=10, fmt="%.2f")
    y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(y_ticks)
    ax.legend().set_title('')
    ax.figure.set_size_inches(size)

    if title is not None:
        ax.set_title(title, fontdict={'size': 16, 'weight': 'bold'})

    if save_path is not None:
        ax.figure.savefig(save_path)
    else:
        plt.show()


def show_conf_mt(y_true, y_pred, forget_cls, size=(5, 5), title=None, save_path=None):
    y_true = [y if y in forget_cls else -1 for y in y_true]
    y_pred = [y if y in forget_cls else -1 for y in y_pred]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    tick_labels = ["Other"] + [f"C{y}" for y in forget_cls]

    plt.clf()

    ax = sn.heatmap(cm, xticklabels=tick_labels, yticklabels=tick_labels, annot=True, fmt='.2f', cbar=False)
    ax.figure.set_size_inches(size)

    if title is not None:
        ax.set_title(title, fontdict={'size': 16, 'weight': 'bold'})

    if save_path is not None:
        ax.figure.savefig(save_path)
    else:
        plt.show()


def show_tsne(embeddings, labels, forget_cls, size=(5, 5), title=None, save_path=None):
    # Apply t-SNE using MulticoreTSNE for speedup
    tsne_data = TSNE(n_components=2).fit_transform(embeddings).T
    # styles = ["Forgotten" if y in forget_cls else "Others" for y in labels]
    labels = [f"C{y}" if y in forget_cls else "Others" for y in labels]
    tsne_df = pd.DataFrame({"x":tsne_data[0], "y":tsne_data[1], "Class":labels})

    # Plotting the result of tsne
    plt.clf()

    custom_order = sorted(list(set(labels)))

    ax = sn.scatterplot(data=tsne_df, x='x', y='y',
                   hue='Class', palette="muted", hue_order=custom_order)
    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    ax.legend().set_title('')
    ax.figure.set_size_inches(size)
    plt.legend(fontsize=7)

    if title is not None:
        ax.set_title(title, fontdict={'size': 16, 'weight': 'bold'})

    if save_path is not None:
        ax.figure.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
