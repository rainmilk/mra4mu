import shutil

from configs import settings
import os
import numpy as np
import torch


def sample_class_forget_data(train_data, train_labels, classes, forget_ratio):
    """按比例从每个类别中均衡抽取样本"""
    df_idx = []
    idx_retain = np.ones(len(train_labels), dtype=bool)

    for c in classes:
        idx = np.where(train_labels == c)[0]
        forget_idx = np.random.choice(idx, int(forget_ratio * len(idx)), replace=False)
        df_idx += list(forget_idx)

    idx_retain[df_idx] = False
    idx_forget = ~idx_retain
    return train_data[idx_retain], train_labels[idx_retain], train_data[idx_forget], train_labels[idx_forget]


def split_data(dataset_name, train_dataset, test_dataset, classes, forget_ratio=0.5):
    rawcase = None
    train_data_path = settings.get_dataset_path(dataset_name, rawcase, "train_data")
    train_label_path = settings.get_dataset_path(dataset_name, rawcase, "train_label")
    test_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_label")
    forget_data_path = settings.get_dataset_path(dataset_name, rawcase, "forget_data")
    forget__label_path = settings.get_dataset_path(dataset_name, rawcase, "forget_label")
    retain_data_path = settings.get_dataset_path(dataset_name, rawcase, "retain_data")
    retain_label_path = settings.get_dataset_path(dataset_name, rawcase, "retain_label")

    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)

    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)

    # 构建类均衡的 D_0 和 D_inc_0
    retain_data, retain_labels, forget_data, forget_labels = sample_class_forget_data(train_data, train_labels,
        classes, forget_ratio=forget_ratio
    )

    subdir = os.path.dirname(train_data_path)
    os.makedirs(subdir, exist_ok=True)

    # 保存训练数据集
    np.save(train_data_path, train_data)
    np.save(train_label_path, train_labels)

    # 保存测试数据集
    np.save(test_data_path, test_data)
    np.save(test_label_path, test_labels)

    np.save(forget_data_path, forget_data)
    np.save(forget__label_path, forget_labels)

    np.save(retain_data_path, retain_data)
    np.save(retain_label_path, retain_labels)
