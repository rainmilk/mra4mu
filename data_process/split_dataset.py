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
    return (train_data[idx_retain], train_labels[idx_retain],
            train_data[idx_forget], train_labels[idx_forget])


def gen_forget_cls_data(test_data, test_labels, forget_classes):
    ts_idx = []
    for c in forget_classes:
        idx = np.where(test_labels == c)[0]
        ts_idx += list(idx)
    forget_cls_data = test_data[ts_idx]
    forget_cls_labels = test_labels[ts_idx]
    return forget_cls_data, forget_cls_labels


def split_data(dataset_name, train_dataset, test_dataset, forget_classes, forget_ratio=0.5):
    rawcase = None
    train_data_path = settings.get_dataset_path(dataset_name, rawcase, "train_data")
    train_label_path = settings.get_dataset_path(dataset_name, rawcase, "train_label")
    test_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_label")

    case = settings.get_case(forget_ratio)
    forget_data_path = settings.get_dataset_path(dataset_name, case, "forget_data")
    forget_label_path = settings.get_dataset_path(dataset_name, case, "forget_label")
    retain_data_path = settings.get_dataset_path(dataset_name, case, "retain_data")
    retain_label_path = settings.get_dataset_path(dataset_name, case, "retain_label")
    forget_cls_data_path = settings.get_dataset_path(dataset_name, case, "forget_cls_data")
    forget_cls_label_path = settings.get_dataset_path(dataset_name, case, "forget_cls_label")

    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)

    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)



    # 构建类均衡的 D_0 和 D_inc_0
    retain_data, retain_labels, forget_data, forget_labels = sample_class_forget_data(
        train_data, train_labels, forget_classes, forget_ratio=forget_ratio)

    forget_cls_data, forget_cls_labels = gen_forget_cls_data(test_data, test_labels, forget_classes)

    # forget_cls_data = np.concatenate([forget_cls_data, forget_data], axis=0)
    # forget_cls_labels = np.concatenate([forget_cls_labels, forget_labels], axis=0)


    subdir = os.path.dirname(forget_data_path)
    os.makedirs(subdir, exist_ok=True)

    # 保存训练数据集
    np.save(train_data_path, train_data)
    np.save(train_label_path, train_labels)

    # 保存测试数据集
    np.save(test_data_path, test_data)
    np.save(test_label_path, test_labels)

    np.save(forget_data_path, forget_data)
    np.save(forget_label_path, forget_labels)

    np.save(forget_cls_data_path, forget_cls_data)
    np.save(forget_cls_label_path, forget_cls_labels)

    np.save(retain_data_path, retain_data)
    np.save(retain_label_path, retain_labels)

    return train_labels


if __name__ == '__main__':
    dataset = 'flower-102'
    forget_classes = [50, 72, 76, 88, 93]
    case = settings.get_case(0.5)
    test_data_path = settings.get_dataset_path(dataset, None, "test_data")
    test_label_path = settings.get_dataset_path(dataset, None, "test_label")
    forget_cls_data_path = settings.get_dataset_path(dataset, case, "forget_cls_data")
    forget_cls_label_path = settings.get_dataset_path(dataset, case, "forget_cls_label")
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)
    forget_cls_data, forget_cls_labels = gen_forget_cls_data(test_data, test_labels, forget_classes)
    np.save(forget_cls_label_path, forget_cls_labels)


