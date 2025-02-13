import os
import argparse
from torchvision import datasets, transforms
from configs import settings
from split_dataset import split_data
import torchvision
import numpy as np
import torch
from nets.datasetloader import BaseTensorDataset


def create_dataset_files(
    forget_classes,
    forget_ratio=0.5,
    data_ratio=0.5
):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    data_transform = transforms.Compose([weights.transforms()])

    # 加载 CIFAR-10 数据集
    dataset_name = "flower-102"
    data_dir = os.path.join(settings.root_dir, "data", dataset_name, "normal")

    os.path.join(settings.root_dir, "data/flower-102/normal"),
    train_dataset = datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=data_transform
    )
    test_dataset = datasets.Flowers102(
        root=data_dir, split="train", download=True, transform=data_transform
    )

    data, labels = zip(*train_dataset)
    data = torch.stack(data)
    labels = torch.tensor(labels)
    nb_data = len(labels)
    idx = np.random.choice(nb_data, size=round(nb_data * data_ratio))
    data = data[idx]
    labels = labels[idx]
    train_dataset = BaseTensorDataset(data, labels)

    train_labels = split_data(
        dataset_name, train_dataset, test_dataset, forget_classes, forget_ratio
    )
    results = np.unique(train_labels, return_index=True, return_counts=True)
    print(results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate FLOWER102 experimental datasets."
    )

    parser.add_argument(
        "--forget_classes",
        nargs='+',
        type=int,
        help="forget_classes",
    )

    parser.add_argument(
        "--forget_ratio", type=float, default=0.5, help="忘记比例（默认 0.5）"
    )

    parser.add_argument(
        "--data_ratio", type=float, default=0.5, help="data ratio"
    )

    args = parser.parse_args()

    create_dataset_files(
        forget_classes=args.forget_classes,
        forget_ratio=args.forget_ratio,
        data_ratio = args.data_ratio
    )


if __name__ == "__main__":
    main()
