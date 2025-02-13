import os
import argparse
from torchvision import datasets, transforms
from configs import settings
from split_dataset import split_data
import torchvision
import numpy as np


def create_dataset_files(
    forget_classes,
    forget_ratio=0.5,
):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    data_transform = transforms.Compose([weights.transforms()])

    # 加载 CIFAR-10 数据集
    dataset_name = "pet-37"
    data_dir = os.path.join(settings.root_dir, "data", dataset_name, "normal")

    os.path.join(settings.root_dir, "data/pet-37/normal"),
    train_dataset = datasets.OxfordIIITPet(
        root=data_dir, download=True, transform=data_transform
    )
    test_dataset = datasets.OxfordIIITPet(
        root=data_dir, split="test", download=True, transform=data_transform
    )
    train_labels = split_data(
        dataset_name, train_dataset, test_dataset, forget_classes, forget_ratio
    )
    results = np.unique(train_labels, return_index=True, return_counts=True)
    print(results)

def main():
    parser = argparse.ArgumentParser(
        description="Generate PET37 experimental datasets."
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

    args = parser.parse_args()

    create_dataset_files(
        forget_classes=args.forget_classes,
        forget_ratio=args.forget_ratio,
    )


if __name__ == "__main__":
    main()
