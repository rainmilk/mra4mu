import os
import argparse
from torchvision import datasets, transforms
from configs import settings
from split_dataset import split_data


def create_dataset_files(
    forget_classes,
    forget_ratio=0.5,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]),
        ]
    )
    # 加载 CIFAR-10 数据集
    dataset_name = "cifar-10"
    data_dir = os.path.join(settings.root_dir, "data", dataset_name, "normal")

    os.path.join(settings.root_dir, "data/cifar-10/normal"),
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    split_data(
        dataset_name, train_dataset, test_dataset, forget_classes, forget_ratio
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 experimental datasets."
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
