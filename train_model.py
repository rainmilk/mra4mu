import os
import shutil
import warnings
import numpy as np

import torch
from nets.custom_model import ClassifierWrapper, load_custom_model
from configs import settings
from train_test_utils import train_model
from arg_parser import parse_args

def get_num_of_classes(dataset_name):
    # 根据 dataset_name 设置分类类别数
    if dataset_name == "cifar-10":
        num_classes = 10
    elif dataset_name == "pet-37":
        num_classes = 37
    elif dataset_name == "cifar-100":
        num_classes = 100
    elif dataset_name == "food-101":
        num_classes = 101
    elif dataset_name == "flower-102":
        num_classes = 102
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return num_classes


def load_dataset(file_path, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101, pet-37, flower-102)
    :param file_name: 数据文件名
    :param is_data: 是否为数据文件（True 表示数据文件，False 表示标签文件）
    :return: PyTorch 张量格式的数据
    """
    data = np.load(file_path)

    if is_data:
        # 对于数据文件，转换为 float32 类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # 对于标签文件，转换为 long 类型
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


def train_step(
    args,
    writer=None,
):
    """
    根据步骤训练模型
    :param step: 要执行的步骤（0, 1, 2, ...）
    :param subdir: 数据子目录路径
    :param ckpt_subdir: 模型检查点子目录路径
    :param output_dir: 模型保存目录
    :param dataset_name: 使用的数据集类型（cifar-10 或 cifar-100）
    :param load_model_path: 指定加载的模型路径（可选）
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    """
    warnings.filterwarnings("ignore")

    dataset_name = args.dataset
    num_classes = get_num_of_classes(dataset_name)

    print(f"数据集类型: {dataset_name}")
    print(
        f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}"
    )

    model_name = args.model
    train_mode = args.train_mode
    
    uni_name = args.uni_name

    test_data = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_data")
    )
    test_labels = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_label"), is_data=False
    )

    case = None if train_mode in ["train", "pretrain"] else settings.get_case(args.forget_ratio)

    model_path = settings.get_ckpt_path(
        dataset_name, case, model_name, train_mode)

    if uni_name is None:
        train_data = np.load(
            settings.get_dataset_path(dataset_name, case, f"{train_mode}_data")
        )
        train_labels = np.load(
            settings.get_dataset_path(dataset_name, case, f"{train_mode}_label")
        )

        load_pretrained = True
        model_p0 = load_custom_model(model_name, num_classes, load_pretrained=load_pretrained)
        model_p0 = ClassifierWrapper(model_p0, num_classes)

        print(f"Train on ({dataset_name})...")

        model_p0 = train_model(
            model_p0,
            num_classes,
            train_data,
            train_labels,
            test_data,
            test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            dataset_name=args.dataset,
            writer=writer,
        )
        subdir = os.path.dirname(model_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_p0.state_dict(), model_path)
        print(f"Model saves to {model_path}")


def main():
    args = parse_args()

    writer = None
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/experiment")

    train_step(
        args,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":

    main()
