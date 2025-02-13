import os


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

dataset_paths = {
    "cifar-10": os.path.join(root_dir, "data", "cifar-10"),
    "cifar-100": os.path.join(root_dir, "data", "cifar-100"),
    "food-101": os.path.join(root_dir, "data", "food-101"),
    "flower-102": os.path.join(root_dir, "data", "flower-102"),
    "pet-37": os.path.join(root_dir, "data", "pet-37"),
}

num_classes_dict = {
    "cifar-10": 10,
    "cifar-100": 100,
    "food-101": 101,
    "flower-102": 102,
    "pet-37": 37,
}

forget_classes_dict = {
    "cifar-10": [1,3,5,7,9],
    "cifar-100": [10,30,50,70,90],
    "food-101": [1,3,5,7,9],
    "flower-102": [50, 72, 76, 88, 93],
    "pet-37": [1, 8, 15, 21, 29],
}

cifar10_config = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}

cifar100_config = {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]}

food101_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

suffix_name = "ijcai"


def get_case(forget_ratio=0.5, suffix=suffix_name):
    return f"fr_{forget_ratio}_{suffix}"


def get_ckpt_path(dataset, case, model, model_suffix, step=None, unique_name=None):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "ckpt", dataset)
    if case is not None:
        path = os.path.join(path, case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")
    if unique_name is not None:
        path = os.path.join(path, unique_name)

    return os.path.join(path, f"{model}_{model_suffix}.pth")


def get_visual_result_path(dataset, case, unique_name, model, model_suffix, type_name):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "result_visual", dataset)
    if case is not None:
        path = os.path.join(path, case)

    return os.path.join(path, f"{unique_name}_{model}_{model_suffix}_{type_name}.pdf")


# get ckpt files for sensitivity experiment models
def get_pretrain_ckpt_path(
    dataset, case, model, model_suffix, step=None, unique_name=None
):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "ckpt", dataset, case)
    pretrain_case = "pretrain"
    path = os.path.join(path, pretrain_case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")
    if unique_name is not None:
        path = os.path.join(path, unique_name)

    return os.path.join(path, f"{model}_{model_suffix}.pth")


def get_dataset_path(dataset, case, type, step=None):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "data", dataset, "gen")
    if case is not None:
        path = os.path.join(path, case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")

    return os.path.join(path, f"{type}.npy")
