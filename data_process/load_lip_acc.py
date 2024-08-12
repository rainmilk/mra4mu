import numpy as np
import os
import torch
import pandas as pd


def get_acc(output_root, data_name, backbone_name, model_names, eval_result_file):
    data_path = backbone_name + "_" + data_name
    output_path = os.path.join(output_root, data_path)

    eval_results = []

    for model_name in model_names:
        if backbone_name == "vgg16" and model_name == "FF":
            eval_results.append([0, 0, 0, 0, 0])
            continue
        eval_path = os.path.join(output_path, model_name)
        eval_result_path = os.path.join(eval_path, eval_result_file)
        if not os.path.exists(eval_result_path):
            print(eval_result_path)
            print("eval file does not!")
            break

        eval_result = torch.load(eval_result_path)
        new_accuracy = eval_result["accuracy"]
        eval_results.append(new_accuracy)

    return eval_results


if __name__ == "__main__":

    # For normal experiments
    # path = "/nvme/szh/data/3ai/lips/outputs-after-05-1725"

    # Attack model before fine tune
    # eval_file_attack_before = "lipnet_eval_result.pth.tar"

    # Attack model after fine tune
    # eval_file_attack_after = "lipnet_eval_result_ft.pth.tar"
    # csv_file_attack_after = "lipnet_acc_after.csv"
    # csv_file_attack_before = "lipnet_acc_before.csv"

    
    # For Ablaition experiments
    # Input data path
    path = '/nvme/szh/data/3ai/lips/outputs-after-step05-ablation-2-0812-1015'
    # Attack model after fine tune
    eval_file_attack_ablation_2 = "lipnet_eval_result_ft.pth.tar"
    # CSV file name
    csv_file_attack_after = "lip_acc_after_ablation_2.csv"


    unlearn_model_names = ["retrain", "FT", "FF", "GA", "IU", "FT_prune"]
    data_list = ["cifar10", "cifar100", "tinyimg", "fmnist"]
    backbones = ["resnet18", "vgg16"]
    eval_all = {}

    # Before fine tune
    # (eval_file, csv_file) = (eval_file_attack_before, csv_file_attack_before)

    # After fine tune
    # (eval_file, csv_file) = (eval_file_attack_after, csv_file_attack_after)
    
    # Ablation 2
    (eval_file, csv_file) = (eval_file_attack_ablation_2, csv_file_attack_after)

    for data in data_list:
        for backbone in backbones:
            eval_result = get_acc(path, data, backbone, unlearn_model_names, eval_file)
            data_key = data + "_" + backbone
            eval_all[data_key] = eval_result

    df_eval = pd.DataFrame.from_dict(eval_all)
    df_eval.index = unlearn_model_names
    df_eval.to_csv(csv_file)
    print("save attack model's acc done!")
