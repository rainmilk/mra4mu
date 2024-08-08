import numpy as np
import os
import torch
import pandas as pd

def get_acc(output_root, data_name, backbone_name, model_names, eval_result_file):
    data_path = backbone_name + '_' + data_name
    output_path = os.path.join(output_root, data_path)

    eval_results = []

    for model_name in model_names:
        if backbone_name == 'vgg16' and model_name == 'FF':
            eval_results.append([0, 0, 0, 0, 0])
            continue
        eval_path = os.path.join(output_path, model_name)
        model_true_name = model_name
        if model_name == 'FF':
            model_true_name = 'fisher'
        elif model_name == 'IU':
            model_true_name = 'wfisher'
        eval_file = model_true_name + eval_result_file
        eval_result_path = os.path.join(eval_path, eval_file)
        if not os.path.exists(eval_result_path):
            print(eval_result_path)
            print('eval file does not!')
            break

        eval_result = torch.load(eval_result_path)
        new_accuracy = eval_result['accuracy']
        eval_results.append(new_accuracy)

    return eval_results


if __name__ == "__main__":
    # path = '../outputs'
    # path = '/nvme/szh/data/3ai/lips/08-07-after-ablation-2/outputs'
    path = '/nvme/szh/data/3ai/lips/outputs'
    unlearn_model_names = ['retrain', 'FT', 'FF', 'GA', 'IU', 'FT_prune']
    data_list = ['cifar10', 'cifar100', 'tinyimg', 'fmnist']
    backbones = ['resnet18', 'vgg16']
    eval_all = {}

    eval_file = 'eval_result.pth.tar'
    # eval_file = 'eval_result_ft.pth.tar'
    for data in data_list:
        for backbone in backbones:
            eval_result = get_acc(path, data, backbone, unlearn_model_names, eval_file)
            data_key = data + '_' + backbone
            eval_all[data_key] = eval_result

    df_eval = pd.DataFrame.from_dict(eval_all)
    df_eval.index = unlearn_model_names
    df_eval.to_csv('unlearn_acc_before_0808-1320.csv')
    print('save acc done!')
