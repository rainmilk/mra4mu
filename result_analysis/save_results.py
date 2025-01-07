import os
import numpy as np
import pandas as pd

from configs import settings


def execute():
    result_dir = os.path.join(settings.root_dir, 'results')
    datasets = os.listdir(result_dir)
    global_result_dict = {}
    forget_result_dict = {}

    for dataset in datasets:
        global_result_dict[dataset] = {}
        forget_result_dict[dataset] = {}

        data_dir = os.path.join(result_dir, dataset)
        forget_ratios = os.listdir(data_dir)
        for forget_ratio in forget_ratios:
            global_result_dict[dataset][forget_ratio] = {}
            forget_result_dict[dataset][forget_ratio] = {}

            ratio_dir = os.path.join(data_dir, forget_ratio)
            uni_names = os.listdir(ratio_dir)
            for uni_name in uni_names:
                uni_dir = os.path.join(ratio_dir, uni_name)
                result_files = os.listdir(uni_dir)
                for result_file in result_files:
                    result_file_path = os.path.join(uni_dir, result_file)
                    # load data
                    result = np.load(result_file_path, allow_pickle=True)

                    result_type = result_file.split('_')[-2]
                    if result_type not in global_result_dict[dataset][forget_ratio].keys():
                        global_result_dict[dataset][forget_ratio][result_type] = {}
                    if uni_name not in global_result_dict[dataset][forget_ratio][result_type].keys():
                        global_result_dict[dataset][forget_ratio][result_type][uni_name] = {}

                    if result_type not in forget_result_dict[dataset][forget_ratio].keys():
                        forget_result_dict[dataset][forget_ratio][result_type] = {}
                    if uni_name not in forget_result_dict[dataset][forget_ratio][result_type].keys():
                        forget_result_dict[dataset][forget_ratio][result_type][uni_name] = {}

                    if '_global.npy' in result_file_path:
                        global_result_dict[dataset][forget_ratio][result_type][uni_name] = result
                    if '_forget.npy' in result_file_path:
                        forget_result_dict[dataset][forget_ratio][result_type][uni_name] = result

        # save df global
        csv_dir = os.path.join(settings.root_dir, 'results_csv')
        os.makedirs(csv_dir, exist_ok=True)

        for dataset in global_result_dict.keys():
            for forget_ratio in global_result_dict[dataset].keys():
                df = pd.DataFrame(global_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_global.csv' % (dataset, forget_ratio))
                df.to_csv(csv_path, index=True, header=True)

        # save df forget
        for dataset in forget_result_dict.keys():
            for forget_ratio in forget_result_dict[dataset].keys():
                df = pd.DataFrame(forget_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_forget.csv' % (dataset, forget_ratio))
                df.to_csv(csv_path, index=True, header=True)


if __name__ == "__main__":
    execute()
