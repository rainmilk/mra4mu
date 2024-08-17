OUTPUT_BASE_DIR="/nvme/szh/data/3ai/lips/outputs"

SAVE_DATA_BASE_DIR="/nvme/szh/data/3ai/lips/saved_data"


python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/FF --shuffle 
