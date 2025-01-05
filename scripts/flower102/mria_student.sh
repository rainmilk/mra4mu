# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 29.55  after: teacher 18.94  student 20.99
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# RL
# before: 30.36   after: teacher 50.28  student 50.2
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# GA
# before: 23.89  after: teacher 78.61  student 74.74
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# IU
# before: 29.15  after: teacher 77.19  student 74.82
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# BU
# before: 34.01   after: teacher 90.92  student 88.0
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# FT_l1
# before:  after: teacher 38.04  student 38.67
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# GA_l1
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# SalUn
# before: 26.72  after: teacher 87.61  student 83.74
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32

# UNSC
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
