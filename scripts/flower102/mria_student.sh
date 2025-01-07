# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 29.55  after: teacher 18.94  student 20.99
# python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# FT_l1
# before:  after: teacher 38.04  student 38.67
# python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32

# RL
# before: 63.77   after: teacher 63.77  student 74.43
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# GA
# before: 29.91  after: teacher 29.91  student 48.93
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# IU
# before: 46.88  after: teacher 46.88  student 57.93
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# BU
# before: 63.46   after: teacher 63.46  student 67.01
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# SalUn
# before: 45.15  after: teacher 45.15  student 59.43
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# fisher
# before: 28.26  after: teacher 28.26  student 37.1
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# GA_l1
# before: 49.01  after: teacher 49.01  student 59.35
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# UNSC
# before:   after: teacher   student
# python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 1 --align_epochs 2 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
