# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 18.4  after: teacher 53.94  student 52.16
# python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# FT_l1
# before: 33.6   after: teacher 53.56  student 62.72
# python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64

# RL
# before: 72.7   after: teacher 72.7   student 72.49
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# GA
# before: 70.68  after: teacher 70.68  student 70.71
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# IU
# before: 59.5   after: teacher 59.5   student 56.72
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# BU
# before: 60.98  after: teacher 60.98  student 62.52
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# SalUn
# before: 66.9   after: teacher 66.9   student 67.9
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# fisher
# before: 25.49  after: teacher 25.49  student 25.69
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# GA_l1
# before: 70.5   after: teacher 70.5   student 69.15
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# UNSC
# before: 73.92  after: teacher 73.92  student 72.9
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
