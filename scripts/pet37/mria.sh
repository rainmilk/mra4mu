# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 9.6   after: teacher 56.32  student 55.45
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# RL
# before: 4.8   after: teacher 80.51  student 84.56
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# GA
# before: 20.8  after: teacher 84.95  student 86.53
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# IU
# before: 15.6  after: teacher 85.0   student 84.82
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# BU
# before: 4.4   after: teacher 80.02  student 78.54
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# FT_l1
# before: 23.6  after: teacher 56.01  student 63.54
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# SalUn
# before: 9.2   after: teacher 84.03  student 84.18
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64