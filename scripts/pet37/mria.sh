# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 18.4  after: teacher 53.94  student 52.16
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# RL
# before: 10.0  after: teacher 80.81  student 85.46
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# GA
# before: 22.4  after: teacher 85.81  student 85.35
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# IU
# before: 20.0  after: teacher 86.73  student 84.89
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# BU
# before: 8.4   after: teacher 87.45  student 85.23
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# FT_l1
# before: 33.6  after: teacher 53.56  student 62.72
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# SalUn
# before: 12.8  after: teacher 87.65  student 86.25
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64