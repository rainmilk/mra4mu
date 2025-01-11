# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 18.4  after: teacher 53.94  student 52.16
# python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# FT_l1
# before: 33.6  after: teacher 53.56  student 62.72
# python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64

# RL
# before: 72.7   after: teacher 85.05   student 85.17
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# GA
# before: 70.68  after: teacher 85.71  student 85.53
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# IU
# before: 59.5   after: teacher 85.97  student 85.81
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# BU
# before: 60.98  after: teacher 85.84  student 86.25
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# SalUn
# before: 66.9   after: teacher 86.35  student 87.17
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# fisher
# before: 25.49  after: teacher 77.41  student 72.83
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# GA_l1
# before: 70.5   after: teacher 84.38  student 84.15
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# UNSC
# before: 73.92  after: teacher 86.37  student 86.53
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
