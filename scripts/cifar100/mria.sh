# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 20.72  after: teacher 46.45  student 11.36
# python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# FT_l1
# before: 27.12  after: teacher 47.96  student 15.08
# python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128

# RL
# before: 53.66   after: teacher 61.42  student 57.32
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# GA
# before: 49.45   after: teacher 60.88  student 55.62
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# IU
# before: 14.68   after: teacher 62.69  student 51.84
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# BU
# before: 49.83   after: teacher 63.32  student 57.4
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# SalUn
# before: 38.92   after: teacher 57.83  student 54.99
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# fisher
# before: 15.14   after: teacher 51.7   student 51.06
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# GA_l1
# before: 49.97   after: teacher 60.47  student 56.11
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# UNSC
# before: 26.73   after: teacher 64.7   student 54.71
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 2 --align_epochs 5 --distill_epochs 10 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256