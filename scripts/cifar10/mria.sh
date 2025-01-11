# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 34.97  after: teacher 45.64   student 39.33
# python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# FT_l1
# before: 8.76   after: teacher 7.03    student 10.94
# python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256


# RL
# before: 25.76  after: teacher 38.34   student 45.17
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256
# GA
# before: 37.27  after: teacher 54.64  student 51.03
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 4e-4 --lr_student 4e-4 --batch_size 256
# IU
# before: 9.69   after: teacher 80.48    student 56.63
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256
# BU
# before: 25.12   after: teacher 83.11   student 67.78
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256
# SalUn
# before: 28.09  after: teacher 91.26   student 80.12
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256
# fisher
# before: 15.19  after: teacher 69.54   student 61.24
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256
# GA_l1
# before: 28.62  after: teacher 43.28   student 42.3
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 4e-4 --lr_student 4e-4 --batch_size 256
# UNSC
# before: 10.29  after: teacher 80.87   student 59.61
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 2 --align_epochs 3 --distill_epochs 8 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256