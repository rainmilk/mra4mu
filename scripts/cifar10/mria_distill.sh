# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before:   after: teacher    student
# python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# FT_l1
# before:    after: teacher     student
# python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256

# RL
# before: 25.76  after: teacher 25.76   student 14.39
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# GA
# before: 37.27  after: teacher 37.27   student 27.71
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# IU
# before: 9.69   after: teacher 9.69    student 13.75
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# BU
# before: 25.12  after: teacher 25.12   student 19.16
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# SalUn
# before: 28.09  after: teacher 28.09  student 17.52
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# fisher
# before: 15.19  after: teacher 15.19  student 7.22
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# GA_l1
# before: 28.62  after: teacher 28.62  student 19.12
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256
# UNSC
# before: 10.29  after: teacher 10.29  student 4.53
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 1 --align_epochs 0 --distill_epochs 10 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256