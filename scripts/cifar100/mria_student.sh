# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 20.72  after: teacher 46.45  student 11.36
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 128
# RL
# before: 11.6   after: teacher 56.3   student 10.07
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128
# GA
# before: 20.08  after: teacher 58.7   student 8.3
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128
# IU
# before: 20.08   after: teacher 59.25  student 0.88
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128
# BU
# before: 25.52  after: teacher 64.27  student 7.65
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128
# FT_l1
# before: 27.12  after: teacher 47.96  student 15.08
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128
# SalUn
# before: 22.88  after: teacher 42.04  student 7.89
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128
# UNSC
#
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 1 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5  --lr_student 5e-5 --no_t_update --batch_size 128