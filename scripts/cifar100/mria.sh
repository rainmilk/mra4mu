# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 11.12  after: teacher 31.57  student 16.32
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# RL
# before: 6.72   after: teacher 64.22  student 25.23
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# GA
# before: 21.92  after: teacher 59.79  student 24.52
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# IU
# before: 7.28   after: teacher 62.38  student 17.6
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# BU
# before: 26.56  after: teacher 62.83  student 25.34
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# FT_l1
# before: 31.92  after: teacher 48.91  student 29.85
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128
# SalUn
# before: 21.84  after: teacher 53.61  student 24.42
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 128