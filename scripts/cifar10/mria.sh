# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 35.34  after: teacher 42.03 student 44.25
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# RL
# before: 20.96  after: teacher 36.17   student 38.8
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# GA
# before: 19.28  after: teacher 51.25  student 36.75
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# IU
# before: 9.13   after: teacher 66.56  student 46.43
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# BU
# before: 8.28  after: teacher 51.9   student 39.9
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# FT_l1
# before: 8.98   after: teacher 24.54  student 40.86
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# SalUn
# before: 13.02  after: teacher 83.41  student 63.43
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256