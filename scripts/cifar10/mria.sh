# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 34.97  after: teacher 45.64   student 39.33
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# RL
# before: 20.49  after: teacher 36.77  student 30.99
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# GA
# before: 26.72  after: teacher 47.73  student 19.97
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# IU
# before: 9.05   after: teacher 86.35   student 17.93
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# BU
# before: 8.23   after: teacher 80.48   student 37.22
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# FT_l1
# before: 8.76   after: teacher 7.03    student 10.94
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256
# SalUn
# before: 11.69  after: teacher 91.69  student 46.99
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --batch_size 256