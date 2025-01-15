# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 33.2    after: 96.0  (forget_acc)
# before: 75.39   after: teacher 85.04  student 84.3 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# GA
# before: 22.4    after: 95.2  (forget_acc)
# before: 73.97   after: teacher 84.27  student 85.99 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# IU
# before: 20.0    after: 93.6  (forget_acc)
# before: 62.2    after: teacher 83.76  student 82.75 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# BU
# before: 8.4     after: 96.0  (forget_acc)
# before: 64.57   after: teacher 84.63  student 84.19 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# SalUn
# before: 12.8    after: 94.8  (forget_acc)
# before: 70.59   after: teacher 84.87  student 82.72 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# fisher
# before: 22.4    after: 90.0  (forget_acc)
# before: 25.7    after: teacher 78.2  student 71.44 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# GA_l1
# before: 19.6    after: 92.0  (forget_acc)
# before: 73.97   after: teacher 84.14  student 83.7 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
# UNSC
# before: 30.4    after: 97.6  (forget_acc)
# before: 76.89   after: teacher 85.58  student 85.99 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64
