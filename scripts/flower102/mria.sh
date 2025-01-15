# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 14.98   after: 98.79  (forget_acc)
# before: 75.59   after: teacher 91.47  student 86.76 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# GA
# before: 23.89   after: 70.85  (forget_acc)
# before: 31.37   after: teacher 77.16  student 69.9 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# IU
# before: 29.15   after: 72.47  (forget_acc)
# before: 51.18   after: teacher 82.94  student 78.14 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# BU
# before: 34.01   after: 97.17  (forget_acc)
# before: 70.59   after: teacher 88.92  student 84.22 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# SalUn
# before: 26.72   after: 91.5  (forget_acc)
# before: 49.61   after: teacher 85.69  student 76.27 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# fisher
# before: 23.48   after: 48.18  (forget_acc)
# before: 29.41   after: teacher 60.69  student 52.45 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# GA_l1
# before: 32.39   after: 93.52  (forget_acc)
# before: 53.04   after: teacher 85.59  student 80.49 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
# UNSC
# before:   after: teacher   student
# python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 32
