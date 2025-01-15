# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 14.98   after: 56.68  (forget_acc)
# before: 75.59   after: teacher 75.59  student 83.14 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# GA
# before: 23.89   after: 35.22  (forget_acc)
# before: 31.37   after: teacher 31.37  student 50.98 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# IU
# before: 29.15   after: 36.44  (forget_acc)
# before: 51.18   after: teacher 51.18  student 58.92 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# BU
# before: 34.01   after: 47.77  (forget_acc)
# before: 70.59   after: teacher 70.59  student 75.78 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# SalUn
# before: 26.72   after: 42.91  (forget_acc)
# before: 49.61   after: teacher 49.61  student 59.9 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# fisher
# before: 23.48   after: 31.17  (forget_acc)
# before: 29.41   after: teacher 29.41  student 37.65 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# GA_l1
# before: 32.39   after: 53.85  (forget_acc)
# before: 53.04   after: teacher 53.04  student 65.49 (test_acc)
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
# UNSC
# before:   after: teacher   student
# python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4  --lr_student 2e-4 --no_t_update --batch_size 32
