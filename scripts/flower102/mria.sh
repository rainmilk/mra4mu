# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn

# FT
# before: 34.18  after: teacher 32.06  student 31.33
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name FT --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 32
# RL
# before: 34.6   after: teacher 90.76  student 93.78
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 24
# GA
# before: 20.46  after: teacher 91.23  student 90.7
 python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 12 --batch_size 32

# IU
# before: 25.74  after: teacher 84.54  student 84.4
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 32
# BU
# before: 24.26  after: teacher 8.3    student 0.27
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 32
# FT_l1
# before: 21.31  after: teacher 8.43   student 13.92
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name FT_l1 --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 32
# SalUn
# before: 35.44  after: teacher 91.7   student 87.08
python nets/mria.py --dataset flower-102 --model swin_t --st_model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 32