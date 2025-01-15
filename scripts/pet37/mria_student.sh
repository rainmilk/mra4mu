# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 33.2    after: 88.4  (forget_acc)
# before: 75.39   after: teacher 75.39  student 81.6 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# GA
# before: 22.4    after: 52  (forget_acc)
# before: 73.97   after: teacher 73.97  student 75.74 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# IU
# before: 20.0    after: 44.8  (forget_acc)
# before: 62.2    after: teacher 62.2  student 68.19 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# BU
# before: 8.4     after: 85.6  (forget_acc)
# before: 64.57   after: teacher 64.57  student 78.52 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# SalUn
# before: 12.8    after: 71.2  (forget_acc)
# before: 70.59   after: teacher 70.59  student 77.84 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# fisher
# before: 22.4    after: 38.8  (forget_acc)
# before: 25.7    after: teacher 25.7  student 48.57 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# GA_l1
# before: 19.6    after: 44  (forget_acc)
# before: 73.97   after: teacher 73.97  student 75.8 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
# UNSC
# before: 30.4    after: 66  (forget_acc)
# before: 76.89   after: teacher 76.89  student 79.86 (test_acc)
python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 2e-4  --lr_student 2e-4 --no_t_update --batch_size 64
