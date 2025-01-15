# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 8.56   after: 89.84 (forget_acc)
# before: 59.3   after: teacher 59.94    student 53.95 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256 --ls_gamma 0.01
# GA
# before: 20.08  after: 68.16 (forget_acc)
# before: 53.12  after: teacher 58.9    student 47.45 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# IU
# before: 20.08  after: 96.72 (forget_acc)
# before: 14.01  after: teacher  59.59  student 38.84 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# BU
# before: 25.44  after: 96.96 (forget_acc)
# before: 52.88  after: teacher 58.67   student 50.26 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# SalUn
# before: 22.72  after: 94.48 (forget_acc)
# before: 40.95  after: teacher 56.39   student 50.02 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# fisher
# before: 9.44   after: 68.64 (forget_acc)
# before: 15.85  after: teacher 52.12   student 43.71 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# GA_l1
# before: 19.92  after: 76.4  (forget_acc)
# before: 53.73  after: teacher 59.79   student 50.63  (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256
# UNSC
# before: 20.88  after: 98.48 (forget_acc)
# before: 27.46  after: teacher 59.82   student 39.01  (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 1e-4 --lr_student 2e-4 --batch_size 256