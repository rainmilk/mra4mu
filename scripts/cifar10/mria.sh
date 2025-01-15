# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 15.34   after: 35.06  (forget_acc)
# before: 38.8   after: teacher 47.13  student 50.08  (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1
# GA
# before: 26.72   after: 50.92  (forget_acc)
# before: 50.45   after: teacher 58.18  student 53.18 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1 --ls_gamma 0.01
# IU
# before: 9.05   after: 75.87  (forget_acc)
# before: 10.5   after: teacher 68.86  student 59.94 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1
# BU
# before: 8.23   after: 87.74  (forget_acc)
# before: 46.24  after: teacher 78.3  student 72.29 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1
# SalUn
# before: 11.69  after: 99.68  (forget_acc)
# before: 48.6   after: teacher 82.87  student 78.75 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1
# fisher
# before: 14.21   after: 72.58 (forget_acc)
# before: 16.41   after: teacher 70.79  student 65.53 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1
# GA_l1
# before: 15.41   after: 33.8  (forget_acc)
# before: 45.14   after: teacher 47.85  student 48.8  (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 4e-4 --lr_student 4e-4 --batch_size 256 --top_conf 0.1 --ls_gamma 0.01
# UNSC
# before: 3.08   after: 92.13  (forget_acc)
# before: 19.3   after: teacher 77.37  student 69.95 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 1e-4 --batch_size 256 --top_conf 0.1