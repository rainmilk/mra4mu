# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 15.34  after: 32.81  (forget_acc)
# before: 38.8   after: teacher 38.8   student 47.88 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6
# GA
# before: 26.72  after: 35.82 (forget_acc)
# before: 50.45  after: teacher 50.45  student 52.76 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6
# IU
# before: 9.05   after: 9.55  (forget_acc)
# before: 10.5   after: teacher 10.5   student 11.46 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 8e-5 --no_t_update --batch_size 256 --top_conf 0.6 --ls_gamma 1e-3
# BU
# before: 8.23   after: 50.67  (forget_acc)
# before: 46.24  after: teacher 46.24  student 65.08 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6
# SalUn
# before: 11.69  after: 61.48  (forget_acc)
# before: 48.6   after: teacher 48.6   student 69.69 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6
# fisher
# before: 14.21   after: 19.52  (forget_acc)
# before: 16.41   after: teacher 16.41  student 25.23 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6
# GA_l1
# before: 15.41   after: 23.14  (forget_acc)
# before: 45.14   after: teacher 45.14  student 45.62 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6
# UNSC
# before: 3.08    after: 51.33  (forget_acc)
# before: 19.3    after: teacher 19.3   student 56.82 (test_acc)
python nets/mria.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5 --lr_student 5e-5 --no_t_update --batch_size 256 --top_conf 0.6