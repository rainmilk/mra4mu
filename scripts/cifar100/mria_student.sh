# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, L1, SalUn, GA_l1, UNSC

# RL
# before: 8.56   after: 53.68 (forget_acc)
# before: 59.3   after: teacher 59.3    student 60.71 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# GA
# before: 20.08  after: 27.04 (forget_acc)
# before: 53.12  after: teacher 53.12   student 55.1  (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# IU
# before: 20.08  after: 25.2 (forget_acc)
# before: 14.01  after: teacher 14.01   student 29.7  (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name IU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# BU
# before: 25.44  after: 42.32 (forget_acc)
# before: 52.88  after: teacher 52.88   student 56.56 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name BU --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# SalUn
# before: 22.72  after: 35.76 (forget_acc)
# before: 40.95  after: teacher 40.95   student 52.17 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name SalUn --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# fisher
# before: 9.44   after: 21.36 (forget_acc)
# before: 15.85  after: teacher 15.85   student 29.52 (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name fisher --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# GA_l1
# before: 19.92  after: 25.76  (forget_acc)
# before: 53.73  after: teacher 53.73   student 54.98  (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA_l1 --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6
# UNSC
# before: 20.88  after: 37.2 (forget_acc)
# before: 27.46  after: teacher 27.46   student 48.01  (test_acc)
python nets/mria.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name UNSC --num_epochs 3 --align_epochs 5 --distill_epochs 5 --learning_rate 5e-5  --lr_student 2e-4 --no_t_update --batch_size 256 --top_conf 0.6