# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn
# 29.72
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-1 --uni_name FT --num_epochs 5  --batch_size 256
# 22.26  test_acc 反复
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 0.18 --uni_name RL --num_epochs 5  --batch_size 256
# 18.07
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 10  --batch_size 256
# 9.13
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 0.3 --unlearn_lr 1e-3 --uni_name IU --num_epochs 10  --batch_size 256
# 11.22
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-4 --uni_name BU --num_epochs 10  --batch_size 256
# 8.98  LOSS 118
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 0.02 --unlearn_lr 5e-3 --uni_name FT_l1 --num_epochs 10  --batch_size 256
# 13.02
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-4 --uni_name SalUn --num_epochs 10  --batch_size 256 --class_to_replace 1 3 5 7 9 --mask_thresh 0.8




