# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn
# 34.97
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 116e-3 --uni_name FT --num_epochs 10  --batch_size 256
# 20.49  test_acc 反复
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 0.18 --uni_name RL --num_epochs 5  --batch_size 256
# 26.72
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 10  --batch_size 256
# 9.05
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 0.3 --unlearn_lr 1e-3 --uni_name IU --num_epochs 10  --batch_size 256
# 8.23
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 0.3 --unlearn_lr 5e-4 --uni_name BU --num_epochs 10  --batch_size 256
# 8.76  LOSS 118
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 0.02 --unlearn_lr 5e-3 --uni_name FT_l1 --num_epochs 10  --batch_size 256
# 11.69
python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 8e-5 --uni_name SalUn --num_epochs 10  --batch_size 256 --class_to_replace 1 3 5 7 9 --mask_thresh 0.8

python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 18 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 256

python main_mu.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-4 --uni_name UNSC --num_epochs 10 --batch_size 256


