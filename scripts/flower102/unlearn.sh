# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn
# 29.55
# python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 2.4e-2 --uni_name FT --num_epochs 6 --batch_size 32 --print_freq 20
# 31.17 无法复现
# python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 0.25 --unlearn_lr 1e-5 --uni_name FT_l1 --num_epochs 10 --batch_size 32  --print_freq 20

# 14.98
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-4 --uni_name RL --num_epochs 10 --batch_size 32 --print_freq 20
# 23.89
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 8e-5 --uni_name GA --num_epochs 8 --batch_size 32 --print_freq 20
# 29.15
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 200 --uni_name IU --num_epochs 10  --batch_size 32 --print_freq 20 --WF_N 1000
# 34.01
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 0.2 --unlearn_lr 1e-5 --uni_name BU --num_epochs 20 --batch_size 32 --print_freq 20
# 26.72
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-6 --uni_name SalUn --num_epochs 10 --batch_size 32 --class_to_replace 50 72 76 88 93 --mask_thresh 0.8  --print_freq 20
# 23.48
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 16 --unlearn_lr 5e-4 --uni_name fisher --num_epochs 10 --batch_size 32  --print_freq 20 --WF_N 50
# 32.39
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 0.4 --unlearn_lr 2e-5 --uni_name GA_l1 --num_epochs 10 --batch_size 32  --print_freq 20
#
# python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-5 --uni_name UNSC --num_epochs 10 --batch_size 32
