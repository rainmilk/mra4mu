# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn
# 34.18
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 2e-2 --uni_name FT --num_epochs 10  --batch_size 32 --print_freq 20
# 34.6
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 12e-4 --uni_name RL --num_epochs 10  --batch_size 32 --print_freq 20
# 20.46
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 8e-5 --uni_name GA --num_epochs 8  --batch_size 32 --print_freq 20
# 25.74
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 80 --uni_name IU --num_epochs 10  --batch_size 32 --print_freq 20 --WF_N 200
# 24.26
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 0.15 --unlearn_lr 1e-3 --uni_name BU --num_epochs 10  --batch_size 32 --print_freq 20
# 21.31 test_acc 会先降低再升高, 中间反复
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 0.15 --unlearn_lr 1e-4 --uni_name FT_l1 --num_epochs 10 --batch_size 32  --print_freq 20
# 35.44
python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-4 --uni_name SalUn --num_epochs 10  --batch_size 32 --class_to_replace 50 72 76 88 93 --mask_thresh 0.8  --print_freq 20
# 11.81
# python main_mu.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --alpha 18 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 32  --print_freq 20