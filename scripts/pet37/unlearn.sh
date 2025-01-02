# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn
# 18.4
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 288e-4 --uni_name FT --num_epochs 4  --batch_size 32
# 10
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-2 --uni_name RL --num_epochs 10  --batch_size 32
# 22.4
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 6e-5 --uni_name GA --num_epochs 10  --batch_size 32
# 20
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 4.6 --unlearn_lr 1e-1 --uni_name IU --num_epochs 10  --batch_size 32
# 8.4
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-5 --uni_name BU --num_epochs 10  --batch_size 32
# 33.6
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 0.02 --unlearn_lr 4e-4 --uni_name FT_l1 --num_epochs 10  --batch_size 32
# 12.8
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 9e-5 --uni_name SalUn --num_epochs 10  --batch_size 32 --class_to_replace 1 8 15 21 29 --mask_thresh 0.8
#
# python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 30 --unlearn_lr 1e-1 --uni_name fisher --num_epochs 10  --batch_size 32