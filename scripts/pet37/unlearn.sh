# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# FT, RL, GA, IU, BU, L1, SalUn
# 9.6
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 12e-2 --uni_name FT --num_epochs 6  --batch_size 32
# 4.8
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-2 --uni_name RL --num_epochs 10  --batch_size 32
# 20.8
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 9e-5 --uni_name GA --num_epochs 6  --batch_size 32
# 15.6
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-3 --uni_name IU --num_epochs 10  --batch_size 32
# 4.4
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-3 --uni_name BU --num_epochs 20  --batch_size 32
# 23.6
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 0.02 --unlearn_lr 4e-4 --uni_name FT_l1 --num_epochs 10  --batch_size 32
# 9.2
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 4e-4 --uni_name SalUn --num_epochs 10  --batch_size 32 --class_to_replace 1 8 15 21 29 --mask_thresh 0.8
# 12.8
# python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 30 --unlearn_lr 1e-1 --uni_name fisher --num_epochs 10  --batch_size 32