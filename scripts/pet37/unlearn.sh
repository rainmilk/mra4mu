# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# Fisher, RL, GA, IU, BU, L1, SalUn
python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 3 --unlearn_lr 1e-3 --uni_name fisher --num_epochs 10  --batch_size 32

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-3 --uni_name RL --num_epochs 10  --batch_size 32

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 8e-5 --uni_name GA --num_epochs 10  --batch_size 32

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 3 --unlearn_lr 1e-2 --uni_name IU --num_epochs 10  --batch_size 32

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 5e-3 --uni_name BU --num_epochs 20  --batch_size 32

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 0.02 --unlearn_lr 1e-3 --uni_name FT_l1 --num_epochs 10  --batch_size 32

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-3 --uni_name SalUn --num_epochs 10  --batch_size 32 --class_to_replace 1 8 15 21 29 --mask_thresh 0.8