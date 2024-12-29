# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 1e-4 --uni_name GA --num_epochs 10  --batch_size 64

python main_mu.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --alpha 0.01 --unlearn_lr 1e-4 --uni_name GA_l1 --num_epochs 10  --batch_size 64