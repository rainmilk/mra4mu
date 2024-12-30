# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python main_mu.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --alpha 1 --unlearn_lr 4e-5 --uni_name GA --num_epochs 10  --batch_size 256