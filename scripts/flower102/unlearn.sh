# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python main_mu.py --dataset flower-102 --model resnet18 --forget_ratio 0.5 --alpha 1 --unlearn_lr 8e-5 --uni_name GA --num_epochs 10  --batch_size 32