# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python data_process/gen_cifar100_exp_data.py --forget_ratio 0.5 --forget_classes 10 30 50 70 90