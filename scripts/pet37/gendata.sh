# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python data_process/gen_pet37_exp_data.py --forget_ratio 0.5 --forget_classes 1 8 15 21 29