# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python data_process/gen_flower102_exp_data.py --forget_ratio 0.5 --forget_classes 50 72 76 88 93