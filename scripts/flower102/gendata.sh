# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python data_process/gen_flower102_exp_data.py --forget_ratio 0.5 --forget_classes 1 3 5 7 9