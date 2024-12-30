# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python result_analysis/evaluate_results.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix restore

python result_analysis/visual_results.py --dataset cifar-10 --model efficientnet_s --forget_ratio 0.5 --uni_name GA