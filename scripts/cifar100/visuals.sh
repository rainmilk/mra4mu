# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name RL --model_suffix student_only --batch_size 256
# GA
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix restore --batch_size 256
python result_analysis/visual_results.py --dataset cifar-100 --model efficientnet_s --forget_ratio 0.5 --uni_name GA --model_suffix student_only --batch_size 256