# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL
python result_analysis/visual_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name RL --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset flower-102 --model resnet18 --forget_ratio 0.5 --uni_name RL --model_suffix student_only --batch_size 64
# GA
python result_analysis/visual_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset flower-102 --model resnet18 --forget_ratio 0.5 --uni_name GA --model_suffix student_only --batch_size 64