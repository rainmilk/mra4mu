# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"
# RL, GA, IU, BU, SalUn,fisher, GA_l1, UNSC
# RL
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name RL --model_suffix student_only --batch_size 64
# GA
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --model_suffix student_only --batch_size 64
# IU
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name IU --model_suffix student_only --batch_size 64
# BU
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name BU --model_suffix student_only --batch_size 64
# SalUn
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name SalUn --model_suffix student_only --batch_size 64
# fisher
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name fisher --model_suffix restore --batch_size 64 --fig_title Fisher
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name fisher --model_suffix student_only --batch_size 64 --fig_title Fisher
# GA_l1
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --model_suffix restore --batch_size 64 --fig_title L1-SP
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA_l1 --model_suffix student_only --batch_size 64 --fig_title L1-SP
# UNSC
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name UNSC --model_suffix restore --batch_size 64
python result_analysis/visual_results.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name UNSC --model_suffix student_only --batch_size 64