# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# RL, GA, IU, BU, SalUn, fisher, GA_l1
# RL
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name RL --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name RL --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name RL --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name RL --model_suffix student_only
# GA
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA --model_suffix student_only
# IU
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name IU --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name IU --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name IU --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name IU --model_suffix student_only
# BU
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name BU --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name BU --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name BU --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name BU --model_suffix student_only
# SalUn
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name SalUn --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name SalUn --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name SalUn --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name SalUn --model_suffix student_only
# fisher
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name fisher --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name fisher --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name fisher --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name fisher --model_suffix student_only
# GA_l1
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA_l1 --model_suffix ul
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA_l1 --model_suffix restore
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA_l1 --model_suffix distill
python result_analysis/evaluate_results.py --dataset flower-102 --model swin_t --forget_ratio 0.5 --uni_name GA_l1 --model_suffix student_only
