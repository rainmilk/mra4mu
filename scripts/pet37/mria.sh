# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python nets/mria.py --dataset pet-37 --model resnet18 --forget_ratio 0.5 --uni_name GA --num_epochs 2 --align_epochs 3 --distill_epochs 5 --learning_rate 2e-4 --lr_student 2e-4 --batch_size 64