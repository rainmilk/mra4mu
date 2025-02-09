# Review for "Recalling The Forgotten Class Memberships: Unlearned Models Can Be Noisy Labelers to Leak Privacy"

## 1. COLUR Framework implementation
/nets/mria.py: The code for building the MRA framework

## 2. Implementation of Machine Unlearning Methods
/unlearn: The directory accommodates all implementations of MU methods used in the paper.

## 3. Scripts for Experiments
### 3.1 Script Directory Structure
/scripts: The directory accommodates all scripts for the experiments.
- /scripts/cifar10: scripts for CIFAR-10
- /scripts/cifar100: scripts for CIFAR-100
- /scripts/flower102: scripts for Flower-102
- /scripts/pet37: scripts for Pet-37

### 3.2 Script Directory Structure
Under each script directory for a dataset, e.g. /scripts/cifar10 for CIFAR-10, it contains
1. gendata.sh: generate train and test data for experiments
2. train.sh: training model with $D_{tr}$ for TRM 
3. unlearn.sh: running all the MU methods with $D_{f}$  for ULM
4. mria_student.sh: running the MRA in the closed-source case for RCM
5. mria_student.sh: running the MRA in the open-source case for RCM
6. mria_distill: running the MRA with distillation only
7. evals.sh: evaluating the experimental results
8. visuals.sh: draw figures for the experiments





