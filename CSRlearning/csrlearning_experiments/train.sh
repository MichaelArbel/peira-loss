#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --account=jsc@a100
#SBATCH --constraint=a100
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append

# module purge
# module load arch/a100
# module load pytorch-gpu/py3/2.5.0


parent_work_dir="../../data/.workdir"
parent_log_dir="../../data/outputs/debug"


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="unconstrained_cca"\
#                 task.init_scale=0.0001\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_bilevel"\
#                 task.init_scale=0.0001\
#                 optimizer.kwargs.lr=0.0001\
#                 optimizer.name="torch.optim.AdamW"\
#                 algorithm.kwargs.lr=1.\
#                 algorithm.kwargs.n_iter=1\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir



TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
                algorithm="cse_bilevel_joint"\
                task.init_scale=0.0001\
                optimizer.kwargs.lr=0.0001\
                optimizer.name="torch.optim.AdamW"\
                algorithm.kwargs.lr=1.\
                algorithm.kwargs.ema=.9\
                data.kwargs.deterministic=False\
                data.kwargs.sign_flip=False\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir




# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_exact"\
#                 task.init_scale=0.0001\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir



# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_sto"\
#                 task.init_scale=0.0001\
#                 algorithm.kwargs.lr=1.\
#                 optimizer.kwargs.lr=0.001\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir






# if [[ $? == 124 ]]; then
#   scontrol requeue $SLURM_JOB_ID
# fi
