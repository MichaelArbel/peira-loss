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
parent_log_dir="../../data/outputs/deterministic"




TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
                algorithm="ssl_exact"\
                task.init_scale=0.000001,0.00001,0.0001,0.001,0.01\
                algorithm.kwargs.reg=0.00001\
                optimizer.kwargs.lr=10.,100.,1000.\
                task.total_steps=400000\
                data.kwargs.deterministic=True\
                data.kwargs.sign_flip=False\
                with_logging=False\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir



TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
                algorithm="cse_exact","unconstrained_cca"\
                algorithm.kwargs.reg=0.1\
                task.init_scale=0.000001,0.00001,0.0001,0.001,0.01\
                optimizer.kwargs.lr=.1\
                task.total_steps=400000\
                data.kwargs.deterministic=True\
                data.kwargs.sign_flip=False\
                with_logging=False\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir






# if [[ $? == 124 ]]; then
#   scontrol requeue $SLURM_JOB_ID
# fi
