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
#                 algorithm="ey_cca"\
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


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="ey_cca"\
#                 task.init_scale=0.0001\
#                 optimizer.kwargs.lr=.1\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 with_logging=False\
#                 task.total_steps=400000\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir




# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_bilevel"\
#                 task.init_scale=0.0001\
#                 optimizer.kwargs.lr=0.01\
#                 optimizer.name="torch.optim.SGD"\
#                 algorithm.kwargs.lr=1.\
#                 algorithm.kwargs.reg=0.1\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir







# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_bilevel_joint"\
#                 task.init_scale=0.0001\
#                 optimizer.kwargs.lr=0.01\
#                 optimizer.name="torch.optim.SGD"\
#                 algorithm.kwargs.lr=1.\
#                 algorithm.kwargs.ema=1.\
#                 algorithm.kwargs.reg=0.1\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir




# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="ssl_exact"\
#                 algorithm.kwargs.reg=0.00001\
#                 task.init_scale=10.\
#                 optimizer.kwargs.lr=200.\
#                 task.total_steps=400000\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 with_logging=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_exact"\
#                 task.init_scale=0.000001\
#                 algorithm.kwargs.reg=0.4\
#                 optimizer.kwargs.lr=.4\
#                 task.total_steps=400000\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 with_logging=False\
#                 stop_when_completed=True\
#                 latent_dim=4\
#                 with_scheduler=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir


####   100- 122000 | 4-148700 | 30-127800 | 4- 140000


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_sto"\
#                 task.init_scale=0.000001\
#                 algorithm.kwargs.reg=0.4\
#                 algorithm.kwargs.lr=0.1\
#                 optimizer.kwargs.lr=.5\
#                 task.total_steps=400000\
#                 data.kwargs.deterministic=False\
#                 data.kwargs.sign_flip=False\
#                 with_logging=False\
#                 stop_when_completed=True\
#                 latent_dim=30\
#                 with_scheduler=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir








# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="ey_cca_sto"\
#                 task.init_scale=0.000001\
#                 optimizer.kwargs.lr=.1\
#                 optimizer.name="torch.optim.SGD"\
#                 task.total_steps=400000\
#                 data.kwargs.deterministic=False\
#                 data.kwargs.sign_flip=False\
#                 data.kwargs.batch_size=50\
#                 with_logging=False\
#                 stop_when_completed=True\
#                 latent_dim=4\
#                 eval_on_determinisitc=True\
#                 with_scheduler=True\
#                 scheduler.name="csrlearning_experiments.utils.utils.InverseTimeLRScheduler"\
#                 +scheduler.kwargs.power=0.5\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir





# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="ey_cca_sto"\
#                 task.init_scale=0.000001\
#                 optimizer.kwargs.lr=.1\
#                 optimizer.name="torch.optim.SGD"\
#                 +scheduler.kwargs.T_max=400000\
#                 scheduler.name="torch.optim.lr_scheduler.CosineAnnealingLR"\
#                 task.total_steps=400000\
#                 data.kwargs.deterministic=False\
#                 data.kwargs.sign_flip=False\
#                 data.kwargs.batch_size=50\
#                 with_logging=False\
#                 stop_when_completed=True\
#                 with_scheduler=True\
#                 latent_dim=4\
#                 eval_on_determinisitc=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir






# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="ssl_exact"\
#                 task.init_scale=0.01\
#                 algorithm.kwargs.reg=0.00001\
#                 optimizer.kwargs.lr=.1\
#                 task.total_steps=400000\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 with_logging=False\
#                 stop_when_completed=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir













# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="ey_cca"\
#                 task.init_scale=0.01\
#                 optimizer.kwargs.lr=.1\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 task.total_steps=400000\
#                 with_logging=False\
#                 with_scheduler=False\
#                 stop_when_completed=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir




# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_sto"\
#                 task.init_scale=0.0001\
#                 algorithm.kwargs.lr=1.\
#                 optimizer.kwargs.lr=0.01\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir




















# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 algorithm="cse_bilevel_joint"\
#                 task.init_scale=0.0001\
#                 optimizer.kwargs.lr=0.01\
#                 optimizer.name="torch.optim.SGD"\
#                 algorithm.kwargs.lr=1.\
#                 algorithm.kwargs.ema=1.\
#                 algorithm.kwargs.reg=0.1\
#                 task.total_steps=40000\
#                 data.kwargs.deterministic=True\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir




TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python main.py \
                algorithm="ey_cca"\
                seed=0\
                task.init_scale=0.000001\
                optimizer.kwargs.lr=.1\
                task.total_steps=400000\
                data.kwargs.deterministic=True\
                data.kwargs.sign_flip=False\
                with_scheduler=False\
                latent_dim=4\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\








# if [[ $? == 124 ]]; then
#   scontrol requeue $SLURM_JOB_ID
# fi
