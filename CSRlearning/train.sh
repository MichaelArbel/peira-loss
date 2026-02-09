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


parent_work_dir="../data/.workdir"
parent_log_dir="../data/outputs/deterministic_9"


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python main.py \
#                 learner.name="learners.CSELearningModel"\
#                 learner.kwargs.algorithm_name="algorithm_steps.CanonicalSubspaceEmbeddingExactAlgorithm","algorithm_steps.SemiGradientSSLAlgorithm"\
#                 task.init_seed=0,1,2,3,4\
#                 task.init_scale=1.,0.1,0.01,0.001,0.0001\
#                 learner.kwargs.reg_c=0.01\
#                 learner.kwargs.lr_u=10.\
#                 task.steps_per_epoch=20000\
#                 +mlxp.use_version_manager=True\
#                 +mlxp.interactive_mode=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir


parent_work_dir="../data/.workdir"
parent_log_dir="../data/outputs/debug"


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 learner.name="learners.CSELearningModel"\
#                 learner.kwargs.algorithm_name="algorithm_steps.SemiGradientSSLAlgorithm"\
#                 task.init_scale=1.\
#                 data.kwargs.deterministic=True\
#                 learner.kwargs.reg_c=0.01\
#                 learner.kwargs.lr_u=1.\
#                 learner.kwargs.params_q.update_rule="exact_inverse"\
#                 learner.kwargs.params_q.momentum=True\
#                 learner.kwargs.params_q.cold_start=False\
#                 learner.kwargs.eta=.01\
#                 learner.kwargs.n_opt_pred=10\
#                 learner.kwargs.full_solve_v=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 learner.name="learners.CSELearningModel"\
#                 learner.kwargs.algorithm_name="algorithm_steps.CanonicalSubspaceEmbeddingExactAlgorithm"\
#                 task.init_scale=0.0001\
#                 data.kwargs.deterministic=True\
#                 learner.kwargs.reg_c=0.01\
#                 learner.kwargs.lr_u=1.\
#                 learner.kwargs.lr_v=1.\
#                 task.steps_per_epoch=20000\
#                 learner.kwargs.n_opt_pred=1\
#                 learner.kwargs.n_opt_v=1\
#                 learner.kwargs.full_solve_v=True\
#                 learner.kwargs.params_q.update_rule="exact_inverse"\
#                 learner.kwargs.params_q.momentum=True\
#                 learner.kwargs.params_q.cold_start=False\
#                 learner.kwargs.eta=1.\
#                 ++learner.kwargs.params_q.kwargs.eta=0.25\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\



TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
                learner.name="learners.CSELearningModel"\
                learner.kwargs.algorithm_name="algorithm_steps.CanonicalSubspaceEmbeddingBilevelAlgorithm"\
                task.init_scale=0.0001\
                data.kwargs.deterministic=True\
                learner.kwargs.reg_c=0.1\
                learner.kwargs.lr_u=.1\
                learner.kwargs.lr_v=.1\
                learner.kwargs.eta=1.\
                learner.kwargs.n_opt_pred=1\
                data.kwargs.sign_flip=False\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir







# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
#                 learner.name="learners.CSELearningModel"\
#                 learner.kwargs.algorithm_name="algorithm_steps.UnconstrainedCCAAlgorithm"\
#                 task.init_scale=0.0001\
#                 data.kwargs.deterministic=True\
#                 learner.kwargs.reg_c=0.1\
#                 learner.kwargs.lr_u=.1\
#                 learner.kwargs.lr_v=.1\
#                 learner.kwargs.eta=1.\
#                 learner.kwargs.n_opt_pred=1\
#                 data.kwargs.sign_flip=False\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir















# if [[ $? == 124 ]]; then
#   scontrol requeue $SLURM_JOB_ID
# fi
