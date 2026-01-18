#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=19:10:00
#SBATCH --account=jsc@a100
#SBATCH --constraint=a100
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append

# module purge
# module load arch/a100
# module load pytorch-gpu/py3/2.5.0


parent_work_dir="../data/.workdir"
parent_log_dir="../data/outputs/debug"


TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
                dataloader.num_steps=384\
                training.aggregate_k_gradients=3\
                dataloader.batch_size=24\
                data.path="$WORK/documents/code/KernelTabPFN/data"\
                model=biattention_ktabpfn\
                model/decoder=mlp_decoder_non_equi\
                model.bkbn.kwargs.nlayers=6\
                model.bkbn.kwargs.equivariant_encoder=False\
                model.bkbn.kwargs.compile_model=True\
                model.bkbn.kwargs.feature_mask_mode='Bq2Bk'\
                training.compile=False\
                +mlxp.use_version_manager=True\
                +mlxp.interactive_mode=False\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir



# if [[ $? == 124 ]]; then
#   scontrol requeue $SLURM_JOB_ID
# fi