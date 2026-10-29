#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --account=jsc@cpu
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append

set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi

parent_work_dir="../../data/.workdir"
parent_log_dir="../../data/outputs/deterministic_2"

for algorithm in self_distillation_exact peira_exact; do
    if [[ "$algorithm" == self_distillation_exact ]]; then
        learning_rate=10.
        regularization=0.00001
    else
        learning_rate=0.1
        regularization=0.4
    fi
    python main.py \
        algorithm="$algorithm" \
        seed=0 \
        task.init_scale=0.000001,0.00001,0.0001,0.001,0.01 \
        algorithm.kwargs.reg="$regularization" \
        optimizer.kwargs.lr="$learning_rate" \
        task.total_steps=400000 \
        +mlxp.logger.parent_log_dir="$parent_log_dir" \
        +mlxp.version_manager.parent_work_dir="$parent_work_dir" \
        +mlxp.use_version_manager=True \
        +mlxp.interactive_mode=True
done
