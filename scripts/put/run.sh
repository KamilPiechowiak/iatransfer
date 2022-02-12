#!/bin/bash
#SBATCH --job-name=iatransfer
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH -p lab-ci-student

srun docker build -t iatransfer-pytorch put
srun nvidia-docker run --rm --name iatransfer --ipc=host --user 16023 -v /home/inf136780/iatransfer:/workspace/iatransfer iatransfer-pytorch bash -c "cd iatransfer && python3 -m iatransfer.researchput.runner $1 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT"
