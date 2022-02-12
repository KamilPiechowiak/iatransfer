#!/bin/bash
#SBATCH --job-name=iatransfer
#SBATCH --output=iatransfer.out
#SBATCH --error=iatransfer.err
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH -p lab-ci-student

echo $1