#!/bin/bash
#SBATCH --job-name=dsgnnperf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

module load python/3.8
module load pytorch/1.9.1

srun --ntasks=2 --export=ALL python GCN_Cora_Distributed_MulNode.py
