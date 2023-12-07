#!/bin/bash

#SBATCH --job-name=gcn_distributed
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

source /home/dsys2352/miniconda3/bin/activate /home/dsys2352/miniconda3/envs/dist_GNN
export RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS

srun python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=3 \
        --node_rank=$SLURM_NODEID \
        --master_addr=10.141.0.1 \
        --master_port=12345 \
        --use_env \
        GAT_Cora_Distributed.py

