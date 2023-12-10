#!/bin/bash

#SBATCH --job-name=gcn_distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

#mannully activate virtual env before submitting job script

export RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS

srun python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node_rank=$SLURM_NODEID \
        --master_addr=10.141.0.1 \
        --master_port=12345 \
        --use_env \
	GCN_Cora_Distributed_MulNode.py
