#!/bin/bash

#SBATCH -o test_moco.log-%j
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2       # number of gpus per node
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node (must match #gpus-per-node)
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks) - num_workers as well


## Conda module
source /etc/profile
module load anaconda/2021a

echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date


srun ./scripts/moco_v1.sh

echo "Run completed at:- "
date
