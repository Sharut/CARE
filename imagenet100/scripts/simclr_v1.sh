#!/bin/bash

source /etc/profile
module load anaconda/2021a

# We extract the master node address (the one that every node must connects to)
LISTNODES=`scontrol show hostname $SLURM_JOB_NODELIST`
MASTER=`echo $LISTNODES | cut -d" " -f1`


SAVE_DIR=./results
DATA_DIR=./data/imagenet100
PROJECT_NAME='project_name'
USER='sample_user'
RUN_NAME='test'

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} --master_addr=${MASTER} --master_port=1625 main.py \
  -a resnet50 \
  --lr 0.4 \
  --batch-size 256 \
  --dist-url env:// \
  --moco-t 0.2 \
  --moco-m 0.99  \
  --epochs 200 \
  --cos \
  --log-root ${SAVE_DIR}/${RUN_NAME} \
  --data $DATA_DIR \
  --method simclr  \
  --run-name $RUN_NAME \
  --split-batch \
  --weight 0.005 \
  --equiv-splits-per-gpu 4 \
  --save-only-last-checkpoint \
  --project $PROJECT_NAME \
  --user $USER \






