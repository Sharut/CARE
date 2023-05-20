#!/bin/bash

#SBATCH -o test.log-%j
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2

CUDA_VISIBLE_DEVICES=0,1 python main.py \
						 --model resnet50 \
						 --optimizer SGD \
						 --lr 0.06 \
						 --weight-decay 5e-4 \
						 --lr-schedule-type cosine \
						 --warmup-epochs 10 \
						 --temperature 0.5 \
						 --epochs 400 \
						 --batch-size 256 \
						 --weight 0.01 \
						 --equiv-splits 8 \
						 --dataset-name stl10 \
						 --log-freq 1 \
						 --save-root ./results \
						 --data-root ./data \
						 --project project_name \
						 --user sample_user \
						 --run-name test \
wait
echo "Run completed at:- "
date

