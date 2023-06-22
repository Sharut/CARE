# Structuring Representation Geometry with Rotationally Equivariant Contrastive Learning
By Sharut Gupta*, Joshua Robinson*, Derek Lim, Soledad Villar, Stefanie Jegelka.  

Self-supervised learning converts raw perceptual data such as images to a compact space where simple Euclidean distances measure meaningful variations in data. In this paper, we extend this formulation by adding additional geometric structure to the embedding space by enforcing transformations of input space to correspond to simple (i.e., linear) transformations of embedding space. Specifically, in the contrastive learning setting, we introduce an *equivariance* objective and theoretically prove that its minima forces augmentations on input space to correspond to *rotations* on the spherical embedding space. We show that merely combining our equivariant loss with a non-collapse term results in non-trivial  representations, without requiring invariance to data augmentations. Optimal performance is achieved by also encouraging approximate invariance, where input augmentations correspond to small rotations. Our method, **CARE**: **C**ontrastive **A**ugmentation-induced **R**otational **E**quivariance, leads to improved performance on downstream tasks, and ensures sensitivity in embedding space to important variations in data (e.g., color) that standard contrastive methods do not achieve. 

<p align='center'>
<img src='./figures/main-fig.png' width='1000'/>
</p>


The key contributions of this work include: 
- Introducing CARE, a novel equivariant contrastive learning framework that trains transformations (cropping, jittering, blurring, etc.) in input space to approximately correspond to local orthogonal transformations in representation space. 
- Theoretically proving and empirically demonstrating that CARE places an orthogonally equivariant structure on the embedding space.
- Showing that CARE increases sensitivity to features (e.g., color) compared to invariance-based contrastive methods, and  also improves performance on image recognition tasks.


## CARE: Contrastive Augmentation-induced Rotational Equivariance

### Prerequisites
The code has the following package dependencies:
- Pytorch >= 0.13.0 (preferably 2.0.0)
- Torchvision >=  0.12.0 

After installing these dependencies, install ImageNet100 dataset using a subset of the official ImageNet dataset, as given at https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt 

### Training and Evaluation

#### CIFAR10, CIFAR100, STL10

To train CARE with SimCLR bacbone using ResNet50 model on STL10 dataset, run the following command: 
```
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
  --save-root PATH/TO/LOG \
  --data-root PATH/TO/DATA \
  --project PROJECT_NAME \
  --user SAMPLE_USER \
  --run-name RUN_NAME \
```
To run standard SimCLR model, simply add another argument `--train-simclr`.
This script uses all the default hyper-parameters as described in the CARE paper. Training should fit on 2 NVIDIA Tesla V100 GPUs with 32GB accelerator RAM. 

For linear evaluation, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1 python linear.py 
  --model-path PATH/TO/MODEL \
  --model resnet50 \
  --lin-eval \
  --load-epoch 400 \
  --dataset-name STL10 \ 
  --data-root PATH/TO/DATA \ 
  --save-root PATH/TO/LOG \
  --project-name PROJECT_NAME \
  --user SAMPLE_USER \
  --run-name RUN_NAME

```

To reproduce our results on CIFAR10, CIFAR100, STL10, run the scripts in ./cifar10_cifar100_stl10/scripts/{dataset_name}.sh

#### ImageNet100
For ImageNet100, we use distributed training. To train CARE with SimCLR bacbone using ResNet50 model, run the following command: 
```
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} --master_addr=${MASTER} --master_port=1625 main.py \
  -a resnet50 \
  --lr 0.4 \
  --batch-size 256 \
  --dist-url env:// \
  --moco-t 0.2 \
  --moco-m 0.99  \
  --epochs 200 \
  --cos \
  --log-root PATH/TO/LOG \
  --data PATH/TO/DATA \
  --run-name RUN_NAME \
  --method simclr  \
  --split-batch \
  --weight 0.005 \
  --equiv-splits-per-gpu 4 \
  --save-only-last-checkpoint \
  --project PROJECT_NAME \
  --user SAMPLE_USER \
```
To run CARE with MoCo-v2 backbone, simply replace `--method simclr` with `--method moco`
To run our ImageNet100 experiments, we use the SLURM scheduler. All experiments are ran on 2 Nodes, each with 2 NVIDIA Tesla V100 GPUs with 32GB accelerator RAM.
Run the scripts ./imagenet100/scripts/simclr.sh and ./imagenet100/scripts/moco.sh to reproduce our results of using CARE with SimCLR and MoCo-v2 backbone respectively. 

For linear evaluation, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} --master_addr=${MASTER} --master_port=1626 main_lincls.py \
  --pretrained PATH/TO/MODEL \
  --dist-url env:// \
  --arch resnet50 \
  --lr 10.0 \
  --batch-size 128  \
  --schedule 30 40 50 \
  --epochs 60 \
  --dist-url tcp://localhost:10001 \
  --log-root PATH/TO/LOG \
  --data PATH/TO/DATA \
  --run-name RUN_NAME \
```

### Results

<p align='center'>
<img src='./figures/accuracy_bars.png' width='1000'/>
</p>
