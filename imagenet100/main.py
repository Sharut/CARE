#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from functorch import vmap

from model import loader
from model import builder
from model.loss import loss, equivariance_loss
import log
import warnings
warnings.filterwarnings('ignore')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='./data/imagenet100', type=str, 
                    help='path to data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--split-batch', action='store_true',
                    help='Split the batch size across GPUs or use given batch-size for each GPU')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=3, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--sync-batchnorm', default = None, action='store_true', help='Sync batch norm layers across multiple GPU\'s.')

# equivariant model configs:
parser.add_argument('--train-simclr', action='store_true', help='Train RSSL or SimCLR')
parser.add_argument('--weight', default=0.01, type=float, help='weight to regularization')
parser.add_argument('--equiv-splits-per-gpu', default=8, type=int, help='Number of batch subdivisions for equivariance')
parser.add_argument('--kernel-size', default=5, type=int, help='kernel size for gaussian blur')
parser.add_argument('--equivariant-batch-size', default=128, type=int, help='batch size for equivariant loss.')
parser.add_argument('--test-freq', default=1, type=int, help='Frequency of testing in epochs')


# options for moco v2
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--weight_scale', default=0.0, type=float,
                    help='softmax temperature (default: 0.01)')
parser.add_argument('--log-root', default='', type=str, metavar='PATH',
                    help='path to logger directory')
parser.add_argument('--method', default='', type=str,
                    help='training method to use')
parser.add_argument('--knn-k', default=20, type=int,
                    help='the k in kNN')
parser.add_argument('--no_knn', action='store_true',
                    help='turn off kNN eval.')
parser.add_argument("--local-rank", type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')


## Wandb arguments
parser.add_argument('--debug', action='store_true', help='do not run wandb logging')
parser.add_argument('--run-name', default='', type=str, help='Choose name of wandb run')
parser.add_argument('--project', default='project_name', type=str, help='wandb project dataset_name')
parser.add_argument('--user', default='sample_user', type=str, help='wandb username')
parser.add_argument('--save-only-last-checkpoint', action='store_true', help='Delete all checkpoints besides last checkpoint.')


def main():
    args = parser.parse_args()
    ## Seed and CUDA benchmarking
    if args.seed is not None:
        set_seed(args.seed, torch.cuda.is_available())
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    ## For torch.distributed.launch
    if args.dist_url == "env://" and args.world_size == -1: 
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank=os.environ['LOCAL_RANK']

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count()
    
    ## Single-Node Multi-GPU training (Rank = 0; world-size = 1)
    if args.multiprocessing_distributed:
        # To account for used sockets
        args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
        args.world_size = ngpus_per_node * args.world_size # world_size initially is #nodes, so total GPU's = #nodes X #gpus-per-node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.local_rank, ngpus_per_node, args) # args.local_rank gives local rank of the process, automatically calculated by torch.distributed.launch


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = int(gpu) # local rank

    if args.gpu is not None: # Set GPU device for each process
        torch.cuda.set_device(args.gpu)
    
    ## CUDA benchmarking
    cudnn.benchmark = True
    
    # Initialize Distributed process
    if args.distributed:
        ## For torch.distributed.launch
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"]) # Global Rank
            print("=> Torch.distributed.launch: dist-url:{} at PROCID {} / {} on GPU {} / {}".format(args.dist_url, args.rank, args.world_size, gpu, ngpus_per_node)) #args.rank 
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        
        ## Single-Node Multi-GPU training (Rank = [0,#ngpus-per-node - 1]; world-size = 1)
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print(f"=> Rank of process is {args.rank}")
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    

    ## Suppress printing if not master
    is_master = args.rank == 0   # Check to see if global rank is 0

    ## Verify initialization
    if is_master: print(f"=> Group initialized? {dist.is_initialized()}", flush=True)


    ## Initialize Wandb (only on master node)
    if is_master and not args.debug:
        #initialize logger
        os.makedirs(args.log_root, exist_ok=True)
        os.environ['WANDB_DIR']=args.log_root
        os.environ['WANDB_START_METHOD']="thread"
        logger=log.Logger(args)
    else:
        logger=None
        if is_master:   print('=> You are working under [DEBUG] mode')


    ## Build model
    if is_master: print("=> Creating model '{}'".format(args.arch))
    model = builder.Model(
        models.__dict__[args.arch],  args.method,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    if args.sync_batchnorm == None:
        if not args.train_simclr: # Equivariant module
            args.sync_batchnorm = True
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #SyncBatchNorms for multiprocessing
        else:   args.sync_batchnorm = False

    #
    if is_master:   print(f'=> Using [BatchNormSync] across nodes: {args.sync_batchnorm}')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            if args.split_batch:
                total_batch_size =  args.batch_size
                args.batch_size = int(args.batch_size / args.world_size)
                if is_master:   print(f'=> Effective batch size on each GPU is [{total_batch_size}/{args.world_size}] = {args.batch_size}')
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        if args.method=='moco':
            # AllGather implementation (batch shuffle, queue update, etc.) in
            # this code only supports DistributedDataParallel.
            raise NotImplementedError("Only DistributedDataParallel is supported for MoCo training.")

    ## Training setup
    criterion = loss(model, args)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    ## Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # We only save the model that uses device "cuda:0", so load only that
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## Dataset generation and dataloaders
    # if 'imagenet100' in args.data.lower():
    #     args.num_classes = 100
    # elif 'imagenet' in args.data.lower():
    #     args.num_classes = 1000

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomApply([loader.TorchGaussianBlur(kernel_size = args.kernel_size, sigma = [.1, 2.])], p=0.5), # Using torch gaussian blur, [TODO]: play with kernel_size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    if is_master:   print(f'=> Using Torch Gaussian Blur with kernel size {args.kernel_size}')
    
    input_transforms = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), # resizing has already been done
                                      transforms.RandomGrayscale(p=0.2),
                                      # transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
                                      transforms.RandomApply([loader.TorchGaussianBlur(kernel_size = args.kernel_size, sigma = [.1, 2.])], p=0.5), # Using torch gaussian blur, [TODO]: play with kernel_size
                                      transforms.RandomHorizontalFlip(),
                                      normalize
                                      ])
    val_aug = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]

    train_dataset = datasets.ImageFolder(traindir, loader.TwoCropsTransform(base_transform = transforms.Compose(augmentation), std_transform = transforms.Compose(val_aug[0:3])))
    if args.distributed:
        # Restricts data loading to a subset of the dataset exclusive to the current process
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    ## Memory and Test loader don't have to follow distributed sampling strategy
    memory_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose(val_aug)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_aug)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if is_master:   print(f'=> Training using [{args.method}] with equivariant loss [{not args.train_simclr}]')
    ## Training module
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, val_loader, memory_loader, model, criterion, optimizer, epoch, input_transforms, args, logger)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            filename='checkpoint_{:04d}.pth.tar'.format(epoch)
            root = os.path.join(args.log_root, args.run_name)
            os.makedirs(root, exist_ok=True)

            filename='checkpoint_final.pth.tar' if (epoch==args.epochs-1) else 'checkpoint.pth.tar' 

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, root=root, filename=filename)



def train(train_loader, val_loader, memory_loader, model, criterion, optimizer, epoch, batch_transforms, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        # output1, target1 = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        ## Equivariance (RSSL) training module
        if not args.train_simclr:
            images2 = images[2][:int(args.equivariant_batch_size/args.world_size)]
            subset_size = args.equivariant_batch_size//args.equiv_splits_per_gpu
            batch_split = torch.split(images2, split_size_or_sections = subset_size, dim = 0)
            batch_split = torch.stack(batch_split, dim = 0) # Till here, 0.6sec/batch

            batch_1, batch_2 = vmap(process, (0, None), randomness = 'same')(batch_split, batch_transforms) # Till here,  ~0.7sec/batch

            batch_1 = batch_1.view(-1, batch_1.shape[2], batch_1.shape[3], batch_1.shape[4]).cuda(args.gpu, non_blocking=True)
            batch_2 = batch_2.view(-1, batch_2.shape[2], batch_2.shape[3], batch_2.shape[4]).cuda(args.gpu, non_blocking=True) # Till here, ~0.7sec/batch

            z_1 = model(im_q=batch_1, return_embedding=True)
            z_2 = model(im_q=batch_2, return_embedding=True)
            regularize_loss = equivariance_loss(z_1, z_2, device = args.gpu, num_chunks = args.equiv_splits_per_gpu)
            # regularize_loss = torch.tensor([0.]).cuda(args.gpu, non_blocking=True)
        ## SimCLR training module
        else:
            regularize_loss = torch.tensor([0.]).cuda(args.gpu, non_blocking=True)

        loss = loss + args.weight * regularize_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        if args.method=='moco': #can only record training accuraccy for MoCo runs
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = [0.], [0.]
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0: # Global rank
            if i % args.print_freq == 0:
                progress.display(i)
            if not args.debug and (i==len(train_loader)-1) and not args.sync_batchnorm: # Do single GPU inference if syncbatchnorm is false
                if args.rank == 0:  
                    print(f'=> Since SyncBatchNorm is {args.sync_batchnorm}, doing inference on only one GPUs')
                pos, neg = log.compute_similarities(model, im_q=images[0], im_k=images[1])

                info = {"train_loss": losses.avg,
                        "train_acc_1": acc1,
                        "train_acc_5": acc5,
                        "positive_cosine": pos,
                        "negative_cosine": neg,
                        "learning rate": log.extract_lr(optimizer)
                        }
                if not args.no_knn:
                    knn_top1, knn_top5 = log.knn(memory_loader, val_loader, model, args)
                    print(f'KNN Accuracy is [top@1]: {knn_top1} and [top@5]: {knn_top5}')
                    info["knn_top1"] = knn_top1
                    info["knn_top5"] = knn_top5
                if not args.debug: # log only on master node
                    print('=> Logging onto Wandb on GPU with Rank 0')
                    logger.log(info)

        if not args.debug and (i==len(train_loader)-1) and args.sync_batchnorm: # Multi GOU inference if sync batch norm is true
        # if not args.debug and args.sync_batchnorm: # Multi GOU inference if sync batch norm is true
            if args.rank == 0:  
                print(f'=> Since SyncBatchNorm is {args.sync_batchnorm}, doing inference across multiple GPUs')
            pos, neg = log.compute_similarities(model, im_q=images[0], im_k=images[1])
            info = {"train_loss": losses.avg,
                    "train_acc_1": acc1,
                    "train_acc_5": acc5,
                    "positive_cosine": pos,
                    "negative_cosine": neg,
                    "learning rate": log.extract_lr(optimizer)
                    }
            if not args.no_knn:
                knn_top1, knn_top5 = log.knn(memory_loader, val_loader, model, args)
                info["knn_top1"] = knn_top1
                info["knn_top5"] = knn_top5
            if not args.debug and args.rank==0: #log only on master node
                print('=> Logging onto Wandb on GPU with Rank 0')
                logger.log(info)

def process(batch, batch_transform):
    # Assuming input is a subset of the batch of shape (subset_size X 3 X H X W)
    batch_1 = batch_transform(batch.detach().clone()) # All images in the batch are transformed according to one set of parameters
    batch_2 = batch_transform(batch.detach().clone()) # All images in the batch are transformed according to another set of parameters
    return batch_1, batch_2


def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root,filename), \
                        os.path.join(root,'model_best.pth.tar'))




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_seed(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    print(f'=> Cuda available: {torch.cuda.is_available()}')
    main()
