import argparse
import os
import pandas
import wandb
import datetime
from torchvision import transforms
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import string
import random
from collections import ChainMap
from utils import *
import dataset
import log
from model import Model
import time
from functorch import vmap, make_functional, make_functional_with_buffers, grad_and_value
from functorch.experimental import replace_all_batch_norm_modules_
import sys
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))


## Process batch to apply same augmentation across entire batch
def process(batch):
    # Assuming input is a subset of the batch of shape (subset_size X 3 X H X W)
    batch_1 = batch.detach().clone()    
    batch_2 = batch.detach().clone()
    
    # Sample two augmentations
    t_1 = dataset.get_transforms(args)
    t_2 = dataset.get_transforms(args)

    # Apply same augrmentations to all images in the batch
    for index in range(len(t_1)):
        batch_1 = vmap(t_1[index], randomness = 'same')(batch_1)
        batch_2 = vmap(t_2[index], randomness = 'same')(batch_2)
    return batch_1, batch_2

## ==================================== Training module ====================================
def train(epoch, net, data_loader, train_optimizer, temperature, weight, train_simclr = False):
    net.train()
    total_loss, total_infonce_loss, total_reg_loss, total_num, train_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader, file = sys.stdout) # The default output of tqdm is stderr
    total_pos, total_neg = 0., 0.
    all_pos = []
    for batch_ind, (img, pos_1, pos_2, target) in enumerate(train_bar, start=(epoch - 1) * len(data_loader)):
        # Ajust learning rate
        if args.lr_schedule_type is not None:
            lr = adjust_learning_rate(type=args.lr_schedule_type,
                                      step=batch_ind,
                                      epoch=epoch,
                                      loader=data_loader,
                                      optimizer=optimizer,                   
                                      args = args)


        pos_1, pos_2 = pos_1.to(device,non_blocking=True, dtype=torch.float), pos_2.to(device,non_blocking=True, dtype=torch.float)
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        align_plus_uniform, align_loss, uniform_loss = infoNCE(out_1, out_2, temperature, device)
        if args.loss_subtype == 'infonce':
            infonce = align_plus_uniform
        elif args.loss_subtype == 'alignment':
            infonce = align_loss
        elif args.loss_subtype == 'uniformity':
            infonce = uniform_loss
        else:
            infonce = torch.tensor([0.]).to(device,non_blocking=True, dtype=torch.float)
            
        ## Orthogonal equivariance (CARE) training module
        if not train_simclr:
            subset_size = args.batch_size//args.equiv_splits
            # Split (B X 3 X H X W) tensor into (equiv_splits x subset_size X 3 X H X W)
            batch_split = torch.split(img, split_size_or_sections = subset_size, dim = 0)
            batch_split = torch.stack(batch_split, dim = 0)
            batch_1, batch_2 = vmap(process, randomness = 'same')(batch_split)

            # Concatenate dimensions of the splits into one tensor i.e. (equiv_splits x subset_size X 3 X H X W) to (B X 3 X H X W) 
            batch_1 = batch_1.view(-1, batch_1.shape[2], batch_1.shape[3], batch_1.shape[4]).to(device,non_blocking=True, dtype=torch.float)
            batch_2 = batch_2.view(-1, batch_2.shape[2], batch_2.shape[3], batch_2.shape[4]).to(device,non_blocking=True, dtype=torch.float)

            # Get embeddings for the concatenated views
            _, z_1 = net(batch_1)
            _, z_2 = net(batch_2)

            # Calculate regularize loss as an average of loss over each chunk
            regularize_loss = equivariance_loss(z_1, z_2, device, num_chunks = args.equiv_splits)
        
        ## SimCLR training module
        else:
            regularize_loss = torch.tensor([0.]).to(device,non_blocking=True, dtype=torch.float)

        # Combined loss of InfoNCE and Equivariance
        loss = infonce + weight * regularize_loss
    
        # Update model parameters
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # Compute performance metrics
        pos, neg, bs_pos = log.compute_similarities(out_1, out_2)
        all_pos.append(bs_pos)
        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        total_infonce_loss += infonce.item() * args.batch_size
        total_reg_loss += regularize_loss.item() * args.batch_size
        total_pos += pos.item() * args.batch_size
        total_neg += neg.item() * args.batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Step: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, batch_ind, args.epochs * len(data_loader), total_loss / total_num))

    all_pos = torch.cat(all_pos, dim=0).t().contiguous()
    return total_loss / total_num, total_infonce_loss / total_num, total_reg_loss / total_num,  total_pos / total_num, total_neg / total_num, all_pos


## ========================================= Testing module =========================================
## Testing module using weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():

        # Generate feature bank
        for batch_ind, (img, data, _, target) in enumerate(tqdm(memory_data_loader, desc='Feature extracting', file = sys.stdout)):
            feature, out = net(data.to(device, non_blocking=True, dtype=torch.float))
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()

        # Generate feature labels
        if 'cifar' in args.dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in args.dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

        # Weighted KNN based testing
        test_bar = tqdm(test_data_loader, file = sys.stdout)
        for batch_ind, (img, data, _, target) in enumerate(test_bar):
            data, target = data.to(device, non_blocking=True, dtype=torch.float), target.to(device, non_blocking=True, dtype=torch.float)
            feature, out = net(data)

            # Compute cos similarity between each feature vector and feature bank
            sim_matrix = torch.mm(feature, feature_bank)
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1) # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)   # [B, K]
            sim_weight = (sim_weight / args.temperature).exp()

            # Counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, c, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
            pred_labels = pred_scores.argsort(dim=-1, descending=True)

            # Update metrics
            total_num += data.size(0)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--seed', default=0, type=int, help='seed value for the run')
    
    ## Model configurations
    parser.add_argument('--dataset-name', default='cifar10', type=str, help='Choose data name', choices = ['cifar10', 'cifar100', 'stl10', 'flowers102', 'kaggle_flowers102'])
    parser.add_argument('--model', type=str, default='resnet18', help='Backbone architecture')
    parser.add_argument('--data-parallel', action='store_false', help='Data parallel training')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers in dataloader')
    # Optimizer specific arugments
    parser.add_argument('--optimizer', type=str, default='Adam', help='Type of optimizer - any pytorch acceptable optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr-schedule-type', type=str, default=None, help='Type of LR scheduler')
    parser.add_argument('--lr-decay-epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of epochs for warmup in LR scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='Weight decay rate of optimizer')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')

    parser.add_argument('--loss-subtype', type=str, default='infonce', help='whether to use infonce or alignment or uniformity loss')
    parser.add_argument('--feature-dim', default=128, type=int, help='Feature dim for latent vector - 2 in this code base')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch-size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--train-simclr', action='store_true', help='Train RSSL or SimCLR')
    parser.add_argument('--weight', default=0.01, type=float, help='weight to regularization')
    parser.add_argument('--equiv-splits', default=8, type=int, help='Number of batch subdivisions for equivariance')
    
    ## Checkpoint and path arguments
    parser.add_argument('--data-root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--save-root', default=' ./results', type=str, help='root directory where results are saved')
    parser.add_argument('--log-freq', default=1, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', default='', type=str, help='Path to pytorch model')
    parser.add_argument('--run_id', default='', type=str, help='Run ID for Wandb model to resume')
    parser.add_argument('--save-all-checkpoints', action='store_true', help='save all intermediate checkpoints.')
    parser.add_argument('--save-last', default=5, type=int, help='Save last N epoch models')
    parser.add_argument('--lin-eval', action='store_true', help='Load only labeled STL10 dataset')

    ## Wandb arguments
    parser.add_argument('--debug', action='store_true', help='do not run wandb logging')
    parser.add_argument('--run-name', default='', type=str, help='Choose name of wandb run')
    parser.add_argument('--project', default='project_name', type=str, help='wandb project dataset_name')
    parser.add_argument('--user', default='sample_user', type=str, help='wandb username')

    ## Parse arguments
    args = parser.parse_args()

    ## Set seed
    set_seed(args.seed, torch.cuda.is_available())

    args.lin_eval=False  # set to false to ensure use of all unlabled data during pretraining.
    
    ## Initialize Wandb
    if(args.run_name == ''):    args.run_name = ''.join(random.choice(string.ascii_letters) for i in range(10))
    args.run_name = args.run_name + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    ## Path to save run artifacts
    root = os.path.join(args.save_root, args.run_name)   
    os.makedirs(root, exist_ok=True)
    os.environ['WANDB_DIR'] = root
    if not args.debug:
        logger=log.Logger(args)

    ## Prepare dataset
    train_data, memory_data, test_data = dataset.get_dataset(args.dataset_name, root=args.data_root, args = args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  pin_memory=True, drop_last=True, num_workers=args.num_workers)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False,  pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,  pin_memory=True, num_workers=args.num_workers)

    ## Model setup and optimizer config
    model = Model(args).to(device)
    if args.data_parallel:
        print('=> Initializing data parallel')
        model = nn.DataParallel(model)

    # Extract optimizer using name
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr= args.lr, weight_decay= args.weight_decay, betas = (args.beta_1, args.beta_2))

    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    ## Resume from saved checkpoint
    if args.resume:
        ckpt = load_checkpoint(args.resume_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_accuracy = ckpt['test_acc_1']
        print('=> Resuming from epoch', start_epoch)
    else:
        start_epoch = 1

    ## Parse decay epochs for scheduler
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    ## Model training loop
    best_accuracy, count = 0., 0
    for epoch in range(start_epoch, args.epochs + 1):
        ## Training
        train_loss, infonce_loss, regularize_loss, avg_pos, avg_neg, all_pos = train(epoch, model, train_loader, optimizer, args.temperature, args.weight, train_simclr = args.train_simclr)
        
        if epoch % args.log_freq == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)

            ## Save checkpoint
            if args.save_all_checkpoints:   filename=os.path.join(root, 'checkpoint_ep' + str(epoch) + '.pth.tar')
            elif args.save_last + epoch > args.epochs:
                filename=os.path.join(root, 'checkpoint_ep' + str(epoch) + '.pth.tar')
                print(f'=> Saving last models [{count+1}/{args.save_last}] for epoch #{epoch}');    count+=1;
            else:   filename=os.path.join(root, 'checkpoint.pth.tar') # Save this just in case we want to resume training

            model_dict = {'epoch': epoch,
                          'args': args,
                          'best_acc_1': best_accuracy,
                          'test_acc_1': test_acc_1,
                          'state_dict': model.state_dict(),
                          'optimizer' : optimizer.state_dict()
                           }
            save_checkpoint(model_dict, filename=filename)

            if (test_acc_1-best_accuracy) >  1e-2:
                best_accuracy = test_acc_1
                model_dict['best_acc_1'] = best_accuracy
                print(f'=> Saving the model with best top-1 KNN accuracy at epoch {epoch}')
                save_checkpoint(model_dict, filename=os.path.join(root, 'best_checkpoint.pth.tar'))

            ## Record Wandb metrics
            info = {"train_loss": train_loss,
                    "infonce_loss": infonce_loss,
                    "regularize_loss": regularize_loss,
                    "test_acc_1": test_acc_1,
                    "test_acc_5": test_acc_5,
                    "positive_cosine": avg_pos,
                    "negative_cosine": avg_neg,
                    "learning rate": extract_lr(optimizer)
                    }

            if not args.debug:
                logger.log(info)


