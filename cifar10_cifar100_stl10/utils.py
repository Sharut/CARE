import numpy as np
import torch
import random
import os
from PIL import Image
import wandb
import torch.optim as optim
import torch
import inspect
import torchvision.transforms as T
import math

def set_seed(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f'=> Seed of the run set to {seed}')

def get_optimizer(optimizer_name, parameters, **kwargs):
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Filter any arguments pertaining to other optimizers
    optimizer_args = inspect.getfullargspec(optimizer_class.__init__).args[1:]
    valid_args = {k: v for k, v in kwargs.items() if k in optimizer_args}
    invalid_args = set(kwargs) - set(optimizer_args)

    # Just FYI
    for arg in invalid_args:
        print(f"Warning: Invalid argument for {optimizer_name} optimizer: {arg}")
    return optimizer_class(parameters, **valid_args)


def extract_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(type, step, epoch, loader, optimizer, args):
    """Sets the learning rate to the initial LR decayed by decay-rate every decay step"""
    if type == 'multistep':
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = args.lr * (args.lr_decay_rate ** steps)
    elif type == 'cosine':
        max_steps = args.epochs * len(loader)
        warmup_steps = args.warmup_epochs * len(loader)
        if step < warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = 0
            lr = args.lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def load_checkpoint(path, epoch=None, all_checkpoints=False, best_model=False):
    if os.path.exists(path):
        try :
            if best_model is True:
                file = os.path.join(path, 'best_checkpoint.pth.tar')
                checkpoint = torch.load(file)
                print("=> Loading pre-trained model '{}'".format(file))           
            elif (all_checkpoints is True) or (epoch is not None):
                file = os.path.join(path, 'checkpoint_ep{}.pth.tar'.format(epoch))
                checkpoint = torch.load(file)
                print("=> Loading pre-trained model '{}'".format(file))
            else:
                file = os.path.join(path, 'checkpoint.pth.tar')
                checkpoint = torch.load(file)
                print("=> Loading pre-trained model '{}'".format(file))
            return checkpoint

        except FileNotFoundError:
            raise AssertionError(f"Specified path to checkpoint {file} doesn't exist :(")
    else:
        raise AssertionError(f"Specified path to checkpoint {path} doesn't exist :(")
    return None

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def infoNCE(out_1, out_2, temperature, device):
    bs=out_1.shape[0]
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(bs).to(device)
    neg = neg.masked_select(mask).view(2 * bs, -1)
    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)  
    Ng = neg.sum(dim=-1)
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss, - torch.log(pos).mean(), torch.log(Ng).mean()


def get_diagonal_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
    return negative_mask

# Regularization loss for preserving simple linear group equivariance in latent space 
def equivariance_loss(out_1, out_2, device, num_chunks = 1):
    chunk_size = out_1.shape[0]//num_chunks
    loss = 0.
    for index in range(num_chunks):
        sub_1 = out_1[index*chunk_size: (index+1)*chunk_size]
        sub_2 = out_2[index*chunk_size: (index+1)*chunk_size]
        bs = sub_1.shape[0]
        # neg score
        t1 = torch.mm(sub_1, sub_1.t().contiguous())  # N X N matrix   
        t2 = torch.mm(sub_2, sub_2.t().contiguous())  # N X N matrix
        mask = get_diagonal_mask(bs).to(device)
        t1 = t1.masked_select(mask).view(bs, -1)
        t2 = t2.masked_select(mask).view(bs, -1)
        loss += 2*torch.norm(t1-t2, p='fro', dim=-1).pow(2).mean()
    loss /= num_chunks
    return loss




