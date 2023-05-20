from __future__ import print_function
import torch
import torch.nn as nn

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
 
def get_info(model):
    try:
        model = model.module
    except:
        model = model
    return model

def loss(model, args):
    if get_info(model).method=='moco':
        return nn.CrossEntropyLoss().cuda(args.gpu)
    elif get_info(model).method=='simclr':
        def simclr(out_1, out_2):
            bs=out_1.shape[0]
            # neg score
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / model.module.T)
            old_neg = neg.clone()
            mask = get_negative_mask(bs).cuda()
            neg = neg.masked_select(mask).view(2 * bs, -1)

            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / model.module.T)
            pos = torch.cat([pos, pos], dim=0)  

            Ng = neg.sum(dim=-1)

            # contrastive InfoNCE loss
            loss = (- torch.log(pos / (pos + Ng) )).mean()

            return loss

        return simclr
    else:
        raise NotImplementedError(f"Training method {get_info(model).method} is not supported.")


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
        mask = get_diagonal_mask(bs).cuda(device, non_blocking=True)
        t1 = t1.masked_select(mask).view(bs, -1)
        t2 = t2.masked_select(mask).view(bs, -1)
        loss += 2*torch.norm(t1-t2, p='fro', dim=-1).pow(2).mean()
    loss /= num_chunks
    return loss