import wandb
from datetime import date

import torch.nn.functional as F
import torch.nn as nn
import torch

class  Logger(object):
    def __init__(self, args, tags=None):
        super(Logger, self).__init__()
        print("=> Project is", args.project)
        self.args=args
        tags=[args.user, tags] if tags is not None else [args.user]
        if args.resume:
            self.run = wandb.init(project=args.project, id = args.run_id, entity=args.user, resume="must", tags=tags)
        elif not args.debug:
            self.run = wandb.init(project=args.project, name = self.args.run_name, entity=args.user, reinit=True, tags=tags)
        config = wandb.config 
        curr_date = date.today()
        curr_date = curr_date.strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True) 
        wandb.config.update(args, allow_val_change=True) 
           

    def log(self, info):
        if not self.args.debug:
            wandb.log(info)


    def finish(self):
        if not self.args.debug:
            self.run.finish()


def extract_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def compute_similarities(model, im_q, im_k):
    encoder = model.module.encoder_q

    with torch.no_grad():
        k = encoder(im_k)        
        q = encoder(im_q)

        pos = nn.CosineSimilarity()(k, q).mean()

        k_and_q = torch.cat([k, q], dim=0)
        k_and_q = F.normalize(k_and_q, dim=-1)

        neg = (k_and_q @ k_and_q.T).mean()

        return pos, neg



def knn(train_loader, val_loader, model, args):

    encoder = model.module.encoder_q
    #remove fc layer at end
    modules=[]
    for name, module in encoder.named_children():
        if not name=='fc':
            modules.append(module)

    encoder=nn.Sequential(*modules)

    #number of classes
    c=100

    total_top1, total_top5, total_num, feature_bank, train_labels = 0.0, 0.0, 0, [], []

    with torch.no_grad():
        # generate feature bank
        for ind, (data, target) in enumerate(train_loader):
            target=target.cuda()
            data=data.cuda()

            feature = encoder(data)
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=-1)
            feature_bank.append(feature)
            train_labels.append(target)
            # if args.rank ==0 and ind % 1000 == 0:
            #     print(f'Creating feature bank, currently at batch {ind}/{len(train_loader)}')

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        train_labels = torch.cat(train_labels, dim=0).t().contiguous()


        for ind, (data, target) in enumerate(val_loader):
            # print('KNN Testing', ind)
            target=target.cuda()
            data=data.cuda()

            feature = encoder(data)
            feature = torch.flatten(feature, start_dim=1)
            feature=F.normalize(feature, dim=-1)


            total_num += data.size(0)

            sim_matrix = torch.mm(feature, feature_bank)
            
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(train_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            #breakpoint()
            sim_weight = (sim_weight / args.moco_t).exp()
            #breakpoint()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.knn_k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # if args.rank == 0: # Global rank
            #     if ind % 500 == 0:
            #         print('KNN Test Batch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
            #                          .format(ind, len(val_loader), total_top1 / total_num * 100, total_top5 / total_num * 100))
        
        return total_top1 / total_num, total_top5 / total_num

