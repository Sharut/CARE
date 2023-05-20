import wandb
from datetime import date

import torch.nn.functional as F
import torch.nn as nn
import torch
class  Logger(object):
    def __init__(self, args, tags=None):
        super(Logger, self).__init__()
        print("Project is", args.project)
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


def compute_similarities(z1, z2):
        z1=z1.detach()
        z2=z2.detach()

        all_pos = nn.CosineSimilarity()(z1, z2)
        pos = all_pos.mean()

        z1_and_z2 = torch.cat([z1, z2], dim=0)
        z1_and_z2 = F.normalize(z1_and_z2, dim=-1)

        all_neg = (z1_and_z2 @ z1_and_z2.T)
        neg = all_neg.mean()

        return pos, neg, all_pos
