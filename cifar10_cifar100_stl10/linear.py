import argparse
import datetime
import wandb
from tqdm import tqdm
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset
from utils import *
from model import Model
import log

emb_dim = {'resnet18': 512, 'resnet50': 2048}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        print(f'=> Pretrained path is {pretrained_path}')
        model = Model(args).to(device)
        model = nn.DataParallel(model)
        if pretrained_path == 'random-init':
            print('=> Randomly initializing the model')
        else:
            ckpt = load_checkpoint(args.model_path, epoch=args.load_epoch, best_model=args.best_model)
            print('=> Loaded model statistics [top@1 KNN accuracy]:', ckpt['test_acc_1'], '[best top@1 KNN accuracy]:', ckpt['best_acc_1'])
            model.load_state_dict(ckpt['state_dict'])
    
        if isinstance(model, nn.DataParallel):
            self.f = model.module.f
        else:
            self.f = model.f
        self.fc = nn.Linear(emb_dim[args.model], num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    # Data args
    parser.add_argument('--seed', default=0, type=int, help='seed value for the run')
    parser.add_argument('--dataset-name', default='cifar10', type=str, help='Choose loss function')
    parser.add_argument('--data-root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--save-root', default=' ./results', type=str, help='root directory where results are saved')
    parser.add_argument('--num-workers', default=8, type=int, help='Worker count for dataloader')

    #model and training args
    parser.add_argument('--model', type=str, default='resnet18', help='Backbone architecture')
    parser.add_argument('--best-model', action='store_true', help='load model with best kNN')
    parser.add_argument('--model-path', type=str, default='./results/sample_model_path',
                        help='The pretrained model path')
    parser.add_argument('--batch-size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--load-epoch', type=int, default=None, help='Number of sweeps over the dataset to train')
    parser.add_argument('--feature-dim', default=128, type=int, help='Feature dim for latent vector - 2 in this code base')

    ## Wandb arguments
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='do not run wandb logging')
    parser.add_argument('--random-features', action='store_true', help='do not run wandb logging')
    parser.add_argument('--lin-eval', action='store_true', help='Load only labeled STL10 dataset')
    parser.add_argument('--run-name', default='lin-eval', type=str, help='Choose name of wandb run')
    parser.add_argument('--project', default='project_name', type=str, help='wandb project dataset_name')
    parser.add_argument('--user', default='sample_user', type=str, help='wandb username')

    args = parser.parse_args()

    ## Set seed
    set_seed(args.seed, torch.cuda.is_available())

    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    dataset_name = args.dataset_name
    
    if(args.run_name == ''):    args.run_name = ''.join(random.choice(string.ascii_letters) for i in range(10))
    model_name=args.model_path.split('/')[-1]
    args.run_name = args.run_name+'-best'+ '-' + str(args.best_model)
    args.run_name = args.run_name + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    ## Path to save run artifacts
    root = os.path.join(args.save_root, args.run_name)   
    os.makedirs(root, exist_ok=True)
    print(f'=> Saving logs at {root}')
    os.environ['WANDB_DIR'] = root
    if not args.debug:
        logger=log.Logger(args, tags='lin-eval')

    train_data, _, test_data = dataset.get_dataset(dataset_name, root=args.data_root, args=args, pair=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).to(device)
    for param in model.f.parameters():
        param.requires_grad = False
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.module.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        if epoch % 5 == 0:
            test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)

            ## Record Wandb metrics
            info = {"train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc_1": train_acc_1,
                    "train_acc_5": train_acc_5,                    
                    "test_acc_1": test_acc_1,
                    "test_acc_5": test_acc_5,
                    "learning rate": extract_lr(optimizer)}

            if not args.debug:
                logger.log(info)
                print("=> Logged data onto wandb...")

