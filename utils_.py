import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

import models
import datasets
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='experiement name', type=str, default='mixer_cifar10')
    parser.add_argument('--use_tb', type=str2bool, default=False)
    # TRAINING ARGUMENTS
    parser.add_argument('--model', help='Model'  , type=str, default='mixer'  , choices=['mixer', 'vit', 'TwoCNN'])
    parser.add_argument('--mcfg', help='Model Configuration', type=str, default='S'  , choices=['S', 'B', 'L'])
    parser.add_argument('--dataset', help='Dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'TinyImageNet', 'Celeba', 'STL10'])
    parser.add_argument('--loss', help='Loss function', type=str, default='CrossEntropyLoss')
    parser.add_argument('--optimizer', help='Optimizer', type=str, default='SGD')
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.015)
    parser.add_argument('--lrdecay', help='lr decay', type=float, default=0.99)
    parser.add_argument('--decay', help='Weight decay', type=float, default=5e-4)
    parser.add_argument('--momentum', help='momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--epochs', help='Epochs', type=int, default=100)
    parser.add_argument('--device', help='Device' , type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--ps', help='Patch size', type=int, default=4)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default=False)
    #ATTACK&DEFENSE ARGUMENT
    parser.add_argument('--attack_name', type=str, default=None)
    parser.add_argument('--eps', type=float, default=2)
    parser.add_argument('--p', type=float, default=0.3)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--defense', type=str, default='no')
    #FEDERATED LEARNING ARGUMENT   
    parser.add_argument('--n_user', help='number of total user', type=int, default=100)
    parser.add_argument('--local_epoch', help='local train epoch', type=int, default=10)
    parser.add_argument('--R', help='Total Round', type=int, default=100)
    parser.add_argument('--iid', help='iid or non-iid', type=str2bool, default=True)
    args = parser.parse_args()
    return args

def build_model(args):
    return getattr(models, args.model)(models.ModelCfg[args.model][args.dataset][args.mcfg])

def build_criterion(args):
    return getattr(nn, args.loss)()

def build_optimizer(model, args):
    return getattr(optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.decay) #, momentum=args.momentum)

def load_dataset(args, is_train):
    return getattr(datasets, args.dataset)(is_train)

def load_user_dataset(dataset):
    return getattr(datasets, 'DM')(dataset)

def calculate_topk_accuracy(predictions, targets, k):
    _, topk_preds = predictions.topk(k, dim=1)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    topk_acc = correct.any(dim=1).float().mean().item() * 100
    return topk_acc

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def np_counter(arr, val):
    return np.count_nonzero(arr==val)