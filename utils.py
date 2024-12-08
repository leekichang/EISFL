import io
import os
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import models
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name'    , help='experiement name' , type=str     , default='EISFL')
    parser.add_argument('--model'       , help='model'            , type=str     , default='MLP')
    parser.add_argument('--dataset'     , help='dataset'          , type=str     , default='MNIST')
    parser.add_argument('--optimizer'   , help='optimizer'        , type=str     , default='SGD')
    parser.add_argument('--lr'          , help='learning rate'    , type=float   , default=1e-3)
    parser.add_argument('--decay'       , help='weight decay'     , type=float   , default=1e-4)
    parser.add_argument('--batch_size'  , help='batch size'       , type=int     , default=64)
    parser.add_argument('--seed'        , help='random seed'      , type=int     , default=0)
    parser.add_argument('--epoch'       , help='number of epochs' , type=int     , default=5)
    parser.add_argument('--use_tb'      , help='use tensorboard'  , type=str2bool, default=False)
    ### FL PARAMS ###
    parser.add_argument('--n_clients'   , help='number of clients', type=int     , default=10)
    parser.add_argument('--rounds'      , help='number of clients', type=int     , default=100)
    parser.add_argument('--alpha'       , help='Dirichlet alpha'  , type=float   , default=0.5)
    parser.add_argument('--p_ratio'     , help='Participant ratio', type=float   , default=0.2)
    parser.add_argument('--client'      , help='Client type'      , type=str     , default='NaiveClient')
    parser.add_argument('--aggregator'  , help='Server aggregator', type=str     , default='FedAvg')
    parser.add_argument('--trim_frac'   , help='Trim-mean frac'   , type=float   , default=0.2)
    ### DP PARAMS ###
    parser.add_argument('--defense'     , help='defense'          , type=str     , default='NoDefense')
    parser.add_argument('--epsilon'     , help='epsilon'          , type=float   , default=8.0)
    parser.add_argument('--delta'       , help='delta'            , type=float   , default=1e-5)
    parser.add_argument('--clip_norm'   , help='clip norm'        , type=float   , default=3.0)
    ### GP PARAMS ###
    parser.add_argument('--prune_ratio' , help='pruning ratio'    , type=float   , default=0.9)
    ### ATTACK PARAMS ###
    parser.add_argument('--atk_type'    , help='Attack type'      , type=str     , default='LabelFlip')
    parser.add_argument('--atk_ratio'   , help='Attack ratio'     , type=float   , default=0.2)
    parser.add_argument('--flip'        , help='flip label'       , type=int     , default=9)
    parser.add_argument('--backdoor_opt', help='backdoor option'  , type=str     , default='left')
    parser.add_argument('--backdoor_tar', help='backdoor target'  , type=int     , default=0)
    ### EISFL PARAMS ###
    parser.add_argument('--shuffle'     , help='shuffle'          , type=str2bool, default=False)
    parser.add_argument('--n_val'       , help='# of val samples' , type=int     , default=1)
    args = parser.parse_args()
    return args
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_model(args):
    return getattr(models, args.model)(n_class=cfg.N_CLASS[args.dataset])  

def build_criterion(args):
    return getattr(torch.nn, 'CrossEntropyLoss')()

def build_optimizer(model, args):
    return getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.decay)

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def print_info(args):
    def print_section(title, content):
        print(f"\n<{title}>")
        print("="*20)
        for name, value in content:
            print(f"{name:<20} {value}")
        print("="*20)

    experiment_params = [
        ('Experiment Name:', args.exp_name),
        ('Model:', args.model),
        ('Dataset:', args.dataset),
        ('Optimizer:', args.optimizer),
        ('Learning Rate:', args.lr),
        ('Weight Decay:', args.decay),
        ('Batch Size:', args.batch_size),
        ('Random Seed:', args.seed),
        ('Number of Epochs:', args.epoch),
        ('Use Tensorboard:', args.use_tb),
    ]
    
    fl_params = [
        ('Number of Clients:', args.n_clients),
        ('Number of Rounds:', args.rounds),
        ('Dirichlet Alpha:', args.alpha),
        ('Participant Ratio:', args.p_ratio),
        ('Client type:', args.client),
        ('Server Aggregator:', args.aggregator),
        ('Trim-mean Fraction:', args.trim_frac),  # 추가된 항목
        ('Defense:', args.defense),  # 추가된 항목
    ]
    
    dp_params = [
        ('Epsilon:', args.epsilon),
        ('Delta:', args.delta),
        ('Clip Norm:', args.clip_norm),
    ]
    
    attack_params = [
        ('Attack Type:', args.atk_type),
        ('Attack Ratio:', args.atk_ratio),
        ('Flip Label:', args.flip),
        ('Backdoor Option:', args.backdoor_opt),
        ('Backdoor Target:', args.backdoor_tar),
    ]
    
    eisfl_params = [
        ('Shuffle:', args.shuffle),
        ('Number of Validation Samples:', args.n_val),
    ]
    
    gp_params = [
        ('Pruning Ratio:', args.prune_ratio),  # 추가된 항목
    ]
    
    print_section("Experiment Configuration", experiment_params)
    print_section("Federated Learning Parameters", fl_params)
    print_section("Differential Privacy Parameters", dp_params)
    print_section("Attack Parameters", attack_params)
    print_section("EISFL Parameters", eisfl_params)
    print_section("Global Pruning Parameters", gp_params)  # 새로운 섹션

    
def format_args(args):
    return "|".join([
        args.exp_name,
        args.model,
        args.dataset,
        args.optimizer,
        str(args.lr),
        str(args.decay),
        str(args.batch_size),
        str(args.seed),
        str(args.epoch),
        str(args.use_tb),
        str(args.n_clients),
        str(args.rounds),
        str(args.alpha),
        str(args.p_ratio),
        args.client,
        args.aggregator,
        str(args.trim_frac),
        str(args.epsilon),
        str(args.delta),
        str(args.clip_norm),
        args.defense,  # 추가된 항목
        args.atk_type,
        str(args.atk_ratio),
        str(args.flip),
        args.backdoor_opt,
        str(args.backdoor_tar),
        str(args.shuffle),
        str(args.n_val),
        str(args.prune_ratio)
    ])




def plot_cluster_history(history, tag=''):
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = history == -1
    cmap = mpl.cm.get_cmap('tab20')
    cmap.set_bad("k")
    # Plot the heatmap
    sns.heatmap(history, cmap=cmap, cbar=True, ax=ax, mask=mask)
    
    # Set labels and title
    ax.set_xlabel('Round')
    ax.set_ylabel('Client ID')
    ax.set_title('Cluster Assignment History')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(f'./cluster_history_{tag}.png', format='png')
    plt.close(fig)

def np_counter(arr, val):
    return np.count_nonzero(arr==val)