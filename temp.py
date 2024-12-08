# %%
import torch
import torch.nn 
import numpy as np
import torchvision
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F

import Servers

# %%
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name'    , help='experiement name' , type=str     , default='EXPFL')
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
    parser.add_argument('--epsilon'     , help='epsilon'          , type=float   , default=2.0)
    parser.add_argument('--delta'       , help='delta'            , type=float   , default=1e-2)
    parser.add_argument('--clip_norm'   , help='clip norm'        , type=float   , default=2.0)
    ### GP PARAMS ###
    parser.add_argument('--prune_ratio' , help='pruning ratio'    , type=float   , default=0.9)
    ### ATTACK PARAMS ###
    parser.add_argument('--atk_type'    , help='Attack type'      , type=str     , default='LabelFlip')
    parser.add_argument('--atk_ratio'   , help='Attack ratio'     , type=float   , default=0.2)
    parser.add_argument('--flip'        , help='flip label'       , type=int     , default=1)
    parser.add_argument('--backdoor_opt', help='backdoor option'  , type=str     , default='left')
    parser.add_argument('--backdoor_tar', help='backdoor target'  , type=int     , default=0)
    ### EISFL PARAMS ###
    parser.add_argument('--shuffle'     , help='shuffle'          , type=str2bool, default=False)
    parser.add_argument('--n_val'       , help='# of val samples' , type=int     , default=1)
    args = parser.parse_args([])
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
    
args = parse_args()

# %%
args.seed = 1
torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(suppress=True, precision=4)
args.n_clients = 50
args.rounds = 50
args.p_ratio = 0.2
# args.atk_type = 'Backdoor'
# args.atk_type = 'NoiseAdd'
args.atk_type = 'LabelFlip'
args.atk_ratio = 0.2
args.alpha = 0.5
args.epoch = 5
args.aggregator = 'WeightedSum'
server = Servers.EISFLServer(args)
threshold = 0.5
selection_history = []
for i in range(args.rounds):
    accs, losses = [], []
    sampled_clients = np.random.choice(range(50), 10, replace=False)
    sampled_clients.sort()
    selection_history = np.concatenate([selection_history, sampled_clients])
    for idx in sampled_clients:
        server.clients[idx].train()
    val_samples = []
    val_labels  = []
    for idx in tqdm(sampled_clients):
        server.clients[idx].shuffle_weight(round=0)
        val_sample, val_label = server.clients[idx].upload_sample(n_samples=server.args.n_val)
        val_samples.append(val_sample)
        val_labels.append(val_label)
    server.val_samples = torch.cat(val_samples, dim=0)/255
    server.val_labels  = torch.cat(val_labels, dim=0)
    
    weights = server.integrity_check(sampled_clients)
    server.update_cluster(weights, sampled_clients, threshold=threshold)
    server.clean_cluster()
    for c in sampled_clients:
        print(c, server.user_cluster[c])
    server.aggregate(sampled_clients, weights=weights, args=args)
    server.dispatch()
    # for c in server.global_models:
    #     if c != -1:
    #         server.global_model.load_state_dict(server.global_models[c].state_dict())
    #         print(f"CLUSTER {c}")
    #         server.global_test(i)
    #         if args.atk_type == 'Backdoor':
    #             server.backdoor_test(i)
    #         print('#'*10)
    threshold = min(0.95, threshold * 1.02)
    for client in range(server.n_clients):
        server.cluster_history[client,i] = server.user_cluster[client]
    import utils
    np.save('./figure/cluster_history_LF.npy', server.cluster_history)
    utils.plot_cluster_history(server.cluster_history, tag='LF')
np.save('./figure/selection_history.npy', selection_history)

# # %%
# cluster_set = []
# plt.figure(figsize=(12,6))
# for i in server.user_cluster:
#     cluster_set.append(server.user_cluster[i])
# cluster_set = set(cluster_set)
# lens = {c:idx for idx, c in enumerate(cluster_set)}

# for i, idx in enumerate(server.user_cluster):
#     c = 'blue' if i < args.n_clients-int(args.n_clients*args.atk_ratio) else 'red'
#     plt.scatter(i, lens[server.user_cluster[i]], color=c, s=100)

# cluster_set = set(cluster_set)
# print(list(cluster_set))
# plt.xticks(np.arange(50))
# plt.yticks(range(len(cluster_set)), list(cluster_set))
# plt.savefig('cluster.png')
