import os
import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import copy
import torch
from tqdm import tqdm
import tensorboard as tb
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

import ops
import utils
import config as cfg
from models import *
from Clients import *
from DataManager import *
from Servers import BaseServer

__all__ = ['FLTrustServer']

class FLTrustServer(BaseServer):
    def __init__(self, args):
        super(FLTrustServer, self).__init__(args)
        self.n_samples = 1000
        self.g0 = copy.deepcopy(self.global_model)
        self.make_server_dataset()

    def make_server_dataset(self):
        idx = torch.randperm(len(self.trainset))[:self.n_samples]
        n_samples = self.n_samples//self.trainset.targets.max()
        for n in range(self.trainset.targets.max()):
            idx_ = idx[self.trainset.targets[idx] == n][:n_samples]
            if n == 0:
                idx0 = idx_
            else:
                idx0 = torch.cat([idx0, idx_])
        self.trainset.data = self.trainset.data[idx_]
        self.trainset.targets = self.trainset.targets[idx_]
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.n_samples, shuffle=True)
    
    def train_global_model(self):
        self.g0.load_state_dict(self.global_model.state_dict())
        self.g0.to(self.device)
        self.g0_optim = torch.optim.Adam(self.g0.parameters(), lr=self.args.lr, weight_decay=self.args.lr*0.1)
        for epoch in range(self.args.epoch):
            self.g0.train()
            for i, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.g0(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.g0_optim.step()
                self.g0_optim.zero_grad()
        self.g0.to('cpu')
        self.g0.eval()
        self.global_update      = self.calc_update(self.g0)
        self.global_update_vec  = self.update2vec(self.global_update)
        self.global_update_norm = self.calc_norm(self.global_update)

    def calc_update(self, model):
        update_dict = {k:v for k, v in self.global_model.state_dict().items()}
        for k, v in model.state_dict().items():
            update_dict[k] = v - update_dict[k]
        return update_dict
    
    def update2vec(self, update):
        vec = []
        for k, v in update.items():
            vec.append(v.view(-1))
        return torch.cat(vec)[None,:]
    
    def cosine_similarity(self, w1, w2):
        return F.cosine_similarity(w1, w2)
    
    def relu(self, x):
        return max(0, x)
    
    def trust_score(self, model):
        update = self.calc_update(model)
        update_vec = self.update2vec(update)
        cossine_sim = self.cosine_similarity(update_vec, self.global_update_vec)
        return self.relu(cossine_sim)
    
    def trust_score_lst(self, sampled_clients):
        tslst = np.zeros(len(sampled_clients))
        for i, client in enumerate(sampled_clients):
            tslst[i] = self.trust_score(self.clients[client].model)
        return tslst

    def calc_norm(self, update):
        vec = self.update2vec(update)
        return torch.norm(vec, p=2)
    
    def calc_norm_lst(self, sampled_clients):
        normlst = np.zeros(len(sampled_clients))
        for i, client in enumerate(sampled_clients):
            update = self.calc_update(self.clients[client].model)
            normlst[i] = self.calc_norm(update)
        return np.array(normlst)

    def run(self):
        for r in range(self.args.rounds):
            self.train_global_model()
            sampled_clients = self.sample_clients(int(self.args.n_clients*self.args.p_ratio))
            for client in sampled_clients:
                self.clients[client].train()
                self.clients[client].test()
            tslst = self.trust_score_lst(sampled_clients)
            if sum(tslst) < 0.1:
                tslst = np.ones(len(sampled_clients))/len(sampled_clients)
            updates = []
            for client in sampled_clients:
                updates.append(self.calc_update(self.clients[client].model))
            norm_lst = self.calc_norm_lst(sampled_clients)/(self.global_update_norm+1e-8)

            # for idx, (ts, update) in enumerate(zip(tslst, updates)):
            #     print(f'{ts:.4f} {norm_lst[idx]:.4f}')
            #     print(f"Client {sampled_clients[idx]}: Trust Score {ts/(sum(tslst)*norm_lst[idx]+1e-9):.4f}")

            aggregated_update = []
            for idx, (ts, update) in enumerate(zip(tslst, updates)):
                # aggregated_update.append({k:ts*v/(sum(tslst)*norm_lst[idx]+1e-8) for k, v in update.items()})
                aggregated_update.append({k:ts*v/(sum(tslst)*norm_lst[idx] + 1e-9) for k, v in update.items()})
            new_state_dict = {k:v for k, v in self.global_model.state_dict().items()}
            for k, v in new_state_dict.items():
                new_state_dict[k] = v + sum([update[k] for update in aggregated_update])
            self.global_model.load_state_dict(new_state_dict)
            self.global_test(r)
            self.dispatch()

if __name__ == '__main__':
    import utils
    args = utils.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    server = FLTrustServer(args)
    server.run()
    