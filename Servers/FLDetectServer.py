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
from defenses.detector import FLDetector

__all__ = ['FLDetectServer']

class FLDetectServer(BaseServer):
    def __init__(self, args, window_size=10):
        super(FLDetectServer, self).__init__(args)
        self.detector = FLDetector.FLDetector(args, window_size=window_size)
        self.start_round = self.args.rounds // 2

    def concat_params(self, model):
        copied_weights = []
        for _, param in model.state_dict().items():
            copied_weights.append(copy.deepcopy(param))
        round_t_weight = torch.cat([param.reshape(-1, 1) for param in copied_weights], dim=0)
        return round_t_weight

    def run(self, save_period=None):
        for round in range(self.args.rounds):
            sampled_clients = self.sample_clients(int(self.args.p_ratio*self.n_clients))
            # accs, losses = [], []
            for client in sampled_clients:
                self.clients[client].train()
                loss, acc = self.clients[client].test()
            
            """
            Global Model
            """
            round_global_weight = self.concat_params(self.global_model)
            # Global Model Weight History
            self.detector.global_weights.append(round_global_weight)
            # Global Model Update History
            # Assume SGD for global updates
            # if self.round == 0:
            #     self.detector.global_updates.append(torch.zeros_like(round_global_weight))
            # else:
            #     self.detector.global_updates.append((self.detector.global_weights[-2] - round_global_weight) / self.args.lr)
            self.detector.global_updates.append((self.detector.global_weights[-2] - round_global_weight) / self.args.lr)
            print(f"Global weight: {self.detector.global_weights[-1].shape}")
            print(f"Global update: {self.detector.global_updates[-1].shape}")

            """
            Local Models
            """
            # Local Model Weight
            round_local_weights = []
            for client in self.clients:
                client_weight = self.concat_params(self.clients[client].model)
                round_local_weights.append(client_weight)
            round_cat_local_weight = torch.cat(round_local_weights, dim=1)
            self.detector.local_weights.append(round_cat_local_weight)
            # Local Model Update
            # Assume SGD for local updates
            # if self.round == 0:
            #     self.detector.local_updates.append(torch.zeros_like(round_cat_local_weight))
            # else:
            #     self.detector.local_updates.append((self.detector.local_weights[-2] - round_cat_local_weight) / self.args.lr)
            self.detector.local_updates.append((self.detector.local_weights[-2] - round_cat_local_weight) / self.args.lr)
            #adversary_list = self.detector.detect_fl(self.round)
            #if len(adversary_list) != 0:
            
            adversary_list = self.detector.detect_fl()
            if len(adversary_list) <= (len(self.clients) // 2):
                sampled_clients -= adversary_list
            
            self.aggregate(sampled_clients)

            self.global_test(r=round)
            if self.args.atk_type=='Backdoor':
                self.backdoor_test(r=round)
            if save_period and (round+1) % save_period == 0:
                self.save_global_model(round+1)
            self.dispatch()
            self.round += 1 # TODO: FIX THIS
            print(f"####### ROUND {round+1} END #######\n")

if __name__ == '__main__':
    import utils
    args = utils.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    server = FLDetectServer(args)
    server.run()