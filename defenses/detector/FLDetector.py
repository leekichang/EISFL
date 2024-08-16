import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count
from collections import defaultdict, OrderedDict
import torchvision.transforms as transforms
import ops
import utils
import config as cfg

from sklearn.cluster import KMeans

import torch.utils.tensorboard as tb

__all__ = ['FLDetector']

NUM_FLDETECTOR_CLUSTER = 2
INIT_ROUND             = 0
EPS                    = 1e-8

class FLDetector(object):
    def __init__(self,
                 args,
                 atk_args,
                 window_size=5,
                 num_sample=5,
                 max_cluster=10
                 ):
        """
        Weight          : Model parameter. Concatenated as a column vector.
        Update          : Model gradient. Calculated by subtraction of weights and division by learning rate.

        LBFGS - Predict Hessian matrix for gradient update
        global_weights  : Global model parameter history per round (# Param x 1)
        global_updates  : Global model parameter update history per round
        window_size     : Storage capacity for calculation

        Gap Statistics, Detection
        local_weights   : Local model parameter history per round, per client (# Param x # Client)
        local_updates   : Local model parameter update history per round, per client
        """
        assert window_size > 1

        self.args                   = args
        self.atk_args               = atk_args

        self.iterations             = args.epochs
        self.window_size            = window_size
        self.num_cluster_fl         = NUM_FLDETECTOR_CLUSTER
        self.num_sample             = num_sample
        self.max_cluster            = max_cluster
        self.round                  = INIT_ROUND

        self.global_weights         = []
        self.global_updates         = []
        self.local_weights          = []
        self.local_updates          = []
        
        self.global_weights_diff    = []
        self.global_updates_diff    = []
        self.curr_weight_diff       = None

        self.pred_Ht_v              = None
        self.debug_mode             = False

    def show_status(self):
        """
        For Debug
        """
        if self.debug_mode == False:
            return
        else:
            print("\n---------------------------------")
            print("        FLDetector STATUS        ")
            print("---------------------------------")
            print("- Current Round")
            print(self.round)
            print(f"- Global Weights({len(self.global_weights)})")
            print(self.global_weights)
            print(f"- Global Weights Diff({len(self.global_weights_diff)})")
            print(self.global_weights_diff)
            print(f"- Global Updates({len(self.global_updates)})")
            print(self.global_updates)
            print(f"- Global Updates Diff({len(self.global_updates_diff)})")
            print(self.global_updates_diff) 
            print("- Current Weight Diff (wt - wt-1)")
            print(self.curr_weight_diff)
            print("- Predicted Hessian")
            print(self.pred_Ht_v)
            print("---------------------------------\n")

    # Compute predicted Hessian matrix H_hat
    def lbfgs(self, iter):
        """
        Input : Global Weight Diff List(Wt: t-N ~ t-1), Global Weight "Update" Diff List(Gt: t-N ~ t-1),
                Global Weight Diff (v = wt - wt-1), Window Size (N)
        Output: Hessian vector product H_hat * v
        """
        # iter(t) -> iter-N ~ iter-1
        self.global_weights_diff = []
        self.global_updates_diff = []
        for t in range(max(iter-self.window_size, 0), iter):
            self.global_weights_diff.append(self.global_weights[t] - self.global_weights[t-1])
            self.global_updates_diff.append(self.global_updates[t] - self.global_updates[t-1])
        
        if len(self.global_weights_diff) == 0:
            self.global_weights_diff.append(torch.zeros_like(self.global_weights[0], device=self.args.device))
        if len(self.global_updates_diff) == 0:
            self.global_updates_diff.append(torch.zeros_like(self.global_updates[0], device=self.args.device))

        self.curr_weight_diff = self.global_weights[iter] - self.global_weights[iter-1]
        self.curr_weight_diff = self.curr_weight_diff.to(self.args.device)

        curr_weight = torch.cat(self.global_weights_diff, dim=1).to(self.args.device)
        curr_grad = torch.cat(self.global_updates_diff, dim=1).to(self.args.device)
        weight_grad = torch.matmul(curr_weight.T, curr_grad)
        weight_weight = torch.matmul(curr_weight.T, curr_weight)
        R_k = np.triu(weight_grad.cpu().detach())
        L_k = weight_grad - torch.tensor(R_k, device=self.args.device)
        D_k_diag = torch.diag(weight_grad)
        sigma_k = torch.matmul(self.global_updates_diff[-1].T, self.global_weights_diff[-1]) \
                  / (torch.matmul(self.global_weights_diff[-1].T, self.global_weights_diff[-1]))
        sigma_k = sigma_k.to(self.args.device)
        upper_mat = torch.cat([sigma_k * weight_weight, L_k], dim=1)
        lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1)
        mat = torch.cat([upper_mat, lower_mat], dim=0)
        mat_inv = torch.linalg.inv(mat)

        approx_prod = sigma_k * self.curr_weight_diff
        p_mat = torch.cat([torch.matmul(curr_weight.T, sigma_k * self.curr_weight_diff), torch.matmul(curr_grad.T, self.curr_weight_diff)], dim=0)
        approx_prod -= torch.matmul(torch.matmul(torch.cat([sigma_k * curr_weight, curr_grad], dim=1), mat_inv), p_mat)

        return approx_prod
    
    # Gap statistics
    def gap_stat(self, byz_scores):
        def gap_func(ref_list, byz_sum):
            byz_sum = EPS if byz_sum == 0 else byz_sum
            return np.log(np.mean(ref_list)) - np.log(byz_sum)
            #return np.mean([np.log(ref) - np.log(byz_sum) for ref in ref_list])

        gap_list = []
        sk_list = []

        byz_scores = byz_scores.cpu().detach().numpy().reshape(-1, 1)
        # Normalize suspicious scores within 0 ~ 1
        max_score, min_score = np.max(byz_scores), np.min(byz_scores)
        byz_scores = (byz_scores - min_score) / max_score

        for num_cluster in range(1, self.max_cluster+1):
            # k-means cluster based on num_cluster in loop (1 ~ K)
            kmeans_k = KMeans(n_clusters=num_cluster).fit(byz_scores)
            centroids_k = kmeans_k.cluster_centers_
            labels_k = kmeans_k.labels_
            
            dist_byz_sum = np.sum(np.power(byz_scores - centroids_k[labels_k], 2))

            ref_list = []
            for reference in range(self.num_sample):
                ref_scores = np.random.rand(self.args.n_user, 1)
                kmeans_ref = KMeans(n_clusters=num_cluster).fit(ref_scores)

                centroids_ref = kmeans_ref.cluster_centers_
                labels_ref = kmeans_ref.labels_
                dist_ref_sum = np.sum(np.power(ref_scores - centroids_ref[labels_ref], 2))
                ref_list.append(dist_ref_sum)

            gap_list.append(gap_func(ref_list, dist_byz_sum))
            sk_list.append(np.sqrt((1 + self.num_sample) / self.num_sample) 
                            * np.std([np.log(EPS) if ref == 0 else np.log(ref) for ref in ref_list]))

        stat_list = [gap_list[k] - gap_list[k+1] + sk_list[k] for k in range(self.max_cluster - 1)]
        stat_list = [float('inf') if stat_list[k] < 0 else stat_list[k] for k in range(self.max_cluster - 1)]

        return np.argmin(stat_list) + 1

    """
    FLDetector
    : Calculate suspicious scores based on predicted vs actual gradients
      and k-means cluster to distinguish benign vs byzantine worker clusters
    
    Outputs
    - List of malicious(byzantine) client indices or None
    """

    def detect_fl(self):
        adversary_list = []
        if self.round == 0:
            print(f"Adversary Not Found")
            return adversary_list
        
        client_update_losses = []
        prev_adv_list = []
        for iter in range(1, self.round+1):
            self.pred_Ht_v = self.lbfgs(iter)
            self.pred_Ht_v = torch.where(torch.isnan(self.pred_Ht_v),
                                         torch.full_like(self.pred_Ht_v, EPS),
                                         self.pred_Ht_v)
            #print(f"Predicted Hessian: {self.pred_Ht_v}")

            # Predict global model "update" (gradient) per client for this iteration
            client_update_loss = []
            for client in range(self.args.n_user):
                g_hat = self.local_updates[max(iter-1, 0)][:, client].unsqueeze(1).to(self.args.device) - self.pred_Ht_v
                client_update_loss.append(torch.norm(g_hat - self.local_updates[iter][:, client].unsqueeze(1).to(self.args.device), p=2).unsqueeze(0))
            client_update_loss = torch.cat(client_update_loss, dim=0)

            l1norm = torch.norm(client_update_loss, p=1)
            client_update_loss /= l1norm
            client_update_losses.append(client_update_loss)
            print(f"d_t: {len(client_update_losses)}")
            byz_scores = torch.mean(torch.stack(client_update_losses[max(iter-self.window_size+1, 0):iter+1]), dim=0)
            print(byz_scores)

            # Get optimal number of clusters via gap statistics
            print(f"Iteration {iter}")
            optnum_cluster = self.gap_stat(byz_scores)
            print(f"Optimal Cluster Number: {optnum_cluster}")
            # If optimal number is greater than 1 (distinct clustering possible)
            if optnum_cluster > 1:
                byz_scores = byz_scores.cpu().detach().numpy().reshape(-1, 1)
                kmeans_detect = KMeans(n_clusters=self.num_cluster_fl).fit(byz_scores)

                labels_detect = kmeans_detect.labels_
                avg_scores = [byz_scores[labels_detect == i].mean() for i in range(self.num_cluster_fl)]
                
                # All clients in one cluster (KMeans Clustering Failed)
                if float('nan') in avg_scores:
                    continue
                else:
                    malicious_cluster = avg_scores.index(max(avg_scores))
                    adversary_list = list(np.where(labels_detect == malicious_cluster)[0])
                    if len(adversary_list) >= (self.args.n_user // 2):    
                        adversary_list = prev_adv_list
                    else:
                        prev_adv_list = adversary_list
                    print(f"Adversary IDs: {adversary_list}")
                    # return adversary_list
                    
        print(f"Final Adversary List: {adversary_list}")
        return adversary_list