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
from client import *
from server import *
from DPSGD import *
from GradPrune import *
from adversary import *

import torch.utils.tensorboard as tb

from sklearn.cluster import KMeans

"""
Byzantine-robust Aggregation Rules
"""
class Aggregator(object):
    def __init__(self,
                 method="fedavg",
                 global_param=None,
                 client_params=None,
                 beta=0.1,
                 ):
        self.method          = method
        self.global_param    = global_param
        self.client_params   = client_params
        self.beta            = beta
    
    def update(self,
               method=None,
               global_param=None,
               client_params=None,
               beta=None,
               ):
        self.global_param = global_param if global_param is not None else self.global_param
        self.sampled_clients = client_params if client_params is not None else self.client_params
        self.method = global_param if method is not None else self.method
        self.beta = beta if beta is not None else self.beta

    def __call__(self,
                 method=None,
                 global_param=None,
                 client_params=None,
                 beta=None,
                 ):
        self.update(global_param, client_params, method, beta)
        assert self.global_param is not None and self.client_params is not None and self.method is not None

        param_shapes = [param.shape for param in self.global_param.values()]
        param_keys = list(self.global_param.keys())

        client_models = []
        for client in client_params:
            copied_weights = []
            for _, param in client:
                copied_weights.append(copy.deepcopy(param))
            reshaped_weight = torch.cat([param.reshape(-1, 1) for param in copied_weights], dim=0)
            client_models.append(reshaped_weight)
        cat_client_param = torch.cat(client_models, dim=1)

        if method == "fedavg":
            aggregated_params = torch.mean(cat_client_param, dim=-1)
        elif method == "trim":
            sorted_client_param, _ = torch.sort(cat_client_param)
            aggregated_params = torch.mean(sorted_client_param[:, int(beta*len(self.client_params)):-int(beta*len(self.client_params))], dim=1).reshape(-1, 1)
        elif method == "median":
            aggregated_params = self.median(cat_client_param)
            aggregated_params = aggregated_params.reshape(-1, 1)
        elif method == "krum":
            raise NotImplementedError
            #TODO?: Update this
            #aggregated_params = 

        averaged_weights = OrderedDict()
        start = 0
        for shape, key in zip(param_shapes, param_keys):
            length = torch.prod(torch.tensor(shape)).item()
            averaged_weights[key] = aggregated_params[start:start+length].view(shape)
            start += length

        return averaged_weights