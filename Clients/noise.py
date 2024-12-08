import os
import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import copy
import torch
import torch.nn as nn


import utils
import Clients
import config as cfg
from DataManager import datamanager

class NoiseAdd(Clients.NaiveClient):
    def __init__(self, args, name):
        super(NoiseAdd, self).__init__(args, name)
        self.tag  = 'Gaussian Noise Attacker'
    
    def add_noise(self, eps=0.1):
        dict_params = self.model.state_dict()
        for key in dict_params.keys():
            dict_params[key] += torch.randn_like(dict_params[key])*eps
        self.model.load_state_dict(dict_params)

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.defense(self.model)
            self.optimizer.step()
        self.model = self.model.to('cpu')
        self.add_noise() # Attack