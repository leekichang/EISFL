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

class LabelFlip(Clients.NaiveClient):
    def __init__(self, args, name):
        super(LabelFlip, self).__init__(args, name)
        self.flip = self.args.flip
        self.tag  = 'LabelFlip Attacker'

    def flip_label(self, targets):
        return (targets+self.flip)%cfg.N_CLASS[self.args.dataset]

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            target = self.flip_label(target) # Attack
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.defense(self.model)
            self.optimizer.step()
        self.model = self.model.to('cpu')