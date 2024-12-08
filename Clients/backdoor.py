import os
import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import copy
import torch
import torch.nn as nn


import utils
import Clients
import config as cfg
from DataManager import datamanager, backdoor_generator

class Backdoor(Clients.NaiveClient):
    def __init__(self, args, name):
        super(Backdoor, self).__init__(args, name)
        self.tag  = 'Backdoor Attacker'
        self.backdoor = backdoor_generator.BackdoorGenerator(args)
    
    def setup(self):
        self.model       = utils.build_model(self.args)
        self.criterion   = nn.CrossEntropyLoss()
        self.trainset.data, self.trainset.targets = self.backdoor(self.trainset,    # Attack
                                                                  option='left',
                                                                  proportion=0.5,
                                                                  target=0)
        
        self.optimizer   = torch.optim.Adam(self.model.parameters(),
                                            lr=self.args.lr,
                                            weight_decay=self.args.lr*0.1)
        
        self.trainloader = torch.utils.data.DataLoader(self.trainset,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True )
        
        self.testloader  = torch.utils.data.DataLoader(self.testset,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=False,
                                                       drop_last=False)
        self.n_samples   = len(self.trainset)
        self.epoch       = self.args.epoch

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            # self.defense(self.model)
            self.optimizer.step()
        self.model = self.model.to('cpu')