import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import ops
import utils
from client import *

__all__ = ['DPSGDClient']

class DPSGDClient(Client):
    def __init__(self,
                 args,
                 client_id,
                 trainset,
                 testset):
        super(DPSGDClient, self).__init__(args, client_id, trainset, testset)
        self.eps   = args.eps
        self.delta = 1e-2
        self.C     = 4
        self.std   = np.sqrt(2*np.log(1.25/self.delta))/self.eps
        
    def local_train(self):
        self.model.train()
        self.model.to(self.device)
        for epoch in range(self.epochs):
            preds, targets, losses = [], [], []
            for X, Y in self.train_loader:
                self.optimizer.zero_grad()
                X, Y = X.to(self.device), Y.to(self.device)
                X    = self.shuffler(X)
                pred = self.model(X)
                loss = self.criterion(pred, Y)
                preds.append(pred.detach().cpu().numpy())
                targets.append(Y.detach().cpu().numpy())
                losses.append(loss.item())
                loss.backward()
                ##### DPSGD #####
                clip_grad_norm_(self.model.parameters(), self.C)
                for i, p in enumerate(self.model.parameters()):
                    p.grad += (torch.normal(mean=0, std=(self.C)*(self.std), size=p.grad.shape)/len(self.train_loader)).to(self.device)
                ##### DPSGD #####
                self.optimizer.step()
        
        loss = np.mean(losses)
        preds    = np.concatenate(preds)
        targets  = np.concatenate(targets)
        
        top1_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        top5_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=5)
        print(f'=== Client {self.id:>2} Finished Training {len(self.trainset)} samples===')
        print(f'client:{self.id:>2} | Top-1 Train Acc:{top1_acc:.2f} | Top-5 Train Acc:{top5_acc:.2f} | Train Loss:{loss:.4f}')
        self.model.to('cpu')
        if "cuda" in self.device : torch.cuda.empty_cache()
        