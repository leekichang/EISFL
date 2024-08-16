import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import ops
import utils
from client import *

__all__ = ['GradPruneClient']

class GradPruneClient(Client):
    def __init__(self,
                 args,
                 client_id,
                 trainset,
                 testset):
        super(GradPruneClient, self).__init__(args, client_id, trainset, testset)
        self.p   = args.p
        
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
                parameters = self.model.parameters()
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                    parameters = list(
                        filter(lambda p: p.grad is not None, parameters))
                input_grads = [p.grad.data for p in parameters]
                threshold = [
                    torch.quantile(torch.abs(input_grads[i]), self.p)
                    for i in range(len(input_grads))
                ]
                for i, p in enumerate(self.model.parameters()):
                    p.grad[torch.abs(p.grad) < threshold[i]] = 0
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
        