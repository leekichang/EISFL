import copy
import torch
import numpy as np
from torch.utils.data import DataLoader

import ops
import utils
import config as cfg

__all__ = ['Client']

class Client(object):
    def __init__(self,
                 args,
                 client_id,
                 trainset,
                 testset):
        self.args            = args
        self.id              = client_id
        self.trainset        = utils.load_user_dataset(trainset)
        self.testset         = utils.load_user_dataset(testset)
        self.device          = args.device
        self.model           = utils.build_model(args)
        self.selected_rounds = []

        self.patch_size      = cfg.PATCHSIZE[args.dataset]
        self.shuffler        = ops.ImageShuffle(patch_size=(self.patch_size,self.patch_size),
                                               patch_shuffle=None,
                                               pixel_shuffle=None)
        self.weight_shuffler = ops.WeightShuffle(self.model.state_dict(), self.patch_size, None, None)
        self.top_k         = 1 if args.dataset == 'Celeba' else 3
    def __len__(self):
        return len(self.trainset)
    
    def setup(self):
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.test_loader  = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        self.optimizer    = utils.build_optimizer(self.model, self.args)
        self.criterion    = utils.build_criterion(self.args)
        self.epochs       = self.args.local_epoch
    
    def set_optim(self, lr):
        self.args.lr      = lr
        self.optimizer    = utils.build_optimizer(self.model, self.args)
    
    def set_shuffler(self, patch, pixel):
        self.shuffler.patch_shuffle = patch
        self.shuffler.pixel_shuffle = pixel
    
    def set_weight_shuffler(self, weight, patch, pixel):
        self.weight_shuffler.new_weight    = copy.deepcopy(weight)
        self.weight_shuffler.patch_shuffle = patch
        self.weight_shuffler.pixel_shuffle = pixel
    
    def shuffle(self, patch, pixel):
        self.set_shuffler(patch, pixel)
        weights = copy.deepcopy(self.model.state_dict())
        self.set_weight_shuffler(weights, patch, pixel)
        shuffled_weight = copy.deepcopy(self.weight_shuffler.shuffle(self.args.model))
        self.model.load_state_dict(copy.deepcopy(shuffled_weight))
    
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
                self.optimizer.step()
        
        loss = np.mean(losses)
        preds    = np.concatenate(preds)
        targets  = np.concatenate(targets)
        
        top1_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        topk_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=self.top_k)
        print(f'=== Client {self.id:>2} Finished Training {len(self.trainset)} samples===')
        print(f'client:{self.id:>2} | Top-1 Train Acc:{top1_acc:.2f} | Top-{self.top_k} Train Acc:{topk_acc:.2f} | Train Loss:{loss:.4f}')
        self.model.to('cpu')
        if "cuda" in self.device : torch.cuda.empty_cache()
        
    @torch.no_grad()
    def local_test(self):
        self.model.eval()
        self.model.to(self.device)
        preds, targets, losses = [], [], []
        
        for X, Y in self.test_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            X    = self.shuffler(X)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            
            preds.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
        
        preds    = np.concatenate(preds)
        targets  = np.concatenate(targets)
        top1_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        topk_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=self.top_k)
        print(f'client:{self.id:>2} | Top-1 Test Acc:{top1_acc:.2f} | Top-{self.top_k} Test Acc:{topk_acc:.2f} | Test Loss:{loss:.4f}')
        self.model.to('cpu')
        if "cuda" in self.device : torch.cuda.empty_cache()