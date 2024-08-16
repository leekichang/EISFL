"""
Created on Thu Aug 28 2023
@author: Kichang Lee
@contact: kichang.lee@yonsei.ac.kr
references
1. code: https://github.com/mit-han-lab/dlg
2. paper: https://arxiv.org/abs/1906.08935
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from .base import *
import config as cfg

__all__ = ['DLG']

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

class DLG(Attacker):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = None
        self.iteration = 300
        self.criterion = cross_entropy_for_onehot
        pass
    
    def attack(self, old_model, new_model, **kwargs):
        old_model.to(self.device)
        new_model.to(self.device)
        
        dy_dx = []
        for idx, param in enumerate(new_model.parameters()):
            dy_dx.append(param)
        for idx, param in enumerate(old_model.parameters()):
            dy_dx[idx] = (param - dy_dx[idx])
        if self.args.optimizer == 'SGD':                # the weight difference it self is not identical to the gradient!!
            dy_dx = [_/self.args.lr for _ in dy_dx]
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        
        # generate dummy data and label
        dummy_data = torch.randn(cfg.IMGSIZE[self.args.dataset]).to(self.device).requires_grad_(True)
        dummy_label = torch.randn((self.args.n_user, cfg.N_CLASS[self.args.dataset])).to(self.device).requires_grad_(True)
        optimizer = optim.LBFGS([dummy_data, dummy_label])
        
        for iters in range(self.iteration):
            def closure():
                optimizer.zero_grad()
                
                dummy_pred = old_model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, old_model.parameters(), create_graph=True)\
                    
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff
            optimizer.step(closure)
            if iters % 10 == 0: 
                current_loss = closure()
                self.loss_trace.append(current_loss.cpu().item())
                print(iters, "%.4f" % current_loss.item())
                self.history.append(transforms.ToPILImage()(dummy_data[0].cpu()))
                if torch.isnan(torch.tensor(current_loss.item())):
                    print("NaN Value Occured in Iteration")
                    break
    
 


    