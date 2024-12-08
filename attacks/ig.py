"""
Created on Thu Aug 28 2023
@author: Kichang Lee
@contact: kichang.lee@yonsei.ac.kr
references
1. code: https://github.com/mit-han-lab/dlg
2. paper: https://arxiv.org/abs/1906.08935
"""
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

from .base import *
import config as cfg

__all__ = ['IG']

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def total_variation(img):
        if len(img.shape) == 3:
            img = img[None,:]        
        img_horizontal = torch.abs(img[:,:,:,:-1] - img[:,:,:,1:])
        img_vertical = torch.abs(img[:,:,:-1,:] - img[:,:,1:,:])
        return torch.mean(img_horizontal) + torch.mean(img_vertical)

class IG(Attacker):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = None
        self.iteration = 100
        self.criterion = cross_entropy_for_onehot
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.alpha     = 0.1 # CIFAR10, MNIST = 0.1
        self.best_idx  = 0
        self.best_loss = 1000000
    
    def attack(self, old_model, new_model, **kwargs):
        true_label = kwargs['gnd_labels'] if 'gnd_labels' in kwargs else None
        print(f"TRUE LABEL: {true_label}")
        old_model.to(self.device)
        new_model.to(self.device)
        # print("FUCK NEW", new_model.state_dict()['mixer_blocks.0.token_mixer.2.net.0.weight'][0,:10])
        # print("FUCK OLD", old_model.state_dict()['mixer_blocks.0.token_mixer.2.net.0.weight'][0, :10])
        dy_dx = []
        for idx, param in enumerate(new_model.parameters()):
            dy_dx.append(param)
        for idx, param in enumerate(old_model.parameters()):
            dy_dx[idx] = (param - dy_dx[idx])
        # if self.args.optimizer == 'SGD':                # the weight difference it self is not identical to the gradient!!
        dy_dx = [_/self.args.lr for _ in dy_dx]
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        
        # generate dummy data and label
        dummy_data  = torch.clamp(torch.randn(cfg.IMGSIZE[self.args.dataset]), 0, 1).to(self.device).requires_grad_(True)
        if true_label is None:
            dummy_label = torch.randn((1, cfg.N_CLASS[self.args.dataset])).to(self.device).requires_grad_(True)
            optimizer   = optim.LBFGS([dummy_data, dummy_label])
        else:
            true_label = true_label.detach()
            target = torch.unsqueeze(true_label, 1)
            onehot_target = torch.zeros(target.size(0), cfg.N_CLASS[self.args.dataset], device=target.device)
            onehot_target.scatter_(1, target, 1)
            dummy_label = onehot_target.to(self.device)
            # dummy_label = onehot_target.to(self.device).requires_grad_(True)
            # dummy_label = onehot_target.to(self.device)
            # dummy_data = kwargs['gnd_data']*0.25 + 0.75*torch.randn(cfg.IMGSIZE[self.args.dataset]).to(self.device)
            dummy_data = torch.randn_like(kwargs['gnd_data']).to(self.device)
            dummy_data = dummy_data.to(self.device).requires_grad_(True)
            optimizer   = optim.LBFGS([dummy_data])
            # optimizer   = optim.Adam([dummy_data], lr=0.075)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[self.iteration // 2.667, self.iteration // 1.6, self.iteration // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            # print(f"Dummy Label: {dummy_label}")
        
        # optimizer   = optim.LBFGS([dummy_data, dummy_label])
        # optimizer   = optim.LBFGS([dummy_data])
        converge = False
        track    = 0
        for iters in range(self.iteration):
            def closure():
                optimizer.zero_grad()

                dummy_pred = old_model(dummy_data)
                # dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_onehot_label = F.softmax(dummy_label)
                # print(f'Dummy Pred: {F.softmax(dummy_pred)}')
                # print(f'Dummy Label: {dummy_onehot_label}')
                
                dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, old_model.parameters(), create_graph=True)    

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += 1 - F.cosine_similarity(gx.flatten(), gy.flatten(), 0, 1e-8)
                grad_diff /= len(dummy_dy_dx)+1e-8
                grad_diff += self.alpha * total_variation(dummy_data)

                batch_diff = 0
                if self.args.model == 'vit____':
                    old_norm1_weight = old_model.state_dict()['transformer.layers.0.norm1.weight']
                    old_norm1_bias = old_model.state_dict()['transformer.layers.0.norm1.bias']
                    old_norm2_weight = old_model.state_dict()['transformer.layers.0.norm2.weight']
                    old_norm2_bias = old_model.state_dict()['transformer.layers.0.norm2.bias']
                    new_norm1_weight = new_model.state_dict()['transformer.layers.0.norm1.weight']
                    new_norm1_bias = new_model.state_dict()['transformer.layers.0.norm1.bias']
                    new_norm2_weight = new_model.state_dict()['transformer.layers.0.norm2.weight']
                    new_norm2_bias = new_model.state_dict()['transformer.layers.0.norm2.bias']
                    batch_diff = F.mse_loss(old_norm1_weight, new_norm1_weight) + F.mse_loss(old_norm1_bias, new_norm1_bias) + F.mse_loss(old_norm2_weight, new_norm2_weight) + F.mse_loss(old_norm2_bias, new_norm2_bias)

                grad_diff += 0.1 * batch_diff
                grad_diff.backward()
                return grad_diff
            
            optimizer.step(closure)
            # scheduler.step()
            if self.args.dataset != 'MITBIH':
                self.history.append(transforms.ToPILImage()(dummy_data[0].cpu()))
            else:
                self.history.append(dummy_data.cpu().detach().numpy().reshape(-1))

            current_loss = closure().cpu().item()
            loss = F.mse_loss(dummy_data, kwargs['gnd_data'])
            self.loss_trace.append(loss)
            gap = self.loss_trace[-1]-current_loss if len(self.loss_trace) > 0 else 1
            if abs(gap)<0.00001:
                track += 1
            else:
                track = 0
            converge = True if track > 200 else False
            converge = True if loss < 0.001 else converge
            converge = True if iters > self.iteration else converge
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_idx = iters
            if iters % 20 == 0:
                print(iters, "%.4f" % loss)
                if self.args.dataset != 'MITBIH':
                    self.history[-1].save(f'{self.name}/last.png')
                else:
                    plt.plot(dummy_data.cpu().detach().numpy().reshape(-1))
                    plt.savefig(f'{self.name}/last.png')
                    plt.close()
            if torch.isnan(torch.tensor(current_loss)):
                print("NaN Value Occured in Iteration")
                break
            # dummy_data = torch.clamp(dummy_data, 0, 1)
            if converge:
                print("Converged")
                break
    
 


    