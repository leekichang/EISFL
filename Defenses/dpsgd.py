import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import torch
import numpy as np

import utils
import Defenses


class DPSGD(Defenses.NoDefense):
    def __init__(self, args):
        super(DPSGD, self).__init__(args)
        self.epsilon = args.epsilon      # Epsilon for noise scale
        self.delta = args.delta          # Delta for differential privacy
        self.clip_norm = args.clip_norm  # Norm for gradient clipping
        self.sensitivities = {'MNIST': 0.01, 'CIFAR10': 0.005}
        self.sensitivity = self.sensitivities[args.dataset]
    def sanitize(self, tensor):
        eps = self.epsilon
        delta = self.delta
        sensitivity = self.sensitivity # MNIST 0.01, CIFAR10 0.005
        clip_norm = self.clip_norm
        sigma = sensitivity*np.sqrt(2*np.log(1.25 / delta)) / eps
        
        tensor_norm = torch.norm(tensor, p=2)
        if tensor_norm > clip_norm:
            tensor = tensor * (clip_norm / tensor_norm)
        noise = torch.normal(mean=0, std=sigma, size=tensor.shape).to(tensor.device)
        tensor = tensor + noise
        return tensor
    
    def sanitize_gradients(self, model):
        for p in model.parameters():
            if p.grad is not None:
                sanitized_grad = self.sanitize(p.grad.data)
                p.grad.data = sanitized_grad

    def __call__(self, model, *args):
        self.sanitize_gradients(model)

if __name__ == '__main__':
    pass