import os
import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import torch
import numpy as np
import torch.nn as nn

import utils
import Defenses
from DataManager import datamanager

class GradPrune(Defenses.NoDefense):
    def __init__(self, args):
        super(GradPrune, self).__init__(args)
        assert args.prune_ratio > 0.0 and args.prune_ratio < 1
        self.prune_ratio = args.prune_ratio      # ratio for gradient pruning

    def prune_gradients(self, model):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
            parameters = list(
                filter(lambda p: p.grad is not None, parameters))
        input_grads = [p.grad.data for p in parameters]
        threshold = [torch.quantile(torch.abs(input_grads[i]), self.prune_ratio) for i in range(len(input_grads))]
        for i, p in enumerate(model.parameters()):
            p.grad[torch.abs(p.grad) < threshold[i]] = 0

    def __call__(self, model, *args):
        self.prune_gradients(model)

if __name__ == '__main__':
    pass