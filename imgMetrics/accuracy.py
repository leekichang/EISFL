import torch
import torchvision.models as M
import torch.nn as nn
import torch.nn.functional as F

import math

"""[REFERENCE]
- https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
- https://torchmetrics.readthedocs.io/en/stable/image/
"""

from .base import *

class Precision(ImageMetric):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def measure(self, pred, target, threshold=0.5):
        result = self._classify(pred, target, threshold, "all")
        assert type(result) == dict

        return result['tp'] / (result['tp'] + result['fp'] + self.eps)
    
    def __call__(self, pred, target, threshold=0.5):
        return self.measure(pred, target, threshold)
    
class Recall(ImageMetric):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def measure(self, pred, target, threshold=0.5):
        result = self._classify(pred, target, threshold, "all")
        assert type(result) == dict
        
        return result['tp'] / (result['tp'] + result['tn'] + self.eps)
    
    def __call__(self, pred, target, threshold=0.5):
        return self.measure(pred, target, threshold)
        
class F1Score(ImageMetric):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def measure(self, pred, target, threshold=0.5):
        # Measure F1 Score
        result = self._classify(pred, target, threshold)
        assert type(result) == dict

        precision = result["tp"] / (result["tp"] + result["fp"] + self.eps)
        recall = result["tp"] / (result["tp"] + result["tn"] + self.eps)
        return 2 * (recall * precision) / (recall + precision)
    
    def __call__(self, pred, target, threshold=0.5):
        return self.measure(pred, target, threshold)