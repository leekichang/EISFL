import torch
import torchvision.models as M
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import math

"""[REFERENCE]
- https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
- https://torchmetrics.readthedocs.io/en/stable/image/
"""

from .base import *

class LPIPS(ImageMetric):
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.vgg = M.vgg19()
        self.alex = M.alexnet()
        self.squeeze = M.squeezenet1_0()

    def measure(self, pred_img, target_img, net_type="vgg", reduction="mean", normalize=True):
        pred_img, target_img = self._sendto_device(pred_img), self._sendto_device(target_img)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, reduction=reduction, normalize=normalize)
        return lpips(pred_img, target_img)

    def __call__(self, pred_img, target_img, net_type="vgg", reduction="mean", normalize=True):
        return self.measure(pred_img, target_img, net_type=net_type, reduction=reduction, normalize=normalize)