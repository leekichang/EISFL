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

class MSE(ImageMetric):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def measure(self, pred_img, target_img):
        pred, target = self._sendto_device(pred_img), self._sendto_device(target_img)
        result = torch.mean((pred - target) ** 2)

        return result
    
    def __call__(self, pred_img, target_img):
        return self.measure(pred_img, target_img)

class PSNR(ImageMetric):
    def __init__(self, device="cpu"):
        super().__init__(device)

    def measure(self, pred_img, target_img, binarize=False):
        assert pred_img.shape == target_img.shape
        pred, target = self._sendto_device(pred_img), self._sendto_device(target_img)
        if binarize:
            pred, target = self._binarize(pred), self._binarize(target)
        if len(pred_img.shape)==4:
            dim=0
        elif len(pred_img.shape)==3:
            dim=0
        B = pred.shape[0]
        mse = torch.mean(((pred - target)**2).reshape(B,-1), dim=-1)
        result = 10 * torch.log10(1 / mse)

        return result
    
    def __call__(self, pred_img, target_img):
        return self.measure(pred_img, target_img)

class SSIM(ImageMetric):
    """SSIM: Structural Similarity Index Measure"""
    def __init__(self, device="cpu", val_range=1.0):
        super().__init__(device)
        self.val_range = val_range
        self.C1 = (0.01 * self.val_range) ** 2
        self.C2 = (0.03 * self.val_range) ** 2
    
    def set_val_range(self, range):
        return setattr(self, "val_range", range)

    def gaussian(self, window_size, sigma):
        """
        Return gaussian function result within the range [0, window_size]
        args:
            window_size: 
            sigma: Standard Deviation
        """
        # Calculate gaussian
        gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 \
                                        / float(2 * sigma ** 2)) for x in range(window_size)])
        # Normalize in discrete manner (sum of gauss items)
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel=1):
        # Create 1-Dimensional window -> torch.Size([1, window_size])
        _1d_window = self.gaussian(window_size, sigma=1.5).unsqueeze(1)
        # Expand 1D window to 2-Dimensional space -> torch.Size([1, 1, 1, window_size])
        _2d_window = torch.matmul(_1d_window, _1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        # Create window -> torch.Size([channel, 1, window_size, window_size])
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def measure(self, pred_img, target_img, window_size, size_avg=True, with_contrast=False):
        pred, target = self._sendto_device(pred_img), self._sendto_device(target_img)
        _, channels, height, width = pred_img.size()

        # Create Window
        real_size = min(window_size, height, width)
        window = self.create_window(real_size, channel=channels).to(self.device)

        # Params
        padding = window_size // 2
        mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
        mu_target = F.conv2d(target, window, padding=padding, groups=channels)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(target * pred, window, padding=padding, groups=channels) - mu_cross

        v1 = 2.0 * sigma_cross + self.C2
        v2 = sigma_pred_sq + sigma_target_sq + self.C2

        contrast = torch.mean(v1 / v2)
        ssim = ((2 * mu_cross + self.C1) * v1) / ((mu_pred_sq + mu_target_sq + self.C1) * v2)

        if size_avg:
            result = ssim.mean()
        else:
            result = ssim.mean(1).mean(1).mean(1)

        if with_contrast:
            return result, contrast
        else:
            return result

    def __call__(self, pred_img, target_img, window_size=8, size_avg=True, with_contrast=False):
        return self.measure(pred_img, target_img, window_size, size_avg, with_contrast)

class TotalVariation(ImageMetric):
    """Total Variation Metric (Anisotropic)"""
    def __init__(self, device="cpu"):
        super().__init__(device)

    def measure(self, pred_img, alpha=1, option="mean"):
        # Anisotropic total variation of predicted image
        pred = self._sendto_device(pred_img)
        pred = self._set_dimension(pred)

        # Calculate Diff in horizontal/vertical direction
        img_horizontal = torch.abs(pred[:,:,:,:-1] - pred[:,:,:,1:])
        img_vertical = torch.abs(pred[:,:,:-1,:] - pred[:,:,1:,:])
        
        if option == "mean":
            return alpha * (torch.mean(img_horizontal) + torch.mean(img_vertical))
        if option == "sum":
            return alpha * (torch.sum(img_horizontal) + torch.sum(img_vertical))

    def __call__(self, pred, alpha=1, option="mean"):
        return self.measure(pred, alpha, option)