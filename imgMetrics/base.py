import torch
import torchvision.models as M
import torch.nn as nn
import torch.nn.functional as F

import math

"""[REFERENCE]
- https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
- https://torchmetrics.readthedocs.io/en/stable/image/
"""

__all__ = ["ImageMetric"]

class ImageMetric(object):
    """Baseline for Metric Classes"""  
    def __init__(self, device="cpu"):
        self.eps = 1e-06
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"

    def _set_device(self, device):
        # Set device of the Metric class
        if torch.cuda.is_available():
            return setattr(self, "device", device)
        else:
            print("[*] CUDA is not available. Set to CPU.")
            return setattr(self, "device", "cpu")
    
    def _set_dimension(self, img):
        # Set Dimension of image tensor to (B, C, H, W)
        if len(img.size()) < 4:
            while len(img.size()) == 4:
                img = img.unsqueeze(0)
        return img        
        
    def _normalize(self, img):
        # Set pixel range within [0, 1]
        if torch.max(img) > 1.0:
            img = img / 255.0
        return img
    
    def _denormalize(self, img):
        # Set pixel range within [0, 1]
        if torch.max(img) <= 1.0:
            img = img * 255.0
        return img
    
    def _sendto_device(self, img, device=None, normalize=True):
        if device != None:
            return self._normalize(img.clone().to(device)) if normalize == True else img.clone().to(device)
        elif torch.cuda.is_available():
            return self._normalize(img.clone().to(self.device)) if normalize == True else img.clone().to(self.device)
        else:
            print("[*] CUDA is not available. Set to CPU.")
            return self._normalize(img.clone().to("cpu")) if normalize == True else img.clone().to("cpu")
        
    def _binarize(self, img, threshold=0.5, is_boolean=False):
        # Binarize each pixel based on given threshold
        if is_boolean is True:
            return img >= threshold
        else:
            img[img < threshold] = 0.0
            img[img >= threshold] = 1.0
            return img

    def _classify(self, pred_img, target_img, threshold=0.5, option="all"):
        # 
        pred, target = self._sendto_device(pred_img), self._sendto_device(target_img)
        pred, target = self._set_dimension(pred), self._set_dimension(target)
        pred, target = self._binarize(pred_img, threshold), self._binarize(target_img, threshold)
        tp = torch.sum(target * pred).float()
        tn = torch.sum((1 - target) * (1 - pred)).float()
        fp = torch.sum((1 - target) * pred).float()
        fn = torch.sum(target * (1 - pred)).float()

        if option == "tp":
            return tp
        if option == "tn":
            return tn
        if option == "fp":
            return fp
        if option == "fn":
            return fn
        if option == "all":
            return {"tp" : tp,
                    "tn" : tn,
                    "fp" : fp,
                    "fn" : fn}
    
    def _prettify(self, tensor):
        assert len(tensor.size()) == 0
        return "{%.4f}".format(tensor.item())

    def measure(self, pred_img, target_img):
        pass

    def __call__(self, pred_img, target_img):
        return self.measure(pred_img, target_img)