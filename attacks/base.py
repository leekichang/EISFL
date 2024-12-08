"""
Created on Thu Aug 28 2023
@author: Kichang Lee
@contact: kichang.lee@yonsei.ac.kr
"""

import os
import torch
import matplotlib.pyplot as plt

__all__ = [
    'Attacker'
]

class Attacker(object):
    def __init__(self, args):
        self.args       = args
        self.name       = args.attack_name
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_trace = []
        self.history    = []
        
    def attack(self, old_model, new_model):
        pass
    
    def vis(self, img_idx=0):
        os.makedirs(self.name, exist_ok=True)
        plt.imshow(self.history[-1])
        plt.axis('off')
        plt.savefig(f'./{self.name}/{img_idx}_result.png')
        # plt.figure(figsize=(12, 8))
        # plt.title(f"{self.name} attack result")
        # plt.axis('off')
        # for i in range(30):
        #     plt.subplot(3, 10, i + 1)
        #     plt.imshow(self.history[i])
        #     plt.title("iter=%d" % (i * 10))
        #     plt.axis('off')
        # plt.savefig(f'./{self.name}_result.png')
    