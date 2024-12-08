import copy
import torch
import utils

import ops
import config as cfg

class BaseClient(object):
    def __init__(self, args, name):
        self.args      = args
        self.exp_name  = args.exp_name
        self.name      = name
        self.model     = None # Assigned in Server Class
        self.trainset  = None # Assigned in Server Class
        self.testset   = None # Assigned in Server Class
        self.save_path = f'./checkpoints/{self.exp_name}'
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.imsize     = cfg.IMGSIZE[self.args.dataset][-1]
        self.patch_size = cfg.PATCHSIZE[self.args.dataset]
        self.patch_shuffle   = None
        self.pixel_shuffle   = None
        self.weight_shuffler = None
        utils.ensure_path(self.save_path)
    
    def set_shuffle_rule(self, seed):
        torch.manual_seed(seed)
        self.patch_shuffle = torch.randperm((self.imsize//self.patch_size)**2) if self.args.shuffle else None
        self.pixel_shuffle = torch.randperm(self.patch_size**2) if self.args.shuffle else None
        self.weight_shuffler = ops.WeightShuffle(self.model.state_dict(),
                                                  self.patch_size,
                                                  patch_shuffle=self.patch_shuffle,
                                                  pixel_shuffle=self.pixel_shuffle)
        self.data_shuffler = ops.ImageShuffle(self.patch_size,
                                              patch_shuffle=self.patch_shuffle,
                                              pixel_shuffle=self.pixel_shuffle)

    def shuffle_weight(self, round):
        pass

    def unshuffle_weight(self, round):
        pass

    def get_name(self):
        return self.name

    def get_type(self):
        return 'Base'
    
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save_model(self, tag):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.name}_{tag}.pt')