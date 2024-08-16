import copy
import torch

import ops

__all__ = [
    'WeightShuffle'
]

class WeightShuffle(object):
    def __init__(self, weight, patch_size, patch_shuffle, pixel_shuffle):
        self.new_weight    = copy.deepcopy(weight)
        self.patch_size    = patch_size
        self.patch_shuffle = patch_shuffle
        self.pixel_shuffle = pixel_shuffle
        # self.pixel_unshuffle_info = torch.argsort(self.patch_info)
        # self.patch_unshuffle_info = torch.argsort(self.pixel_info)
    
    def shuffleConv2d(self, conv_weight):
        pixel_unshuffler = ops.ImageShuffle(patch_size=(self.patch_size, self.patch_size),
                              patch_shuffle=None,
                              pixel_shuffle=self.pixel_shuffle)
        weight_shuffled = pixel_unshuffler(conv_weight)
        return weight_shuffled
    
    def shuffle(self, model='mixer'):
        # Pixel Unshuffling
        if self.pixel_shuffle != None:
            conv_weight = self.new_weight['patch_embedding.embedding.0.weight'].clone()
            shuffled_weight = self.shuffleConv2d(conv_weight)
            self.new_weight['patch_embedding.embedding.0.weight'] = shuffled_weight
        if self.patch_shuffle != None:
            if 'mixer' in model:
                n_layers = 0
                for key in self.new_weight:
                    if 'mixer_blocks' in key:
                        n_layers = max(int(key.split('.')[1])+1, n_layers)
                for i in range(n_layers):
                    param = self.new_weight[f'mixer_blocks.{i}.token_mixer.2.net.0.weight'].clone()
                    shuffle_param = param[:, self.patch_shuffle]
                    self.new_weight[f'mixer_blocks.{i}.token_mixer.2.net.0.weight'] = shuffle_param

                    param = self.new_weight[f'mixer_blocks.{i}.token_mixer.2.net.3.weight'].clone()
                    shuffle_param = param[self.patch_shuffle, :]
                    self.new_weight[f'mixer_blocks.{i}.token_mixer.2.net.3.weight'] = shuffle_param

                    param = self.new_weight[f'mixer_blocks.{i}.token_mixer.2.net.3.bias'].clone()
                    shuffle_param = param[self.patch_shuffle]
                    self.new_weight[f'mixer_blocks.{i}.token_mixer.2.net.3.bias'] = shuffle_param
            elif 'vit' in model:
                if self.new_weight['fc.weight'].shape[1] == 8192:
                    stride = 128
                else:
                    stride = 64
                tmp = torch.zeros(self.new_weight['fc.weight'].shape[1])
                for idx, i in enumerate(self.patch_shuffle):
                    tmp[idx*stride:(idx+1)*stride] = torch.arange(stride)+i*stride
                tmp = tmp.long()
                shuffle_param = self.new_weight['fc.weight']
                self.new_weight['fc.weight'] = shuffle_param[:, tmp]
        return self.new_weight
    