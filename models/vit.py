"""
Created on Thu Sep 06 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import torch.nn as nn
# import nnops
from models import nnops

__all__ = ['vit']

class vit(nn.Module):
    def __init__(self, cfg):
        super(vit, self).__init__()
        image_size  = cfg['image_size']
        patch_size  = cfg['patch_size']
        num_layers  = cfg['num_layers']
        num_class   = cfg['num_class']
        hidden_dim  = cfg['hidden_dim']
        dim         = cfg['dim']
        num_heads   = cfg['num_heads']
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.patch_embedding = nnops.PatchEmbedding(dim, patch_size)
        encoder = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim,)
        self.transformer = nn.TransformerEncoder(encoder, num_layers)
        self.fc = nn.Linear(dim * self.num_patches, num_class)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = x.reshape(B, -1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    import torch
    # from torchsummary import summary
    model = vit().to('cuda')
    print(model)
    # summary(model, (3,32,32))
    x = torch.rand(2,3,32,32).to('cuda')
    out = model(x)
    print(x.shape, out.shape)