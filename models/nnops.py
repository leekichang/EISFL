import torch.nn as nn

__all__ = [
    'Rearrange',
    'PatchEmbedding',
    ]

class Rearrange(nn.Module):
    def __init__(self, dim1=1, dim2=2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class PatchEmbedding(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.embedding = nn.Sequential(
                        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size), # B,3,32,32 -> # B,C,S,S
                        nn.Flatten(2), # B,C,S,S -> B,C,S^2
                        Rearrange(1,2)) # B,C,S^2 -> B,S^2,C
    
    def forward(self, x):
        return self.embedding(x)