import torch
import torch.nn as nn

from models import nnops
# import nnops

__all__ = ["mixer"]

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

'''
Mixer block (Token mixer and Channel mixer)
'''
class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.0):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),  # B,S^2,C -> B,S^2,C
            nnops.Rearrange(1,2),     # B,S^2,C -> B,C,S^2
            FeedForward(num_patch, token_dim, dropout), # B,C,S^2 -> B,C,Ds -> B,C,S^2
            nnops.Rearrange(1,2)      # B,C,S^2 -> B,S^2,C
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim), # B,S^2,C -> B,S^2,C
            FeedForward(dim, channel_dim, dropout), # B,S^2,C -> # B,S^2,Dc -> B,S^2,C
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x

'''
MLP Mixer
https://arxiv.org/abs/2105.01601
'''
class mixer(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        image_size  = 32 # cfg['image_size']
        patch_size  = 4  # cfg['patch_size']
        dim         = 256 # cfg['dim']
        num_layers  = 4 # ['num_layers']
        token_dim   = 128 # ['token_dim']
        channel_dim = 128 # ['channel_dim']
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patch = (image_size//patch_size)**2
        self.patch_embedding = nnops.PatchEmbedding(dim, patch_size)
        self.mixer_blocks = nn.ModuleList([])
        
        for _ in range(num_layers):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(in_features=dim,
                                    out_features=n_class)
    
    def forward(self, x):
        # print(f'1: {x.shape}')
        x = self.patch_embedding(x)
        # print(f'2: {x.shape}')
        for bidx, mixer_block in enumerate(self.mixer_blocks):
            x = mixer_block(x)
            # print(f'{bidx+3}: {x.shape}')
        x = self.layer_norm(x)
        # print(f'7: {x.shape}')
        x = x.mean(dim=1)
        # print(f'8: {x.shape}')
        x = self.classifier(x)
        # print(f'9: {x.shape}')
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = mixer().to('cuda')
    summary(model, (3,32,32))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'# Total parmas: {total_params:,}')
    x = torch.randn(1,3,32,32).to('cuda')
    out = model(x)
    print(x.shape, out.shape)