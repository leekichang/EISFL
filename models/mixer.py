import torch
import torch.nn as nn

from models import nnops
# from models import nnops

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
    def __init__(self,
                 cfg):
        super().__init__()
        image_size  = cfg['image_size']
        patch_size  = cfg['patch_size']
        dim         = cfg['dim']
        num_layers  = cfg['num_layers']
        token_dim   = cfg['token_dim']
        channel_dim = cfg['channel_dim']
        num_class   = cfg['num_class']
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patch = (image_size//patch_size)**2
        self.patch_embedding = nnops.PatchEmbedding(dim, patch_size)
        self.mixer_blocks = nn.ModuleList([])
        
        for _ in range(num_layers):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(in_features=dim,
                                    out_features=num_class)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = mixer(dim=256, patch_size=4, num_class=10).to('cuda')
    summary(model, (3,32,32))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'# Total parmas: {total_params:,}')
    x = torch.randn(1,3,32,32).to('cuda')
    out = model(x)
    print(x.shape, out.shape)