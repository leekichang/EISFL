import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['TwoCNN']

class TwoCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_class = cfg['num_class']
        self.do = nn.Dropout(0.2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2592, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=n_class)
            )

    def forward(self, x):
        x = self.convs(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = TwoCNN(10).to('cuda')
    summary(model, (3,32,32))
    x = torch.randn(1,3,32,32).to('cuda')
    out = model(x)