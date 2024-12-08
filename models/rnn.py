import torch
import torch.nn
import config as cfg

__all__ = ['RNN','RNNWithMLP']

class RNN(torch.nn.Module):
    def __init__(self, n_class=6):
        super(RNN, self).__init__()
        self.hidden_size = 128
        self.num_layers = 4
        self.rnn = torch.nn.RNN(cfg.IMGSIZE['MITBIH'][-1], self.hidden_size, self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, n_class)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

#TODO: Implement RNN with MLPs
class RNNWithMLP(torch.nn.Module):
    def __init__(self, n_class=6):
        super(RNNWithMLP, self).__init__()
        self.hidden_size = 128
        self.num_layers = 1
        self.input_size = 30
        self.rnn_weights = [torch.nn.Linear(self.input_size if i == 0 else self.hidden_size, self.hidden_size).cuda() for i in range(self.num_layers)]
        self.rnn_biases = [torch.nn.Parameter(torch.zeros(self.hidden_size)).cuda() for _ in range(self.num_layers)]
        self.fc = torch.nn.Linear(self.hidden_size, n_class)

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.hidden_size).cuda() for _ in range(self.num_layers)]
        for t in range(x.size(1)):
            for i in range(self.num_layers):
                input_data = x[:, t, :] if i == 0 else h[i-1]
                h[i] = torch.tanh(self.rnn_weights[i](input_data) + self.rnn_biases[i])
        out = self.fc(h[-1])
        return out

if __name__ == '__main__':
    model = RNNWithMLP(n_class=4)
    model.to('cuda')
    x = torch.randn(30, 30, 60).to('cuda')
    o = model(x)
    print(x.shape, o.shape)