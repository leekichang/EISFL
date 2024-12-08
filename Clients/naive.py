import os
import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import copy
import torch
import torch.nn as nn


import utils
import Clients
import Defenses
from DataManager import datamanager

class NaiveClient(Clients.BaseClient):
    def __init__(self, args, name):
        super(NaiveClient, self).__init__(args, name)
        self.tag  = 'Benign Client'

    def setup(self):
        self.model       = utils.build_model(self.args)
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lr*0.1)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True , drop_last=True )
        self.testloader  = torch.utils.data.DataLoader(self.testset , batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        self.n_samples   = len(self.trainset)
        self.epoch       = self.args.epoch
        self.defense     = getattr(Defenses, self.args.defense)(self.args)

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epoch):
            for idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.defense(self.model)
                self.optimizer.step()
        self.model = self.model.to('cpu')

    def test(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            correct, total, loss = 0, 0, 0
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                acc = 100*correct/total
                loss += self.criterion(outputs, target).item()
            print(f'{self.tag} {self.name:<3} Accuracy: {acc:.2f}%')
        self.model = self.model.to('cpu')
        return loss/len(self.testloader), acc
    
    def shuffle_weight(self, round):
        self.set_shuffle_rule(round)    # Shuffle with current shuffle information
        self.weight_shuffler.set_new_weight(self.model.state_dict())
        shuffled_weight = copy.deepcopy(self.weight_shuffler.shuffle(model=self.args.model))
        self.model.load_state_dict(copy.deepcopy(shuffled_weight))

    def unshuffle_weight(self, round):
        self.set_shuffle_rule(round)  # Unshuffle with previous shuffle information
        self.weight_shuffler.set_new_weight(self.model.state_dict())
        unshuffled_weight = copy.deepcopy(self.weight_shuffler.unshuffle(model=self.args.model))
        self.model.load_state_dict(copy.deepcopy(unshuffled_weight))

    def upload_sample(self, n_samples=1):
        label_counts = torch.bincount(self.trainset.targets)
        most_common_label = torch.argmax(label_counts)
        idx = torch.where(self.trainset.targets == most_common_label)[0]
        idx = torch.tensor(torch.randperm(len(idx))[:n_samples])
        # idx = torch.randint(0, len(self.trainset.data), (n_samples,))
        val_samples, val_label = self.trainset[idx]
        val_samples = self.data_shuffler(val_samples)
        # val_samples = self.data_shuffler(self.trainset.data[idx])
        # val_label   = self.trainset.targets[idx]
        #TODO: add more data augmentation methods for more privacy
        return val_samples, val_label
    
if __name__ == '__main__':
    import utils
    import Clients
    import DataManager as DM
    from tqdm import tqdm
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    client = Clients.BaseClient(args, 'Naive')
    client = NaiveClient(args, 'Naive')
    train, test = DM.MNIST()
    client.trainset, client.testset = train, test
    client.setup()
    # client.shuffle(round=0)
    client.train()
    client.test()
    client.save_model('test')