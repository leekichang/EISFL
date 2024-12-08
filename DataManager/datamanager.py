import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import config as cfg
import warnings
warnings.filterwarnings("ignore")

def MNIST():
    MNIST_train  = datasets.MNIST(root='../../../disk1/Kichang/dataset', train=True,  download=False, transform=transforms.ToTensor())
    MNIST_test   = datasets.MNIST(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transforms.ToTensor())
    return MNIST_train, MNIST_test

def EMNIST(split='balanced'):
    EMNIST_train = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split=split, train=True , download=True, transform=transforms.ToTensor())
    EMNIST_test  = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split=split, train=False, download=True, transform=transforms.ToTensor())
    return EMNIST_train, EMNIST_test

def FashionMNIST():
    FashionMNIST_train = datasets.FashionMNIST(root='../../../disk1/Kichang/dataset', train=True , download=False, transform=transforms.ToTensor())
    FashionMNIST_test  = datasets.FashionMNIST(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transforms.ToTensor())
    return FashionMNIST_train, FashionMNIST_test

def CIFAR10():
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]
    transform  = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),])
                                     #transforms.Normalize(mean, std, inplace=True)])
    CIFAR10_train = datasets.CIFAR10(root='../../../disk1/Kichang/dataset', train=True , download=False, transform=transform)
    transform  = transforms.Compose([transforms.ToTensor(),])
                                     # transforms.Normalize(mean, std, inplace=True)])
    CIFAR10_test  = datasets.CIFAR10(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transform)
    CIFAR10_train.targets = torch.LongTensor(CIFAR10_train.targets)
    CIFAR10_test.targets  = torch.LongTensor(CIFAR10_test.targets)
    return CIFAR10_train, CIFAR10_test

def CIFAR100():
    mean = [0.50707516, 0.48654887, 0.44091784]
    std  = [0.26733429, 0.25643846, 0.27615047]
    transform  = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std, inplace=True)])
    CIFAR100_train = datasets.CIFAR100(root='../../../disk1/Kichang/dataset', train=True , download=False, transform=transform)
    transform  = transforms.Compose([transforms.ToTensor(),])
                                     # transforms.Normalize(mean, std, inplace=True)])
    CIFAR100_test  = datasets.CIFAR100(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transform)
    CIFAR100_train.targets = torch.LongTensor(CIFAR100_train.targets)
    CIFAR100_test.targets  = torch.LongTensor(CIFAR100_test.targets)
    return CIFAR100_train, CIFAR100_test

class MITBIH_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        token_size = cfg.IMGSIZE['MITBIH'][-1]
        try:
            data = np.load(root + '/x_{}.npy'.format(split))
            label = np.load(root + '/y_{}.npy'.format(split))
        except:
            root_ = '../../../disk1/Kichang/dataset/mitbih_arr'
            data = np.load(root_ + '/x_{}.npy'.format(split))
            label = np.load(root_ + '/y_{}.npy'.format(split))
            data = data[~((label==5)|(label==3))]
            print(np.count_nonzero(label==0), np.count_nonzero(label==1), np.count_nonzero(label==2), np.count_nonzero(label==3), np.count_nonzero(label==4), np.count_nonzero(label==5))
            label = label[~((label==5)|(label==3))]
            label[label==4] = label[label==4] - 1
            np.save(root + '/x_{}.npy'.format(split), data)
            np.save(root + '/y_{}.npy'.format(split), label)
        self.data = torch.FloatTensor(data)[:,0].reshape(-1, 1800//token_size, token_size)
        self.targets = torch.LongTensor(label)
    
    def __len__(self):
        self.len = len(self.targets)
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def MITBIH():
    root = '../../../disk1/Kichang/dataset/mitbih_arr_new'
    import os
    os.makedirs(root, exist_ok=True)
    trainset = MITBIH_Dataset(root, 'train')
    testset = MITBIH_Dataset(root, 'test')
    print(trainset.data.shape, trainset.targets.shape)
    return trainset, testset

def STL10():
    transform  = transforms.Compose([
                                    transforms.RandomCrop(96, padding=4,padding_mode='reflect'),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])   
    # transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std, inplace=True)])
    STL10_train = datasets.STL10(root='../../../disk1/Kichang/dataset', split='train' , download=False, transform=transform)
    transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    STL10_test  = datasets.STL10(root='../../../disk1/Kichang/dataset', split='test'  , download=False, transform=transform)
    STL10_train.targets = STL10_train.labels
    STL10_test.targets  = STL10_test.labels
    return STL10_train, STL10_test

if __name__ == '__main__':
    # EMNIST_train = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split='balanced', train=True , download=True, transform=transforms.ToTensor())
    # EMNIST_test  = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split='balanced', train=False, download=True, transform=transforms.ToTensor())
    # print(max(EMNIST_train.targets))
    MITBIH()
