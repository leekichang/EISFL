N_CLASS = {
    'cifar10' :10,
    'CIFAR10' :10,
    'cifar100':100,
    'TinyImageNet':200,
    'STL10':10,
    'Celeba':2,
    'MNIST':10,
    'FashionMNIST':10,
    'EMNIST':47,
    'MITBIH':4,
}

IMGSIZE = {
    'MNIST' : (1,28,28),
    'FashionMNIST' : (1,28,28),
    'EMNIST' : (1,28,28),
    'cifar10' : (1,3,32,32),
    'CIFAR10' : (1,3,32,32),
    'cifar100': (1,3,32,32),
    'TinyImageNet':(1,3,64,64),
    'STL10':(1,3,96,96),
    'Celeba':(1,3,224,224),
    'MITBIH':(1,30,60),
}

PATCHSIZE = {
    'MNIST' : 28,
    'FashionMNIST' : 28,
    'EMNIST' : 28,
    'cifar10' : 4,
    'CIFAR10' : 4,
    'cifar100': 4,
    'TinyImageNet':8,
    'STL10':4,
    'Celeba':16,
    'MITBIH':60,
}

