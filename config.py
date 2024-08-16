N_CLASS = {
    'cifar10' :10,
    'cifar100':100,
    'TinyImageNet':200,
    'STL10':10,
    'Celeba':2,
}

IMGSIZE = {
    'cifar10' : (1,3,32,32),
    'cifar100': (1,3,32,32),
    'TinyImageNet':(1,3,64,64),
    'STL10':(1,3,96,96),
    'Celeba':(1,3,224,224),
}

PATCHSIZE = {
    'cifar10' : 4,
    'cifar100': 4,
    'TinyImageNet':8,
    'STL10':12,
    'Celeba':16,
}