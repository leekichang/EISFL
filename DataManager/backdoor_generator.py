try:
    import config as cfg
except:
    import sys
    sys.path.insert(0, '/disk2/Kichang/EISFL')
    import config as cfg

class BackdoorGenerator(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
    
    def MNIST(self, dataset, **kwargs):
        N_CLASS    = cfg.N_CLASS[self.dataset]
        option     = kwargs['option'] if 'option' in kwargs else 'left'
        tar        = kwargs['tar']   if 'tar' in kwargs else 0
        proportion = kwargs['proportion'] if 'proportion' in kwargs else 0.5

        assert tar < N_CLASS and 0 <= tar, f'Target class should be between 0-{N_CLASS-1}'
        data, target = dataset.data, dataset.targets
        N = int(len(target)*proportion) # use just half of the dataset
        if option == 'left':
            data[:N, 0, 0:4] = 255
            data[:N, 2, 0:4] = 255
            data[:N, 0, 6:10] = 255
            data[:N, 2, 6:10] = 255
            target[:N] = tar
        elif option == 'center':
            data[:N, 14:18, 14:18] = 255
            target[:N] = tar
        elif option == 'right':
            data[:N, 24:28, 24:28] = 255
            target[:N] = tar
        else:
            raise ValueError(f'Invalid option: {option}')
        return data, target
    
    def EMNIST(self, dataset, **kwargs):
        return self.MNIST(dataset, **kwargs)

    def FashionMNIST(self, dataset, **kwargs):
        return self.MNIST(dataset, **kwargs)

    def CIFAR10(self, dataset, **kwargs):
        return 'CIFAR10'

    def CIFAR100(self, dataset, **kwargs):
        return 'CIFAR100'

    def __call__(self, dataset, **kwargs):
        return getattr(self, self.dataset)(dataset, **kwargs)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/disk2/Kichang/EISFL')
    import cv2
    import datamanager as dm
    import utils
    args = utils.parse_args()
    backdoor = BackdoorGenerator(args)
    train, test = getattr(dm, args.dataset)()
    cv2.imwrite('original.png', train.data[0].numpy())
    data, target = backdoor(train, option='left')
    cv2.imwrite('backdoor.png', data[0].numpy())
    
    