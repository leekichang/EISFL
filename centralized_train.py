import torch

import utils
import Clients
from tqdm import tqdm
from DataManager import datamanager

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = utils.parse_args()
    # args.dataset = 'STL10'
    # args.model   = 'vit'
    args.epoch = 1
    torch.random.manual_seed(args.seed)
    trainer = Clients.NaiveClient(args, f'{args.exp_name}')
    trainset, testset = getattr(datamanager, args.dataset)()
    trainer.trainset = trainset
    trainer.testset = testset
    trainer.setup()
    
    for epoch in tqdm(range(50)):
        trainer.train()
        trainer.test()
    trainer.save_model(f'{args.seed}')