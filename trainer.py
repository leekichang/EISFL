import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

import ops
import utils
import config as cfg

import torch.utils.tensorboard as tb

__all__ = ['SupervisedTrainer']

class SupervisedTrainer(object):
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu'

        self.imgsize      = cfg.IMGSIZE[self.args.dataset][2]
        self.patch_size   = args.ps
        self.model        = utils.build_model(args).to(self.device)
        self.criterion    = utils.build_criterion(args)
        self.optimizer    = utils.build_optimizer(self.model, args)
        
        self.trainset     = utils.load_dataset(args, is_train=True)
        self.testset      = utils.load_dataset(args, is_train=False)

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True , drop_last=True )
        self.test_loader  = DataLoader(self.testset , batch_size=args.batch_size, shuffle=False, drop_last=False)
    
        self.train_loss = []
        self.test_loss  = []
        self.accs       = []
        self.acc5s      = []
        
        patch_shuffle_info = torch.randperm((self.imgsize//self.patch_size)**2) if args.shuffle != 0 else None
        pixel_shuffle_info = torch.randperm(self.patch_size**2) if args.shuffle != 0 else None
        
        print(patch_shuffle_info, pixel_shuffle_info)

        with open(file=f'{self.save_path}/patch_shuffle_info.pickle', mode='wb') as f:
            pickle.dump(patch_shuffle_info, f)
        with open(file=f'{self.save_path}/pixel_shuffle_info.pickle', mode='wb') as f:
            pickle.dump(pixel_shuffle_info, f)
        
        self.shuffler = ops.ImageShuffle(patch_size=(self.patch_size,self.patch_size),
                                               patch_shuffle=patch_shuffle_info,
                                               pixel_shuffle=pixel_shuffle_info)
        self.use_tb     = args.use_tb
        
        self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}') \
                            if self.use_tb else None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'exp name:{args.exp_name}\nmodel name:{args.model}-{args.mcfg}\ndataset:{args.dataset}\ndevice:{self.device}\nTotal parameter:{total_params:,}')

    def train(self):
        self.model.train()
        losses = []
        for X, Y in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            X, Y = X.to(self.device), Y.to(self.device)
            X    = self.shuffler(X)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.train_loss.append(np.mean(losses))
        if self.use_tb:
            self.TB_WRITER.add_scalar("Train Loss", np.mean(losses), self.epoch+1)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        
        preds, targets, losses = [], [], []
        for X, Y in self.test_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            X    = self.shuffler(X)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            
            preds.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
            
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        acc  = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        if cfg.N_CLASS[self.args.dataset] >= 3:
            acc5 = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=3)
        else:
            acc5 = 0
            
        self.test_loss.append(np.mean(losses))
        self.accs.append(acc)
        self.acc5s.append(acc5)
        
        if self.use_tb:
            self.TB_WRITER.add_scalar(f'Test Loss', np.mean(losses), self.epoch+1)
            self.TB_WRITER.add_scalar(f'Test Accuracy', acc, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Top-3 Test Accuracy', acc5, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss[self.epoch]:>6.4f} Test Loss:{self.test_loss[self.epoch]:>6.4f} Test Accuracy:{self.accs[self.epoch]:>5.2f}% Top-3 Test Accuracy:{self.acc5s[self.epoch]:>5.2f}%', flush=True)
        
if __name__ == '__main__':
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = SupervisedTrainer(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        if (trainer.epoch+1)%10 == 0:
            trainer.save_model()
        trainer.epoch += 1