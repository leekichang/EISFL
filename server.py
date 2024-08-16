import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count
from collections import defaultdict, OrderedDict
import torchvision.transforms as transforms
import ops
import utils
import config as cfg
from client import *
from DPSGD import *
from GradPrune import *

import torch.utils.tensorboard as tb

__all__ = ['Server']

def mp_train_selected_clients(client):
    client.local_train()
    client.model.to('cpu')
    return client.model.state_dict()

class Server(object):
    def __init__(self, args):
        self.args            = args
        self.clients         = None
        self.device          = args.device
        self.defense_method  = args.defense
        print(f'DEFENSE METHOD: {self.defense_method}')
        
        self.global_model  = utils.build_model(args)
        self.global_model.to('cpu')
        self.criterion     = utils.build_criterion(args)
        self.testset       = utils.load_dataset(args, is_train=False)
        
        self.patch_size      = cfg.PATCHSIZE[args.dataset]
        self.imsize          = cfg.IMGSIZE[args.dataset][2]
        self.patch_shuffle   = None
        self.pixel_shuffle   = None
        self.weight_shuffler = ops.WeightShuffle(self.global_model.state_dict(), self.patch_size, None, None)
        
        self.total_user    = args.n_user
        self.user_ids      = [str(user) for user in range(self.total_user)]
        self.users         = {}
        self.user_trainset = {}
        self.user_testset  = {}
        self.mp_flag       = True
        self.lr            = self.args.lr
        self.lr_decay      = self.args.lrdecay
        self.round         = 0
        self.rounds        = args.R
        self.use_tb        = args.use_tb
        
        self.top_k         = 1 if args.dataset == 'Celeba' else 3
        self.TB_WRITER     = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{args.exp_name}') if self.use_tb else None
        
    def prepare_dataset(self, shard_per_user=50):
        if self.args.dataset in ['cifar10', 'cifar100', 'TinyImageNet', 'STL10', 'Celeba']:
            trainset = utils.load_dataset(self.args, is_train=True)
            n_class  = cfg.N_CLASS[self.args.dataset]
            train_idx_per_class = defaultdict(list)
            test_idx_per_class  = defaultdict(list)
            classes = np.array(trainset.targets)
            print(f'trainset.targets:{len(trainset.targets)}')
            print(f'self.testset.targets:{len(self.testset.targets)}')
            for idx, class_idx in enumerate(classes):
                train_idx_per_class[class_idx].append(idx)
            for idx, class_idx in enumerate(np.array(self.testset.targets)):
                test_idx_per_class[class_idx].append(idx)
            # for c in test_idx_per_class:
            #     print(c, len(test_idx_per_class[c]))
            print(f'train_idx_per_class[0]:{len(train_idx_per_class[0])}')
            print(f'test_idx_per_class[0]:{len(test_idx_per_class[0])}')
            
            train_idx_per_class = dict(train_idx_per_class)
            test_idx_per_class  = dict(test_idx_per_class)
            
            shard_size      = len(trainset)//(self.total_user*n_class)
            shard_per_class = len(trainset)//(n_class*shard_size)
            total_shards    = np.arange(int(len(trainset)//shard_size))
            
            print(f'shard_size:{shard_size}')
            print(f'shard_per_class:{shard_per_class}')
            print(f'total_shards:{len(total_shards)}')
            test_shard_size = len(self.testset)//(self.total_user*n_class)
            print(f'test_shard_size:{test_shard_size}')
            
            assert len(trainset) % (self.total_user*shard_size) == 0, "Error: len(trainset) % (self.total_user*shard_size) != 0"
            if self.args.iid:
                for idx, user in enumerate(range(self.total_user)):
                    X_train, Y_train, X_test, Y_test = [], [], [], []
                    for class_idx in range(n_class):
                        # class 하나씩 접근
                        data_idx   = idx*shard_size
                        train_idx  = train_idx_per_class[class_idx][data_idx:data_idx + shard_size]
                        X_train.append(trainset.data[train_idx])
                        Y_train.append(np.array(trainset.targets)[train_idx])
                        
                        data_idx = idx*test_shard_size
                        test_idx = test_idx_per_class[class_idx][data_idx:data_idx + test_shard_size]
                        X_test.append(self.testset.data[test_idx])
                        Y_test.append(np.array(self.testset.targets)[test_idx])
                        
                        # total_shards = np.delete(total_shards, np.where(total_shards==class_idx*shard_per_class+idx))
                        
                    X_train, Y_train, X_test, Y_test = np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(X_test), np.concatenate(Y_test)
                    # print(user, np.shape(X_train), np.shape(Y_train))
                    # print(user, np.shape(X_test), np.shape(Y_test))
                    if self.args.dataset == 'STL10':
                        X_train, X_test = X_train.transpose(0,2,3,1), X_test.transpose(0,2,3,1)
                    self.user_trainset[f'{user}'] = {'data':X_train, 'targets':Y_train, 'transform':trainset.transform}
                    self.testset.tranfrom = transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                                ])
                    self.user_testset[f'{user}']  = {'data': X_test, 'targets': Y_test, 'transform':self.testset.transform}
            else:
                for user in range(self.total_user):
                    selected_shard = np.random.choice(total_shards, shard_per_user, replace=False)
                    X_train, Y_train, X_test, Y_test = [], [], [], []
                    for idx in range(shard_per_user):
                        class_idx = selected_shard[idx]//shard_per_class
                        data_idx  = selected_shard[idx]%shard_per_class
                        train_idx  = train_idx_per_class[class_idx][data_idx:data_idx + shard_size]
                        X_train.append(trainset.data[train_idx])
                        Y_train.append(np.array(trainset.targets)[train_idx])
                        
                        test_idx  = test_idx_per_class[class_idx][data_idx:data_idx + test_shard_size]
                        X_test.append(self.testset.data[test_idx])
                        Y_test.append(np.array(self.testset.targets)[test_idx])
                        
                        total_shards = np.delete(total_shards, np.where(total_shards==selected_shard[idx]))
                        
                    X_train, Y_train, X_test, Y_test = np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(X_test), np.concatenate(Y_test)
                    self.user_trainset[f'{user}'] = {'data':X_train, 'targets':Y_train, 'transform':trainset.transform}
                    self.testset.tranfrom = transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                                ])
                    self.user_testset[f'{user}']  = {'data': X_test, 'targets': Y_test, 'transform':self.testset.transform}
        
        # elif self.args.dataset in ['STL10']:
        #     trainset = utils.load_dataset(self.args, is_train=True)
        #     n_class  = cfg.N_CLASS[self.args.dataset]
        #     train_idx_per_class = defaultdict(list)
        #     test_idx_per_class  = defaultdict(list)
        #     classes = np.array(trainset.labels)
        #     print(f'trainset.labels:{len(trainset.labels)}')
        #     print(f'self.testset.labels:{len(self.testset.labels)}')
        #     for idx, class_idx in enumerate(classes):
        #         train_idx_per_class[class_idx].append(idx)
        #     for idx, class_idx in enumerate(np.array(self.testset.labels)):
        #         test_idx_per_class[class_idx].append(idx)
        #     # for c in test_idx_per_class:
        #     #     print(c, len(test_idx_per_class[c]))
        #     print(f'train_idx_per_class[0]:{len(train_idx_per_class[0])}')
        #     print(f'test_idx_per_class[0]:{len(test_idx_per_class[0])}')
            
        #     train_idx_per_class = dict(train_idx_per_class)
        #     test_idx_per_class  = dict(test_idx_per_class)
            
        #     shard_size      = len(trainset)//(self.total_user*n_class)
        #     shard_per_class = len(trainset)//(n_class*shard_size)
        #     total_shards    = np.arange(int(len(trainset)//shard_size))
            
        #     print(f'shard_size:{shard_size}')
        #     print(f'shard_per_class:{shard_per_class}')
        #     print(f'total_shards:{len(total_shards)}')
        #     test_shard_size = len(self.testset)//(self.total_user*n_class)
        #     print(f'test_shard_size:{test_shard_size}')
            
        #     assert len(trainset) % (self.total_user*shard_size) == 0, "Error: len(trainset) % (self.total_user*shard_size) != 0"
        #     if self.args.iid:
        #         for idx, user in enumerate(range(self.total_user)):
        #             X_train, Y_train, X_test, Y_test = [], [], [], []
        #             for class_idx in range(n_class):
        #                 # class 하나씩 접근
        #                 data_idx   = idx*shard_size
        #                 train_idx  = train_idx_per_class[class_idx][data_idx:data_idx + shard_size]
        #                 X_train.append(trainset.data[train_idx])
        #                 Y_train.append(np.array(trainset.labels)[train_idx])
                        
        #                 data_idx = idx*test_shard_size
        #                 test_idx = test_idx_per_class[class_idx][data_idx:data_idx + test_shard_size]
        #                 X_test.append(self.testset.data[test_idx])
        #                 Y_test.append(np.array(self.testset.labels)[test_idx])
                        
        #                 # total_shards = np.delete(total_shards, np.where(total_shards==class_idx*shard_per_class+idx))
                        
        #             X_train, Y_train, X_test, Y_test = np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(X_test), np.concatenate(Y_test)
        #             self.user_trainset[f'{user}'] = {'data':X_train.transpose(0,2,3,1), 'targets':Y_train, 'transform':trainset.transform}
        #             self.testset.tranfrom = transforms.Compose([
        #                                                         transforms.ToTensor(),
        #                                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                                                         ])
        #             self.user_testset[f'{user}']  = {'data': X_test.transpose(0,2,3,1), 'targets': Y_test, 'transform':self.testset.transform}
        #     else:
        #         for user in range(self.total_user):
        #             selected_shard = np.random.choice(total_shards, shard_per_user, replace=False)
        #             X_train, Y_train, X_test, Y_test = [], [], [], []
        #             for idx in range(shard_per_user):
        #                 class_idx = selected_shard[idx]//shard_per_class
        #                 data_idx  = selected_shard[idx]%shard_per_class
        #                 train_idx  = train_idx_per_class[class_idx][data_idx:data_idx + shard_size]
        #                 X_train.append(trainset.data[train_idx])
        #                 Y_train.append(np.array(trainset.labels)[train_idx])
                        
        #                 test_idx  = test_idx_per_class[class_idx][data_idx:data_idx + test_shard_size]
        #                 X_test.append(self.testset.data[test_idx])
        #                 Y_test.append(np.array(self.testset.labels)[test_idx])
                        
        #                 total_shards = np.delete(total_shards, np.where(total_shards==selected_shard[idx]))
                        
        #             X_train, Y_train, X_test, Y_test = np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(X_test), np.concatenate(Y_test)
        #             self.user_trainset[f'{user}'] = {'data':X_train, 'targets':Y_train, 'transform':trainset.transform}
        #             self.testset.tranfrom = transforms.Compose([
        #                                                         transforms.ToTensor(),
        #                                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                                                         ])
        #             self.user_testset[f'{user}']  = {'data': X_test, 'targets': Y_test, 'transform':self.testset.transform}
                    
        else:
            print(f"UNKNOWN DATASET NAMED {self.args.dataset}")
        
    def create_clients(self):
        clients = {}
        for user in self.user_ids:
            if self.defense_method in ['no']:
                clients[user] = Client(args=self.args,
                                    client_id=user,
                                    trainset=self.user_trainset[str(user)],
                                    testset=self.user_testset[str(user)])
            elif self.defense_method in ['dpsgd']:
                clients[user] = DPSGDClient(args=self.args,
                                    client_id=user,
                                    trainset=self.user_trainset[str(user)],
                                    testset=self.user_testset[str(user)])
            elif self.defense_method in ['gradprune']:
                clients[user] = GradPruneClient(args=self.args,
                                    client_id=user,
                                    trainset=self.user_trainset[str(user)],
                                    testset=self.user_testset[str(user)])
        return clients
    
    def setup_clients(self):
        for client in tqdm(self.clients, leave=False):
            self.clients[client].setup()
    
    def setup(self):
        self.prepare_dataset()
        self.clients    = self.create_clients()
        self.dataloader = DataLoader(self.testset, batch_size=256, shuffle=False)
        self.transmit_model()
        self.setup_clients()
    
    def set_shuffle_rule(self):
        self.patch_shuffle = torch.randperm((self.imsize//self.patch_size)**2) if self.args.shuffle else None
        self.pixel_shuffle = torch.randperm(self.patch_size**2) if self.args.shuffle else None
    
    def set_weight_shuffler(self, weight, patch, pixel):
        self.weight_shuffler.new_weight    = copy.deepcopy(weight)
        self.weight_shuffler.patch_shuffle = patch
        self.weight_shuffler.pixel_shuffle = pixel
    
    def unshuffle_weight(self):
        weights = copy.deepcopy(self.global_model.state_dict())
        
        patch_unshuffle = torch.argsort(self.patch_shuffle) if self.args.shuffle else None
        pixel_unshuffle = torch.argsort(self.pixel_shuffle) if self.args.shuffle else None
        self.set_weight_shuffler(weights, patch_unshuffle, pixel_unshuffle)
        shuffled_weight = self.weight_shuffler.shuffle(self.args.model)
        self.global_model.load_state_dict(copy.deepcopy(shuffled_weight))
    
    def transmit_model(self, sampled_clients:list=None):
        self.set_shuffle_rule()
        self.global_model.to('cpu')
        if sampled_clients == None:   
            for client in tqdm(self.clients, leave=False):
                self.clients[client].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
                self.clients[client].shuffle(self.patch_shuffle, self.pixel_shuffle)
                self.clients[client].set_optim(self.lr)
        else:
            for client in tqdm(sampled_clients, leave=False):
                self.clients[client].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
                self.clients[client].shuffle(self.patch_shuffle, self.pixel_shuffle)
                self.clients[client].set_optim(self.lr)
    
    def sample_clients(self, n_participant:int=5):
        assert n_participant <= len(self.user_ids), "Check 'n_participant <= len(self.clients)'"
        selected_clients = np.random.choice(self.user_ids, n_participant, replace=False) 
        for client in selected_clients:
            self.clients[client].selected_rounds.append(self.round)
        return selected_clients

    def train_selected_clients(self, sampled_clients:list):
        total_sample = 0
        for client in tqdm(sampled_clients, leave=False):
            self.clients[client].local_train()
            total_sample += len(self.clients[client])
        return total_sample
    
    def test_selected_models(self, sampled_clients):
        for client in sampled_clients:
            self.clients[client].local_test()
    
    def average_model(self, sampled_clients, coefficients):
        averaged_weights = OrderedDict()
        for it, client in tqdm(enumerate(sampled_clients), leave=False):
            self.clients[client].model.to('cpu')
            local_weights = copy.deepcopy(self.clients[client].model.state_dict())
            for key in self.global_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.global_model.load_state_dict(copy.deepcopy(averaged_weights))
   
    def train_federated_model(self):
        sampled_clients = self.sample_clients()
        print(f"CLIENTS {sampled_clients} ARE SELECTED!\n")
        
        if self.mp_flag:
            print("TRAIN WITH MP!\n")
            selected_total_size = sum([len(self.clients[c]) for c in sampled_clients])
            clients2train = [self.clients[c] for c in sampled_clients]  
            num_process = min(len(sampled_clients), 3)
            with Pool(processes=num_process) as pool:
                return_c = pool.map(mp_train_selected_clients, clients2train)
            
            for c, c_copy in zip(sampled_clients, return_c):
                self.clients[c].model.load_state_dict(copy.deepcopy(c_copy))
        else:
            print("TRAIN WITH SP!\n")
            selected_total_size = self.train_selected_clients(sampled_clients)

        print("TEST WITH SP!\n")
        self.test_selected_models(sampled_clients)
                
        mixing_coefficients = [len(self.clients[client]) / selected_total_size for client in sampled_clients]
        
        
        self.average_model(sampled_clients, mixing_coefficients)
        self.unshuffle_weight()
        self.transmit_model()
        self.round += 1
        self.lr = max(self.lr*self.lr_decay, 8e-5)
        
    @torch.no_grad()
    def global_test(self):
        self.global_model.eval()
        self.global_model.to(self.device)
        preds, targets, losses = [], [], []
        for X, Y in self.dataloader:
            X, Y = X.to(self.device), Y.to(self.device)
            pred = self.global_model(X)
            loss = self.criterion(pred, Y)
            
            preds.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
        
        preds    = np.concatenate(preds)
        targets  = np.concatenate(targets)
        top1_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        topk_acc = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=self.top_k)
        if self.use_tb:
            self.TB_WRITER.add_scalar(f'Global Test Loss', np.mean(losses), self.round+1)
            self.TB_WRITER.add_scalar(f'Global Test Accuracy', top1_acc, self.round+1)
            self.TB_WRITER.add_scalar(f'Global Top-{self.top_k} Test Accuracy', topk_acc, self.round+1)
        print(f'Global Test Result | Top-1 Test Acc:{top1_acc:.2f} | Top-{self.top_k} Test Acc:{topk_acc:.2f} | Test Loss:{loss:.4f}')
        self.global_model.to('cpu')
        if "cuda" in self.device : torch.cuda.empty_cache()
        
    def save_model(self):
        model_save_path = f'./checkpoints/{self.args.exp_name}'
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        torch.save(self.global_model.state_dict(), f'{model_save_path}/{self.round}.pth')   
    
if __name__ == '__main__':
    args = utils.parse_args()
    server = Server(args)
    server.prepare_dataset()
    server.setup()
    dataset = utils.load_user_dataset(server.user_trainset['0'])
    
    dataloader = DataLoader(dataset, 10)
    for (data, target) in dataloader:
        print(data.shape, target.shape)
    