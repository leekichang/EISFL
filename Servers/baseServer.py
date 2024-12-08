import os
import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import copy
import torch
from datetime import datetime
from tqdm import tqdm
import torch.utils.tensorboard as tb

import ops
import utils
from models import *
from Clients import *
from DataManager import *
from Servers import aggregator

__all__ = ['BaseServer']

class BaseServer(object):
    def __init__(self, args):
        self.args            = args
        self.exp_name        = utils.format_args(self.args)
        self.clients         = []
        self.global_model    = utils.build_model(self.args)
        self.last_model      = copy.deepcopy(self.global_model)
        self.device          = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aggregator      = getattr(aggregator, self.args.aggregator)()
        self.n_clients       = self.args.n_clients
        self.user_cluster    = {cidx: -1 for cidx in range(self.n_clients)}
        self.user_history    = {cidx: [] for cidx in range(self.n_clients)}
        self.global_models   = {-1: self.global_model}
        self.n_clusters      = 0
        self.criterion       = utils.build_criterion(args)
        self.backdoor        = backdoor_generator.BackdoorGenerator(args)
        self.round           = 0
        self.setup()
        
    def setup(self):
        print(f"Experiment: {self.exp_name}")
        self.use_tb    = self.args.use_tb
        self.save_path = f'./checkpoints/{self.exp_name}'
        
        self.prepare_dataset()
        self.init_clients()
        self.dispatch()
        self.TB_WRITER    = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.exp_name}') if self.use_tb else None
    
    def tb_update(self, round, **kwargs):
        if self.use_tb:
            for key, value in kwargs.items():
                self.TB_WRITER.add_scalar(key, value, round)

    def prepare_dataset(self):
        self.trainset, self.testset = getattr(datamanager, self.args.dataset)()
        self.client_trainsets, self.client_testsets = Dirichlet(self.trainset, 
                                                                self.testset,
                                                                self.n_clients,
                                                                self.args.alpha).split_dataset()
        
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        if self.args.atk_type=='Backdoor':    
            self.backdoor_set         = copy.deepcopy(self.testset)
            self.backdoor_set.data    = self.backdoor_set.data[self.backdoor_set.targets != self.args.backdoor_tar]
            self.backdoor_set.targets = self.backdoor_set.targets[self.backdoor_set.targets != self.args.backdoor_tar]
            self.backdoor_set.targets = self.args.backdoor_tar * torch.ones(self.backdoor_set.targets.shape)
            self.backdoor_set.data, self.backdoor_set.targets = self.backdoor(self.backdoor_set,
                                                                              option=self.args.backdoor_opt,
                                                                              target=self.args.backdoor_tar)
            self.backdoor_loader = torch.utils.data.DataLoader(self.backdoor_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        
    def init_clients(self):
        self.benign_clients = int(self.n_clients*(1-self.args.atk_ratio))
        print(f"Initializing {self.n_clients} clients")
        print(f"{self.n_clients-int(self.n_clients*self.args.atk_ratio)} Benign Clients")
        print(f"{int(self.n_clients*self.args.atk_ratio)} {self.args.atk_type} Attack Clients")

        for cidx in tqdm(range(self.benign_clients)):
            self.clients.append(self.create_client(cidx))
            self.clients[cidx].trainset = copy.deepcopy(self.client_trainsets[cidx])
            self.clients[cidx].testset  = copy.deepcopy(self.client_testsets[cidx])
            self.clients[cidx].setup()

        for cidx in tqdm(range(self.benign_clients, self.n_clients)):
            self.clients.append(self.create_client(cidx, is_attacker=True))
            self.clients[cidx].trainset = copy.deepcopy(self.client_trainsets[cidx])
            self.clients[cidx].testset  = copy.deepcopy(self.client_testsets[cidx])
            self.clients[cidx].setup()
            
    def create_client(self, client_id, is_attacker=False):
        if is_attacker:
            return getattr(Clients, self.args.atk_type)(self.args, client_id)
        else:
            return getattr(Clients, self.args.client)(self.args, client_id)
    
    def sample_clients(self, n_participants):
        sampled_clients_idx = np.random.choice(self.n_clients, n_participants, replace=False)
        sampled_clients_idx = np.sort(sampled_clients_idx)
        return sampled_clients_idx
    
    def dispatch(self): # 각 클라이언트에게 할당된 클러스터의 모델을 반환
        for cidx in range(self.n_clients):
            self.clients[cidx].model.load_state_dict(self.global_models[self.user_cluster[cidx]].state_dict())
    
    def aggregate(self, sampled_clients, **kwargs):
        weights = kwargs.get('weights', None)
        new_state_dict = self.aggregator([self.clients[cidx].model.state_dict() for cidx in sampled_clients],
                                         args=self.args,
                                         weights=weights,
                                         sampled_clients=sampled_clients,
                                         trim_fraction=self.args.trim_frac)
        if self.args.aggregator in ['FedAvg', 'Median', 'TrimmedMean', 'Krum', 'MultiKrum', 'Oracle']:
            self.global_models[-1].load_state_dict(new_state_dict)
            self.global_model.load_state_dict(new_state_dict)
        elif self.args.aggregator in ['WeightedSum']:
            # 클러스터링 결과에 따라, 해당 클러스터 내부에 있는 유저들끼리만 aggregate를 해주는 방식으로
            new_state_dict = np.array(new_state_dict)
            model_weights  = np.array([self.clients[c].model.state_dict() for c in sampled_clients])
            clusters = np.array([self.user_cluster[c] for c in sampled_clients])
            noob_idx = self.find_noobs(sampled_clients)
            set_clusters = set(clusters)
            weight_avger = aggregator.FedAvg() # 일단은 FedAvg 사용
            for cluster in set_clusters:
                if cluster not in self.global_models.keys(): # 새로운 클러스터가 생겼다면
                    self.global_models[cluster] = copy.deepcopy(self.global_models[-1]) # 새로운 모델 생성
                cluster_idx = (clusters == cluster)
                if np.sum(cluster_idx) > 1: # 해당 클러스터에 속한 유저가 한 명이 아니라면
                    if not self.is_all_noob(sampled_clients[cluster_idx]): # 일부 유저가 첫 라운드 경우
                        cluster_idx = cluster_idx&~noob_idx # 첫 라운드인 유저는 제외
                    cluster_model = weight_avger(model_weights[cluster_idx]) # 해당 클러스터에 속한 유저들의 모델 aggregate
                    self.global_models[cluster].load_state_dict(cluster_model) # 클러스터 모델 할당
                else:   # 클러스터에 모델이 하나인경우
                    self.global_models[cluster].load_state_dict(model_weights[clusters==cluster][0]) # 해당 클러스터에 속한 유저의 모델 할당
            
            for cidx in sampled_clients:
                self.user_history[cidx].append(self.round) # 유저 히스토리 업데이트

    def is_all_noob(self, sampled_clients):
        noob_idx = self.find_noobs(sampled_clients)
        return np.sum(noob_idx)==len(noob_idx)
    
    def find_noobs(self, clients): # 모든 참여자가 첫 라운드인지 확인
        noobs = []
        for client in clients:
            if len(self.user_history[client])>0:
                noobs.append(False)
            else:
                noobs.append(True)
        return np.array(noobs)
    
    def benign_test(self, r):
        accs, losses = [], []
        model = copy.deepcopy(self.global_model)
        for cidx in range(self.benign_clients):
            model.load_state_dict(self.clients[cidx].model.state_dict())
            correct, loss = 0, []
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == target).sum().item()
                    loss_ = self.criterion(outputs, target)
                    loss.append(loss_.item())
            accs.append(100*correct/len(self.testset))
            losses.append(np.mean(loss))
        model = model.to('cpu')
        print(f'ROUND:{r+1:>03} Benign Accuracy: {np.mean(accs):.2f}%')
        print(f'ROUND:{r+1:>03}     Benign Loss: {np.mean(losses):.4f}')
        self.tb_update(r+1, benign_acc=np.mean(accs), benign_loss=np.mean(losses))

    def benign_backdoor_test(self, r):
        accs, losses = [], []
        model = copy.deepcopy(self.global_model)
        for cidx in range(self.benign_clients):
            model.load_state_dict(self.clients[cidx].model.state_dict())
            correct, loss = 0, []
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                for data, target in self.backdoor_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == target).sum().item()
                    loss_ = self.criterion(outputs, target)
                    loss.append(loss_.item())
            accs.append(100*correct/len(self.backdoor_set))
            losses.append(np.mean(loss))
        model = model.to('cpu')
        print(f'ROUND:{r+1:>03} Benign Backdoor Accuracy: {np.mean(accs):.2f}%')
        print(f'ROUND:{r+1:>03}     Benign Backdoor Loss: {np.mean(losses):.4f}')
        self.tb_update(r+1, benign_backdoor_acc=np.mean(accs), benign_backdoor_loss=np.mean(losses))


    def global_test(self, r):
        correct, loss = 0, 0
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                loss_ = self.criterion(outputs, target)
                loss += loss_.item()
        print(f'ROUND:{r+1:>03} Global Accuracy: {100*correct/len(self.testset):.2f}%')
        print(f'ROUND:{r+1:>03}     Global Loss: {loss/len(self.testloader):.4f}')
        self.global_model = self.global_model.to('cpu')
        self.tb_update(r+1, global_acc=100*correct/len(self.testset), global_loss=loss/len(self.testloader))

    def backdoor_test(self, r):
        correct, loss = 0, 0
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
        with torch.no_grad():
            for data, target in self.backdoor_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                loss_ = self.criterion(outputs, target)
                loss += loss_.item()
        print(f'ROUND:{r+1:>03} Backdoor Accuracy: {100*correct/len(self.backdoor_set):.2f}%')
        print(f'ROUND:{r+1:>03}     Backdoor Loss: {loss/len(self.backdoor_loader):.4f}')
        self.global_model = self.global_model.to('cpu')
        self.tb_update(r+1, backdoor_acc=100*correct/len(self.backdoor_set), backdoor_loss=loss/len(self.backdoor_loader))

    def run(self, save_period=None):
        for round in range(self.args.rounds):
            sampled_clients = self.sample_clients(int(self.args.p_ratio*self.n_clients))
            # accs, losses = [], []
            for client in sampled_clients:
                self.clients[client].train()
                loss, acc = self.clients[client].test()
            #     accs.append(acc)
            #     losses.append(loss)
            # self.tb_update(round+1, p_acc_before_agg=np.mean(accs)    , p_loss_before_agg=np.mean(losses))
            # self.tb_update(round+1, p_acc_std_before_agg=np.std(accs) , p_loss_std_before_agg=np.std(losses))
            self.aggregate(sampled_clients)
            
            # accs, losses = [], []
            # for client in sampled_clients:
            #     loss, acc = self.clients[client].test()
            #     accs.append(acc)
            #     losses.append(loss)
            # self.tb_update(round+1, p_acc_after_agg=np.mean(accs)   , p_loss_after_agg=np.mean(losses))
            # self.tb_update(round+1, p_acc_std_after_agg=np.std(accs), p_loss_std_after_agg=np.std(losses))
            
            # accs, losses = [], []
            # for client in range(self.n_clients):
            #     if client not in sampled_clients:
            #         loss, acc = self.clients[client].test()
            #         accs.append(acc)
            #         losses.append(loss)
            # self.tb_update(round+1, np_acc    =np.mean(accs), np_loss    =np.mean(losses))
            # self.tb_update(round+1, np_acc_std=np.std(accs) , np_loss_std=np.std(losses))

            self.global_test(r=round)
            if self.args.atk_type=='Backdoor':
                self.backdoor_test(r=round)
            if save_period and (round+1) % save_period == 0:
                self.save_global_model(round+1)
            self.dispatch()
            self.round += 1 # TODO: FIX THIS
            print(f"####### ROUND {round+1} END #######\n")

    def client_test(self, client_id):
        return self.clients[client_id].test()
    
    def save_global_model(self, round):
        torch.save(self.global_model.state_dict(), f'{self.save_path }/global_{round}.pth')

if __name__ == '__main__':    
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    server = BaseServer(args)
    server.run(save_period=False)