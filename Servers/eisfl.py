import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')
import torch
from tqdm import tqdm
import tensorboard as tb
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

import utils
import config as cfg
from models import *
from Clients import *
from DataManager import *
from Servers import BaseServer

__all__ = ['EISFLServer']

class EISFLServer(BaseServer):
    def __init__(self, args):
        super(EISFLServer, self).__init__(args)
        ### SHUFFLING INFO ###
        self.imsize          = cfg.IMGSIZE[self.args.dataset]
        self.patch_size      = cfg.PATCHSIZE[self.args.dataset]
        self.weight_shuffler = None
        ### Meta INFO ###
        self.cluster_history = -1*np.ones((self.n_clients, args.rounds))
        self.p_acc_history   = np.zeros((self.n_clients, args.rounds))
        self.user_sim        = torch.diag_embed(torch.ones(self.n_clients)).numpy()
        self.max_th  = -1

    def unshuffle_weight(self):
        '''
        Note that this is only for the evaluation.
        In real-world scenario, the server does not have access to the unshuffled weights.
        '''
        self.weight_shuffler.set_new_weight(self.global_model.state_dict())
        unshuffled_weight = self.weight_shuffler.unshuffle(model=self.args.model)
        self.global_model.load_state_dict(unshuffled_weight)

    def shuffle_weight(self):
        self.weight_shuffler.set_new_weight(self.global_model.state_dict())
        shuffled_weight = self.weight_shuffler.shuffle(model=self.args.model)
        self.global_model.load_state_dict(shuffled_weight)

    def integrity_check(self, sampled_clients):
        embedded_samples = []
        with torch.no_grad():
            for client in sampled_clients:
                self.clients[client].model.eval()
                output = self.clients[client].model(self.val_samples)
                embedded_samples.append(output.numpy())
        embedded_samples = torch.FloatTensor(np.array(embedded_samples))

        cosine_sim_mats = self.calc_cosine_sims(embedded_samples, len(sampled_clients))
        mean_cos_sim = np.mean(cosine_sim_mats.numpy(), axis=0)
        self.update_total_user_sim(mean_cos_sim, sampled_clients)
        return mean_cos_sim
    
    def calc_cosine_sims(self,embedded_samples, N):
        '''
        Calculate the cosine similarity between the samples.
        '''
        M = self.args.n_val * N
        cosine_sim_mats = torch.ones(M, N)
        cosine_sim_mats = torch.diag_embed(cosine_sim_mats)
        for sidx in range(M):
            for i in range(N):
                for j in range(i + 1, N):
                    sim_val = F.cosine_similarity(embedded_samples[i, sidx].unsqueeze(0), embedded_samples[j, sidx].unsqueeze(0))
                    cosine_sim_mats[sidx, i, j] = sim_val
                    cosine_sim_mats[sidx, j, i] = sim_val
        return cosine_sim_mats
    
    def update_total_user_sim(self, cosine_sim_mat, sampled_clients):
        '''
        Update the total user similarity matrix.
        '''
        N = cosine_sim_mat.shape[0]
        for i in range(N):
            for j in range(i+1, N):
                self.user_sim[sampled_clients[i], sampled_clients[j]] = cosine_sim_mat[i, j]
                self.user_sim[sampled_clients[j], sampled_clients[i]] = cosine_sim_mat[i, j]

    def calc_ious(self, similarity_matrix):
        N = similarity_matrix.shape[0]
        iou_mat = np.eye((N))
        for i in range(N):
            u = similarity_matrix[i]
            for j in range(i+1, N):
                v = similarity_matrix[j]
                intersection = np.logical_and(u, v)
                union = np.logical_or(u, v)
                iou = np.sum(intersection) / np.sum(union)
                iou_mat[i,j] = iou
                iou_mat[j,i] = iou
        return iou_mat
    
    def update_cluster(self, cos_sim, sampled_clients, threshold=0.5):
        '''
        Update the cluster based on the cosine similarity and existing clusters.
        
        Parameters:
        - cos_sim: torch.Tensor of shape (N, N) representing the cosine similarity matrix.
        - sampled_clients: List of clients sampled for clustering.
        - threshold: float representing the cosine similarity threshold for clustering.
        
        Returns:
        - Updated clusters as a torch.Tensor of shape (N,).
        '''
        clusters = np.array([self.user_cluster[cidx] for cidx in sampled_clients])
        N = len(sampled_clients)
        
        # Convert cosine similarity matrix to distance matrix
        distance = 1 - cos_sim  # Ensure conversion to NumPy array if needed
        is_similar = distance < (1 - threshold)
        
        # Compute IOU matrix and distance matrix
        iou_mat = self.calc_ious(is_similar)
        distance = 1 - iou_mat
        
        # DBSCAN clustering
        dbscan = DBSCAN(metric='precomputed', eps=0.5, min_samples=1)
        predicted_clusters = dbscan.fit_predict(distance)
        
        for idx, label in enumerate(predicted_clusters):
            print(f"User {sampled_clients[idx]} belongs to Cluster {label}")
        
        # Organize clients and clusters by predicted labels
        temp = {}
        for idx, label in enumerate(predicted_clusters):
            if label not in temp:
                temp[label] = {'clients': [], 'clusters': []}
            temp[label]['clients'].append(sampled_clients[idx])
            temp[label]['clusters'].append(clusters[idx])
        
        cluster_collapse = {}
        is_collapse = False
        
        for label, dicts in temp.items():
            dicts['clients'] = np.array(dicts['clients'])
            dicts['clusters'] = np.array(dicts['clusters'])
            
            if self.is_all_noob(dicts['clients']):
                for client in dicts['clients']:
                    self.user_cluster[client] = self.n_clusters
                self.n_clusters += 1
            else:
                noob_idx = self.find_noobs(dicts['clients'])
                min_cluster = np.min(dicts['clusters'][~noob_idx])  # Smallest cluster number
                if min_cluster not in cluster_collapse:
                    cluster_collapse[min_cluster] = [label]
                else:
                    is_collapse = True
                    cluster_collapse[min_cluster].append(label)
                for client in dicts['clients']:
                    self.user_cluster[client] = min_cluster
        
        if is_collapse:
            for cluster, labels in cluster_collapse.items():
                if len(labels) > 1:
                    counts = [np.sum(temp[label]['clusters'] == cluster) for label in labels]
                    max_idx = np.argmax(counts)
                    for idx, label in enumerate(labels):
                        if idx != max_idx:
                            for client in temp[label]['clients']:
                                self.user_cluster[client] = self.n_clusters
                            self.n_clusters += 1
                        else:
                            for client in temp[label]['clients']:
                                self.user_cluster[client] = cluster
        
    def clean_cluster(self):
        # Remove empty clusters for memory efficiency
        cluster_set = set([self.user_cluster[client] for client in range(self.n_clients)])
        for cluster in range(self.n_clusters): 
            if cluster not in cluster_set and cluster in self.global_models.keys():
                del self.global_models[cluster]
        print(f"model keys {list(self.global_models.keys())}")
        # 그 클러스터에 속한 유저가 없는데 서버는 모델을 들고 있을 이유가 없으니 모델도 삭제해준다
        # print(cluster_set)
        # print(f'N Clusters {len(list(self.global_models.keys()))}')

    def global_test(self, r):
        accs   = []
        losses = []
        total_benign = 0
        for cluster in self.global_models:
            b_user_num = 0
            atk_user_num = 0
            for client in range(int(self.n_clients)):
                if self.user_cluster[client] == cluster and client < int(self.n_clients*(1-self.args.atk_ratio)):
                    b_user_num += 1
                elif self.user_cluster[client] == cluster and client >= int(self.n_clients*(1-self.args.atk_ratio)):
                    atk_user_num += 1
            print(f"Cluster {cluster} has {b_user_num} Benign users and {atk_user_num} Attack users")
            if cluster != -1 and b_user_num > 0:
                self.global_model.load_state_dict(self.global_models[cluster].state_dict())
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
                for _ in range(b_user_num):
                    accs.append(100*correct/len(self.testset))
                    losses.append(loss/len(self.testloader))
            total_benign += b_user_num
        acc = np.mean(accs)
        loss = np.mean(losses)
        print(f'Round {r+1} Global Test Accuracy: {acc:.4f}, Loss: {loss:.4f}')
        self.tb_update(r, global_acc=acc, global_loss=loss)

    def global_test(self, r):
        accs   = []
        losses = []
        total_benign = 0
        n_test = 0
        for client in range(int(self.n_clients*(1-self.args.atk_ratio))):
            curr_client_cluster = self.user_cluster[client]
            if curr_client_cluster != -1:
                loss, acc = self.clients[client].test()
                temp = len(self.clients[client].testset)
                n_test += temp
                accs.append(acc*temp)
                losses.append(loss*temp)
        acc = np.sum(accs)/n_test
        loss = np.sum(losses)/n_test
        print(f'Round {r+1} Global Test Accuracy: {acc:.4f}, Loss: {loss:.4f}')
        self.tb_update(r, global_acc=acc, global_loss=loss)

    def run(self, save_period=None):
        threshold   = 0.25
        for round in range(self.args.rounds):
            sampled_clients = self.sample_clients(int(self.args.p_ratio*self.n_clients))
            self.val_samples = []
            self.val_labels  = []
            
            # accs, losses = [], []
            for client in sampled_clients:
                torch.manual_seed(round)
                self.clients[client].train()
                print(f'Client {client} is in cluster {self.user_cluster[client]} ({self.clients[client].n_samples})')
                # loss, acc = self.clients[client].test()
            #     accs.append(acc)
            #     losses.append(loss)
            # self.tb_update(round+1, p_acc_before_agg=np.mean(accs)   , p_loss_before_agg=np.mean(losses))
            # self.tb_update(round+1, p_acc_std_before_agg=np.std(accs), p_loss_std_before_agg=np.std(losses))

            # for client in sampled_clients:
            #     self.clients[client].shuffle_weight(round=round) # Clients shuffle the weight for the secure global aggregation
            #     val_sample, val_label = self.clients[client].upload_sample(n_samples=self.args.n_val)
            #     self.val_samples.append(val_sample)
            #     self.val_labels.append(val_label)
            val_samples = []
            val_labels  = []
            for idx in tqdm(sampled_clients):
                self.clients[idx].shuffle_weight(round=0)
                val_sample, val_label = self.clients[idx].upload_sample(n_samples=self.args.n_val)
                if self.args.dataset in ['CIFAR10'] and len(val_sample.shape)==3:
                    val_sample = val_sample.unsqueeze(0)
                val_samples.append(val_sample)
                val_labels.append(val_label)

            # self.val_samples = torch.cat(val_samples, dim=0).reshape(-1, 1, 28, 28)/255
            self.val_samples = torch.cat(val_samples, dim=0)
            self.val_labels  = torch.cat(val_labels, dim=0)
            
            weights = self.integrity_check(sampled_clients)
            self.update_cluster(weights, sampled_clients, threshold=threshold)
            self.aggregate(sampled_clients, weights=weights, args=args)
            self.dispatch()
            self.global_test(round)
            self.clean_cluster()
            # for c in self.global_models:
            #     if c != -1:
            #         self.global_model.load_state_dict(self.global_models[c].state_dict())
                    # print(f"CLUSTER {c}")
                    # self.global_test(round)
                    # if args.atk_type == 'Backdoor':
                    #     self.backdoor_test(round)
                    # print('#'*10)
            self.max_th = max(self.max_th, threshold)

            if len(self.global_models.keys()) >= 21:
                threshold = self.max_th
                self.max_th -= 0.0001
            else:
                threshold = min(0.85, threshold * 1.015)
            self.tb_update(round+1, threshold=threshold)

            # print(" ")
            # for client in sampled_clients:
            #     print(f'Client {client} is in cluster {self.user_cluster[client]}')
            #     self.clients[client].test()
            # print(" ")

            # self.val_samples = torch.concat(self.val_samples)/255 # TODO: implement data transform here
            # weights = self.integrity_check(sampled_clients)
            # if self.args.aggregator == 'WeightedSum':
            #     self.update_cluster(weights[:-1], sampled_clients, threshold=threshold)
            # # weights[-1] = torch.mean(weights, dim=0)
            # # print("AVG", weights)
            # # weights = F.softmax(weights, dim=1)
            # self.aggregate(sampled_clients, weights=weights, args=self.args) # Secure aggregation, as clients shuffled the weights

            # self.weight_shuffler = self.clients[client].weight_shuffler
            # self.unshuffle_weight() # Only for evaluation, In real-world scenario, clients unshuffles the weight

            # clusters = [self.user_cluster[cidx] for cidx in sampled_clients]
            # self.dispatch()         # Clients receive shuffled weights
            # self.clean_cluster()

            # accs, losses = [], []
            # for client in sampled_clients:
            #     loss, acc = self.clients[client].test()
            #     accs.append(acc)
            #     losses.append(loss)
            # self.tb_update(round+1, p_acc_after_agg    =np.mean(accs), p_loss_after_agg    =np.mean(losses))
            # self.tb_update(round+1, p_acc_std_after_agg=np.std(accs) , p_loss_std_after_agg=np.std(losses))

            # accs, losses = [], []
            # for client in range(self.n_clients):
            #     if client not in sampled_clients:
            #         loss, acc = self.clients[client].test()
            #         accs.append(acc)
            #         losses.append(loss)
            # self.tb_update(round+1, np_acc    =np.mean(accs), np_loss    =np.mean(losses))
            # self.tb_update(round+1, np_acc_std=np.std(accs) , np_loss_std=np.std(losses))
            
            

            # self.weight_shuffler = self.clients[client].weight_shuffler
            # self.unshuffle_weight() # Only for evaluation, In real-world scenario, clients unshuffles the weight
            # self.global_test(r=round)
            # self.benign_test(r=round)
            
            # if self.args.atk_type=='Backdoor':
            #     self.benign_backdoor_test(r=round)
            #     self.backdoor_test(r=round)
            # if save_period and (round+1) % save_period == 0:
            #     self.save_global_model(round+1)
            # threshold   = min(0.99, threshold*1.05)
            # for client in range(self.n_clients):
            #     self.cluster_history[client,round] = self.user_cluster[client]
            # utils.plot_cluster_history(self.cluster_history)
            # self.tb_update(round+1, threshold=threshold)



if __name__ == '__main__':     
    args = utils.parse_args()
    utils.print_info(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    np.set_printoptions(suppress=True, precision=6)
    server = EISFLServer(args)
    server.run()
    # sampled_clients = server.sample_clients(5)
    # for client in sampled_clients:
    #     # server.clients[client].unshuffle_weight(round=0)
    #     server.clients[client].train()
    #     server.clients[client].test()
    # for client in sampled_clients:
    #     server.clients[client].shuffle_weight(round=0)
    #     server.clients[client].upload_sample(n_samples=1)
    # server.integrity_check(sampled_clients)
    # server.aggregate(sampled_clients)
    # for client in sampled_clients:
    #     server.clients[client].shuffle_weight(round=0)
    #     server.clients[client].upload_sample(n_samples=1) # TODO: Implement sample upload for integrity check
    # server.aggregate(sampled_clients)
    # server.weight_shuffler = server.clients[client].weight_shuffler
    # server.weight_shuffler.new_weight = server.global_model.state_dict()
    # server.unshuffle_weight()
    # server.global_test()