import sys
import torch
import numpy as np
sys.path.insert(0, '/disk2/Kichang/EISFL')

class FedAvg(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        new_state_dict = {}

        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for client_state_dict in client_state_dicts:
            for k in param_lists.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            mean_param = torch.tensor(np.mean(param_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = mean_param

        return new_state_dict

class Oracle(object): 
    def __init__(self):
        # An ideal aggregator that assumes full knowledge of whether a client is benign or malicious.
        pass

    def __call__(self, client_state_dicts, **kwargs):
        
        if 'sampled_clients' in kwargs:
            sampled_clients = kwargs['sampled_clients']
        if 'args' in kwargs:
            args = kwargs['args']
        
        N = args.n_clients - int(args.n_clients * args.atk_ratio)
        new_state_dict = {}
        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for idx, client_state_dict in enumerate(client_state_dicts):
            if sampled_clients[idx] < N:
                for k in param_lists.keys():
                    param_lists[k].append(client_state_dict[k].cpu().numpy())
            else:
                pass

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            mean_param = torch.tensor(np.mean(param_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = mean_param

        return new_state_dict

class Median(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        new_state_dict = {}

        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for client_state_dict in client_state_dicts:
            for k in param_lists.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            median_param = torch.tensor(np.median(param_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = median_param

        return new_state_dict

class TrimmedMean(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        if 'args' in kwargs:
            args = kwargs['args']
        if 'trim_fraction' in kwargs:
            trim_fraction = kwargs['trim_fraction']
        else:
            trim_fraction = 0.2
            
        new_state_dict = {}
        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for client_state_dict in client_state_dicts:
            for k in param_lists.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            n_trim = int(trim_fraction * param_array.shape[0])
            sorted_array = np.sort(param_array, axis=0)
            trimmed_array = sorted_array if n_trim == 0 else sorted_array[n_trim: -n_trim]
            trimmed_mean_param = torch.tensor(np.mean(trimmed_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = trimmed_mean_param
        return new_state_dict

class Krum(object):
    def __init__(self):
        pass
    
    def euclidean_distance(self, w1, w2):
        dist = 0
        for k in w1.keys():
            dist += np.linalg.norm(w1[k] - w2[k])
        return dist
    
    def __call__(self, client_state_dicts, **kwargs):
        if 'args' in kwargs:
            args = kwargs['args']
        if 'n_attackers' in kwargs:
            n_attackers = kwargs['n_attackers']
        else:
            n_attackers = 1

        num_clients = len(client_state_dicts)
        dist_matrix = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = self.euclidean_distance(client_state_dicts[i], client_state_dicts[j])
                dist_matrix[i,j]=dist
                dist_matrix[j,i]=dist
        min_sum_dist = float('inf')
        selected_index = -1
        for i in range(num_clients):
            sorted_indices = np.argsort(dist_matrix[i])
            sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - n_attackers)]]) # exclude itself
            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                selected_index = i
        return client_state_dicts[selected_index]
        

class MultiKrum(Krum):
    def __init__(self):
        pass
        self.TrimmedMean = TrimmedMean()
    
    def __call__(self, client_state_dicts, **kwargs):
        if 'args' in kwargs:
            args = kwargs['args']
        if 'n_attackers' in kwargs:
            n_attackers = kwargs['n_attackers']
        else:
            n_attackers = 1

        num_clients = len(client_state_dicts)
        dist_matrix = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = self.euclidean_distance(client_state_dicts[i], client_state_dicts[j])
                dist_matrix[i,j]=dist
                dist_matrix[j,i]=dist
        dists = []
        for i in range(num_clients):
            sorted_indices = np.argsort(dist_matrix[i])
            sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - n_attackers)]]) # exclude itself
            dists.append(sum_dist)
        selected_indices = np.argsort(dists)
        result = self.TrimmedMean([client_state_dicts[i] for i in selected_indices[:num_clients - n_attackers]], trim_fraction=0.2)
        return result

class WeightedSum(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        weights = torch.FloatTensor(kwargs['weights']) if kwargs['weights'] is not None else torch.FloatTensor([1/len(client_state_dicts)]*len(client_state_dicts))
        assert len(client_state_dicts) == weights.shape[0] # 맨마지막 하나는 전체 모델임.
        # 각 유저들의 모델의 파라미터를 weighted sum하는데, 얼마나 섞을지를 결정해두겠다.
        # 학습에 참여 안한 유저들의 모델 파라미터는? 그냥 평균으로 주나? 참여한 유저들 중에서 가장 유사한 애거 하나를 주나?
        # List to hold the new state_dicts for each client
        new_state_dicts = []

        # Iterate over each client's probability distribution
        for i, client_state_dict_ in enumerate(client_state_dicts):
            new_state_dict = {}
            param_lists = {k: [] for k in client_state_dict_.keys()}

            # Gather parameters for the current client
            for client_state_dict in client_state_dicts:
                for k in param_lists.keys():
                    param_lists[k].append(client_state_dict[k].cpu().numpy())

            # Perform the weighted sum for each parameter using the i-th probability distribution
            for k in param_lists.keys():
                param_array = np.array(param_lists[k])
                weighted_param = np.average(param_array, axis=0, weights=weights[i].cpu().numpy())
                new_state_dict[k] = torch.tensor(weighted_param, dtype=torch.float32)

            # Append the weighted state dict for this client
            new_state_dicts.append(new_state_dict)

        return new_state_dicts