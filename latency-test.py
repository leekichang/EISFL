import time
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReLU, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x):
        return self.dropout(self.relu(self.fc(x)))

class MLP(nn.Module):
    def __init__(self, n_class=10, n_layer=1):
        super(MLP, self).__init__()
        self.fc1 = LinearReLU(784, 256)
        self.fc2_layers = nn.ModuleList([LinearReLU(256, 256) for _ in range(n_layer)])
        self.fc3 = nn.Linear(256, n_class)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # fc2_layers를 반복적으로 적용
        for layer in self.fc2_layers:
            x = layer(x)
            
        x = self.fc3(x)
        return x

def prune_gradients(model):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
        parameters = list(
            filter(lambda p: p.grad is not None, parameters))
    input_grads = [p.grad.data for p in parameters]
    threshold = [torch.quantile(torch.abs(input_grads[i]), 0.9) for i in range(len(input_grads))]
    for i, p in enumerate(model.parameters()):
        p.grad[torch.abs(p.grad) < threshold[i]] = 0

def sanitize(tensor):
    eps = 8
    delta = 0.1
    
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
    clip_norm = 1.5
    
    tensor_norm = torch.norm(tensor, p=2)
    if tensor_norm > clip_norm:
        tensor = tensor * (clip_norm / tensor_norm)
    noise = torch.normal(mean=0, std=sigma * clip_norm, size=tensor.shape).to(tensor.device)
    tensor = tensor + noise
    return tensor

def sanitize_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            sanitized_grad = sanitize(p.grad.data)
            p.grad.data = sanitized_grad

def nodefense(model, iteration=100, batch_size=32, num_batch=100):
    x = torch.randn(batch_size, 1, 28, 28)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model.to('cpu')
    proc_elapsed_times = []
    elapsed_times = []
    peak_mem_usages = []
    for _ in range(iteration):
        proc_start = time.process_time()
        start = time.time()
        for _ in range(num_batch):
            o = model(x)
            y = torch.LongTensor(torch.randint(0,9,(batch_size,)))
            loss = criterion(o, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        proc_end = time.process_time()
        end = time.time()
        proc_elapsed_times.append(proc_end-proc_start)
        elapsed_times.append(end-start)
    return elapsed_times, proc_elapsed_times, peak_mem_usages

def dp(model, iteration=100, batch_size=32, num_batch=100):
    x = torch.randn(batch_size, 1, 28, 28)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model.to('cpu')
    proc_elapsed_times = []
    elapsed_times = []
    peak_mem_usages = []
    for _ in range(iteration):
        proc_start = time.process_time()
        start = time.time()
        for _ in range(num_batch):
            o = model(x)
            y = torch.LongTensor(torch.randint(0,9,(batch_size,)))
            loss = criterion(o, y)
            optimizer.zero_grad()
            loss.backward()
            sanitize_gradients(model)
            optimizer.step()
        proc_end = time.process_time()
        end = time.time()
        proc_elapsed_times.append(proc_end-proc_start)
        elapsed_times.append(end-start)
    return elapsed_times, proc_elapsed_times, peak_mem_usages  

def gp(model, iteration=100, batch_size=32, num_batch=100):
    x = torch.randn(batch_size, 1, 28, 28)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model.to('cpu')
    proc_elapsed_times = []
    elapsed_times = []
    peak_mem_usages = []
    for _ in range(iteration):
        proc_start = time.process_time()
        start = time.time()
        # tracemalloc.start()
        for _ in range(num_batch):
            o = model(x)
            y = torch.LongTensor(torch.randint(0,9,(batch_size,)))
            loss = criterion(o, y)
            optimizer.zero_grad()
            loss.backward()
            prune_gradients(model)
            optimizer.step()
        proc_end = time.process_time()
        end = time.time()
        proc_elapsed_times.append(proc_end-proc_start)
        elapsed_times.append(end-start)
        # peak_mem_usages.append(peak)
    return elapsed_times, proc_elapsed_times, peak_mem_usages 

def eisfl(model, iteration=100, batch_size=32, num_batch=100):
    x = torch.randn(batch_size, 1, 28, 28)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    model.to('cpu')
    proc_elapsed_times = []
    elapsed_times = []
    peak_mem_usages = []
    for _ in range(iteration):
        proc_start = time.process_time()
        start = time.time()
        for _ in range(num_batch):
            o = model(x)
            y = torch.LongTensor(torch.randint(0,9,(batch_size,)))
            loss = criterion(o, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        shuffled_weights = model.state_dict()
        shuffle_idx = torch.randperm(784)
        shuffled_weights['fc1.fc.weight'] = shuffled_weights['fc1.fc.weight'][:, shuffle_idx]
        model.load_state_dict(shuffled_weights, strict=False)
        proc_end = time.process_time()
        end = time.time()
        proc_elapsed_times.append(proc_end-proc_start)
        elapsed_times.append(end-start)
    return elapsed_times, proc_elapsed_times, peak_mem_usages         

model = MLP()
N_UPDATES = [1, 5, 10, 25, 50, 100]
nodef_lst = []
gp_lst = []
dp_lst = []
eisfl_lst = []
proc_nodef_lst = []
proc_gp_lst = []
proc_dp_lst = []
proc_eisfl_lst = []
ITER = 10
for n in N_UPDATES:
    print('Number of updates:', n)
    elapsed_times, proc_elapsed_times, peak_mems = nodefense(model, iteration=ITER, num_batch=n)
    proc_nodef_lst.append(proc_elapsed_times)
    nodef_lst.append(elapsed_times)
    print('Nodefense: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    elapsed_times, proc_elapsed_times, peak_mems = gp(model, iteration=ITER, num_batch=n)
    proc_gp_lst.append(proc_elapsed_times)
    gp_lst.append(elapsed_times)
    print('Gradient Pruning: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    elapsed_times, proc_elapsed_times, peak_mems = dp(model, iteration=ITER, num_batch=n)
    proc_dp_lst.append(proc_elapsed_times)
    dp_lst.append(elapsed_times)
    print('Differential Privacy: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    elapsed_times, proc_elapsed_times, peak_mems = eisfl(model, iteration=ITER, num_batch=n)
    proc_eisfl_lst.append(proc_elapsed_times)
    eisfl_lst.append(elapsed_times)
    print('EISFL: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    print("#"*50)
proc_result = {'Nodefense': proc_nodef_lst, 'Gradient Pruning': proc_gp_lst, 'Differential Privacy': proc_dp_lst, 'EISFL': proc_eisfl_lst}
result = {'Nodefense': nodef_lst, 'Gradient Pruning': gp_lst, 'Differential Privacy': dp_lst, 'EISFL': eisfl_lst}
with open('latency_n_update_proc.pkl', 'wb') as f:
    pickle.dump(proc_result, f)
with open('latency_n_update.pkl', 'wb') as f:
    pickle.dump(result, f)

N_UPDATES = 1
ITER = 20
n_layers = list(range(1, 11))
nodef_lst = []
gp_lst = []
dp_lst = []
eisfl_lst = []  
proc_nodef_lst = []
proc_gp_lst = []
proc_dp_lst = []
proc_eisfl_lst = []
for n in n_layers:
    model = MLP(n_layer=n)
    print('Number of layers:', n)
    elapsed_times, proc_elapsed_times, peak_mems = nodefense(model, iteration=ITER, num_batch=N_UPDATES)
    proc_nodef_lst.append(proc_elapsed_times)
    nodef_lst.append(elapsed_times)
    print('Nodefense: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    # print('Peak memory usage: {:.3f}MB'.format(sum(peak_mems)/len(peak_mems)/1024/1024))
    elapsed_times, proc_elapsed_times, peak_mems = gp(model, iteration=ITER, num_batch=N_UPDATES)
    proc_gp_lst.append(proc_elapsed_times)
    gp_lst.append(elapsed_times)
    print('Gradient Pruning: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    # print('Peak memory usage: {:.3f}MB'.format(sum(peak_mems)/len(peak_mems)/1024/1024))
    elapsed_times, proc_elapsed_times, peak_mems = dp(model, iteration=ITER, num_batch=N_UPDATES)
    proc_dp_lst.append(proc_elapsed_times)
    dp_lst.append(elapsed_times)
    print('Differential Privacy: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    elapsed_times, proc_elapsed_times, peak_mems = eisfl(model, iteration=ITER, num_batch=N_UPDATES)
    proc_eisfl_lst.append(proc_elapsed_times)
    eisfl_lst.append(elapsed_times)
    print('EISFL: {:.3f}ms / {:.3f}ms (proc)'.format(sum(elapsed_times)/len(elapsed_times)*1000, sum(proc_elapsed_times)/len(proc_elapsed_times)*1000))
    print("#"*50)

proc_result = {'Nodefense': proc_nodef_lst, 'Gradient Pruning': proc_gp_lst, 'Differential Privacy': proc_dp_lst, 'EISFL': proc_eisfl_lst}
result = {'Nodefense': nodef_lst, 'Gradient Pruning': gp_lst, 'Differential Privacy': dp_lst, 'EISFL': eisfl_lst}
with open('latency_depth_proc.pkl', 'wb') as f:
    pickle.dump(proc_result, f)
with open('latency_depth.pkl', 'wb') as f:
    pickle.dump(result, f)