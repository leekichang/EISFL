import torch
import numpy as np
import torch.autograd as autograd
import torch.multiprocessing as mp
import torch.utils.tensorboard as tb

import utils
from client import *
from server import *

if __name__ == '__main__':
    autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn')
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    server = Server(args)
    # server.prepare_dataset()
    print("SERVER READY")
    server.setup()
    print('===== ROUND 0 =====\nServer Setup Complete!')
    server.global_test()
    for i in range(server.rounds):
        print(f'===== ROUND {i+1} START! =====\n')
        server.train_federated_model()
        # sampled_clients = server.sample_clients()
        # print(f"CLIENTS {sampled_clients} ARE SELECTED!\n")
        # print("TEST WITH SP!\n")
        # server.test_selected_models(sampled_clients)
        server.global_test()
        if (i+1) % 10 == 0:
            server.save_model()