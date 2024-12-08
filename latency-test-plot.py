import matplotlib.pyplot as plt
import pickle
import numpy as np

result_dir = "results/embedded/"
devices = [result_dir + "jetson_nano/", 
           result_dir + "jetson_tx2/",]
            #result_dir + "raspberry_pi_3b/"]
N_UPDATES = [1, 5, 10, 25, 50, 100]
N_LAYERS = list(range(1, 11))
num_figure = 0
configs = {"Update Time": {"files": ["latency_n_update.pkl", "latency_n_update_proc.pkl"],  "xlabel": "Number of Updates",  "ylabel": ["Relative Time (ms)", "Relative Proc Time (ms)"], "xbar": N_UPDATES},
           "Depth Time":  {"files": ["latency_depth.pkl", "latency_depth_proc.pkl"],        "xlabel": "Number of Layers",   "ylabel": ["Relative Time (ms)", "Relative Proc Time (ms)"], "xbar": N_LAYERS},}
           #"update": {"files": ["latency_n_update.pkl", "latency_n_update_proc.pkl"], "xlabel": "Number of Updates", "ylabel": ["Relative Time (ms)", "Relative Proc Time (ms)"]},}

for device in devices:
    for exp, exp_config in configs.items():
        for i in range(len(exp_config["files"])):
            file_name = device + exp_config["files"][i]
            with open(file_name, 'rb') as f:
                result = pickle.load(f)
            nodef_lst = result['Nodefense']
            gp_lst    = result['Gradient Pruning']
            dp_lst    = result['Differential Privacy']
            eisfl_lst = result['EISFL']
            nodef_lst = np.mean(nodef_lst, axis=1)
            gp_lst    = np.mean(gp_lst, axis=1)
            dp_lst    = np.mean(dp_lst, axis=1)
            eisfl_lst = np.mean(eisfl_lst, axis=1)

            plt.figure(num_figure)
            num_figure += 1

            plt.title(exp)
            plt.plot(exp_config["xbar"], np.array(nodef_lst)/nodef_lst[0], '^-', label='Nodefense')
            plt.plot(exp_config["xbar"], np.array(gp_lst)/nodef_lst[0]   , '^-', label='Gradient Pruning')
            plt.plot(exp_config["xbar"], np.array(dp_lst)/nodef_lst[0]   , '^-', label='Differential Privacy')
            plt.plot(exp_config["xbar"], np.array(eisfl_lst)/nodef_lst[0], '^-', label='EISFL')
            plt.xlabel(exp_config["xlabel"])
            plt.ylabel(exp_config["ylabel"])
            plt.legend()
            
plt.show()